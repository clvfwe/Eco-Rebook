# -*- coding: utf-8 -*-
"""에코리북 AI 선별·등급화 — 파일 하나로 실행 (웹페이지 + 서버 + 등급 규칙 + 히트맵 + 판정 기록)

준비:  이 파일과 같은 폴더에  best.pt (결함 탐지 모델),  type.pt (종이 종류 모델) 를 둡니다.
설치:  pip install flask ultralytics pillow
실행:  python ecorebook_app.py
접속:  http://localhost:8000     (인터넷 없이 이 PC에서만 동작)

AI는 '사실'만 기록하고, 등급은 '규칙'이 매깁니다.
  사실(AI)   : 종이 종류(낱장/제본), 결함 6종의 위치와 개수, 오염 면적
  규칙(RULES): 사실 → 등급(A/B/C/등급 외) · 전처리 후 예상 등급 · 전처리 지침
  → 구매처 기준이 바뀌면 모델을 다시 학습하지 않고 RULES만 고치면 됩니다.

구성 (위에서부터)
  1. 설정            ─ 모델 위치, 포트, 기록 저장 폴더
  2. 등급 규칙 RULES ─ 결함별 지침, 신뢰도 기준, 오염 면적 기준
  3. 등급 grade()    ─ A / B / C / 등급 외,  전처리 후 예상 등급
  4. 히트맵 GradCAM  ─ 판정에 기여한 위치를 빨갛게 표시
  5. 판정 기록       ─ records 폴더에 자동 저장, CSV로 내보내기
  6. 서버            ─ /  (웹페이지),  /predict (사진 분석),  /records (기록)
  7. 웹페이지 HTML   ─ 맨 아래 PAGE 안에 통째로 들어 있음
"""
import base64
import csv
import hashlib
import io
import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, Response, jsonify, request, send_file
from PIL import Image, ImageOps
from ultralytics import YOLO

# ───────────────────────── 1. 설정 ─────────────────────────
HERE = Path(__file__).resolve().parent
DET_CANDIDATES = [HERE / "best.pt", HERE / "deliver" / "model" / "best.pt", HERE.parent / "deliver" / "model" / "best.pt"]
TYPE_CANDIDATES = [HERE / "type.pt", HERE / "deliver" / "model" / "type.pt"]
IMGSZ = 800              # 판정할 때 사진을 줄이는 크기 (학습은 640, 작은 결함을 더 잘 보도록 판정은 800)
TTA = True               # 좌우 뒤집은 사진·크기를 바꾼 사진도 함께 보고 결과를 합침 (조금 느려지지만 놓치는 결함이 줄어듦)
CAM_IMGSZ = 640          # 히트맵을 계산하는 크기
PREDICT_CONF = 0.15      # 모델이 후보를 뽑는 기준 (최종 기준은 RULES의 신뢰도 기준)
HEATMAP = True           # 히트맵을 응답에 넣을지 (끄면 조금 더 빠름)
HEAT_FLOOR = 0.2         # 히트맵에서 이 값보다 약한 반응은 표시하지 않음 (0~1)
RECORD_DIR = HERE / "records"   # 판정 기록(사진·히트맵·records.db)이 쌓이는 폴더
HOST, PORT = "127.0.0.1", 8000

# 화면에 보여 줄 짧은 이름
SHORT = {"spring": "스프링", "staple": "스테이플", "tape": "테이프", "vinyl": "코팅", "stain": "오염", "pii": "개인정보"}
TYPE_KO = {"sheet": "낱장", "bound": "제본"}
GRADE_KO = {"A": "A등급", "B": "B등급", "C": "C등급", "X": "등급 외"}

# ───────────────────────── 2. 등급 규칙 ─────────────────────────
# kind  removable : 떼어내면 종이는 전부 쓸 수 있는 결함      → B등급
#       lossy     : 종이 일부를 덜어내야 하는 결함            → C등급
RULES = {
    "version": "2026-11-18",
    "confidence_threshold": 0.35,     # 이 값 미만의 탐지는 판정에 쓰지 않음
    "class_confidence": {},           # 결함별로 기준을 따로 두고 싶을 때  예) {"staple": 0.3}
    "review_below": 0.5,              # 이 값 미만의 탐지가 있으면 '사람이 한 번 더 확인' 표시
    "stain_reject_ratio": 0.30,       # 오염 면적이 사진의 30%를 넘으면 등급 외
    "coated_sheet_reject": True,      # 낱장 자체가 코팅된 경우(코팅지)는 종이만 분리할 수 없으므로 등급 외
    "coated_sheet_type_conf": 0.70,   # 위 규칙은 '낱장' 판단의 확신도가 이 값 이상일 때만 적용 (낮으면 C등급 + 확인 요청)
    "classes": {
        "spring": {
            "kind": "removable",
            "label_ko": "금속 스프링 제본",
            "instruction": "스프링을 분리하거나 제본부를 절단선 기준으로 잘라낸 뒤 투입하십시오."
        },
        "staple": {
            "kind": "removable",
            "label_ko": "스테이플러 심·금속 이물질",
            "instruction": "스테이플러 심을 제거하거나 제본부를 절단한 뒤 투입하십시오."
        },
        "tape": {
            "kind": "removable",
            "label_ko": "테이프·접착제 잔여물",
            "instruction": "테이프가 붙은 부분을 잘라내거나 해당 장을 분리한 뒤 투입하십시오."
        },
        "pii": {
            "kind": "removable",
            "label_ko": "개인정보 기재 영역",
            "instruction": "이름·학번 등 개인정보가 적힌 부분을 잘라 자체 파기한 뒤 나머지를 투입하십시오."
        },
        "vinyl": {
            "kind": "lossy",
            "label_ko": "비닐·필름 코팅",
            "instruction": "비닐 커버 또는 코팅된 표지를 분리하고 내지만 투입하십시오.",
            "reject_instruction": "종이 전체가 코팅되어 있어 종이만 분리할 수 없습니다. 원료로 쓰지 말고 따로 배출하십시오."
        },
        "stain": {
            "kind": "lossy",
            "label_ko": "오염 얼룩",
            "instruction": "오염된 장을 골라내고 깨끗한 장만 투입하십시오.",
            "reject_instruction": "오염 범위가 넓어 원료로 사용할 수 없습니다. 일반 폐지로 배출하십시오."
        }
    },
    "grades": {
        "A": "결함이 없어 바로 원료로 쓸 수 있습니다.",
        "B": "떼어낼 수 있는 결함만 있습니다. 전처리하면 전량 쓸 수 있습니다.",
        "C": "코팅 또는 부분 오염이 있습니다. 해당 부분을 덜어낸 나머지를 쓸 수 있습니다.",
        "X": "오염이 넓어 원료로 쓸 수 없습니다.",
        "X_coated": "낱장 전체가 코팅되어 있어 원료로 쓸 수 없습니다."
    }
}


# ───────────────────────── 3. 등급 ─────────────────────────
def union_ratio(boxes, w, h, n=200):
    """여러 네모가 겹쳐도 한 번만 세어 사진 대비 면적 비율을 구함"""
    if not boxes or w <= 0 or h <= 0:
        return 0.0
    grid = np.zeros((n, n), dtype=bool)
    for x1, y1, x2, y2 in boxes:
        a, b = int(np.floor(x1 / w * n)), int(np.floor(y1 / h * n))
        c, d = int(np.ceil(x2 / w * n)), int(np.ceil(y2 / h * n))
        grid[max(b, 0):min(d, n), max(a, 0):min(c, n)] = True
    return float(grid.mean())


def grade(detections, rules, img_w, img_h, paper=None, paper_conf=None):
    """탐지 결과(사실) + 규칙 → 등급, 전처리 후 예상 등급, 결함 목록, 오염 면적 비율, 등급 설명, 확인 요청 여부"""
    findings, kinds, stain_boxes = [], set(), []
    for d in detections:
        rule = rules["classes"].get(d["class"])
        need = rules["class_confidence"].get(d["class"], rules["confidence_threshold"])
        if rule is None or d["confidence"] < need:
            continue
        findings.append({
            "type": d["class"],
            "label": rule["label_ko"],
            "kind": rule["kind"],
            "confidence": round(d["confidence"], 3),
            "box_xyxy": d["box"],
            "instruction": rule["instruction"],
        })
        kinds.add(rule["kind"])
        if d["class"] == "stain":
            stain_boxes.append(d["box"])

    review = any(f["confidence"] < rules["review_below"] for f in findings)
    stain_ratio = union_ratio(stain_boxes, img_w, img_h)
    has_vinyl = any(f["type"] == "vinyl" for f in findings)
    coated_sheet = bool(rules.get("coated_sheet_reject")) and has_vinyl and paper == "sheet"
    if coated_sheet and (paper_conf or 0) < rules.get("coated_sheet_type_conf", 0):
        coated_sheet, review = False, True            # 낱장인지 확신이 낮으면 버리지 않고 사람이 확인

    note = None
    if stain_ratio > rules["stain_reject_ratio"]:
        g, after = "X", None
        for f in findings:
            if f["type"] == "stain":
                f["instruction"] = rules["classes"]["stain"]["reject_instruction"]
    elif coated_sheet:
        g, after, note = "X", None, rules["grades"]["X_coated"]
        for f in findings:
            if f["type"] == "vinyl":
                f["instruction"] = rules["classes"]["vinyl"]["reject_instruction"]
    elif "lossy" in kinds:
        g, after = "C", "A"
    elif "removable" in kinds:
        g, after = "B", "A"
    else:
        g, after = "A", "A"
    if g == "X":                                      # 등급 외면 다른 전처리 지침은 의미가 없으므로 사유만 남김
        keep = "stain" if note is None else "vinyl"
        for f in findings:
            if f["type"] not in (keep, "pii"):       # 개인정보 파기 지침은 등급 외여도 남김
                f["instruction"] = None
    return g, after, findings, round(stain_ratio, 3), note or rules["grades"][g], review


# ───────────────────────── 4. 히트맵 (Grad-CAM) ─────────────────────────
STRIDE = 32


def _jet(v):
    """0~1 값을 파랑→초록→노랑→빨강 색으로 (H,W) → (H,W,3)"""
    r = np.clip(1.5 - np.abs(4 * v - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * v - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * v - 1), 0, 1)
    return np.stack([r, g, b], axis=-1)


class GradCAM:
    def __init__(self, yolo, imgsz=640):
        self.net = yolo.model.eval()
        self.names = yolo.names
        self.imgsz = imgsz
        self.acts = {}
        head = self.net.model[-1]
        for i in head.f:                                   # 탐지 헤드 입력층 (yolo11n: 16, 19, 22)
            self.net.model[i].register_forward_hook(
                lambda mod, inp, out, i=i: self.acts.__setitem__(i, out)
            )
        self.layers = list(head.f)

    def _letterbox(self, img):
        """비율을 유지한 채 imgsz 안에 맞추고 32의 배수로 여백을 채움"""
        w, h = img.size
        s = min(self.imgsz / w, self.imgsz / h)
        nw, nh = max(1, round(w * s)), max(1, round(h * s))
        W = int(np.ceil(nw / STRIDE) * STRIDE)
        H = int(np.ceil(nh / STRIDE) * STRIDE)
        canvas = Image.new("RGB", (W, H), (114, 114, 114))
        left, top = (W - nw) // 2, (H - nh) // 2
        canvas.paste(img.resize((nw, nh), Image.BILINEAR), (left, top))
        return canvas, (left, top, nw, nh)

    def cam(self, img, class_names=None, conf=0.35):
        """PIL 이미지 → 원본 크기의 0~1 히트맵 (numpy, H×W)"""
        canvas, (left, top, nw, nh) = self._letterbox(img)
        x = torch.from_numpy(np.array(canvas)).permute(2, 0, 1).float().div(255).unsqueeze(0)
        x.requires_grad_(True)

        with torch.enable_grad():
            out = self.net(x)
            pred = out[0] if isinstance(out, (tuple, list)) else out   # (1, 4+nc, N)
            scores = pred[0, 4:, :]                                    # (nc, N)
            if class_names:
                idx = [i for i, n in self.names.items() if n in class_names]
                scores = scores[idx] if idx else scores[:0]
            hit = scores >= conf
            H, W = canvas.size[1], canvas.size[0]
            if scores.numel() == 0 or not bool(hit.any()):
                total = np.zeros((H, W), dtype=np.float32)             # 탐지 없음 → 차가운 지도
            else:
                target = scores[hit].sum()
                acts = [self.acts[i] for i in self.layers]
                grads = torch.autograd.grad(target, acts, allow_unused=True)
                total = np.zeros((H, W), dtype=np.float32)
                for a, g in zip(acts, grads):
                    if g is None:
                        continue
                    m = F.relu((g * a).sum(dim=1, keepdim=True))       # 위치별 기울기×특징 (탐지 모델에 맞춘 Grad-CAM 변형)
                    m = F.avg_pool2d(m, 3, stride=1, padding=1)        # 살짝 부드럽게
                    m = F.interpolate(m, size=(H, W), mode="bilinear", align_corners=False)[0, 0]
                    m = m.detach().numpy()
                    if m.max() > 0:
                        total += m / m.max()
        self.acts.clear()

        total = total[top:top + nh, left:left + nw]                    # 여백 제거
        if total.max() > 0:
            total = total / total.max()
            total = np.clip((total - HEAT_FLOOR) / (1 - HEAT_FLOOR), 0, 1)       # 약한 반응은 지워서 핵심만 남김
        cam = Image.fromarray((total * 255).astype(np.uint8)).resize(img.size, Image.BILINEAR)
        return np.asarray(cam, dtype=np.float32) / 255

    def overlay(self, img, class_names=None, conf=0.35, alpha=0.5):
        """원본 사진 위에 히트맵을 겹친 PIL 이미지"""
        cam = self.cam(img, class_names, conf)
        base = np.asarray(img, dtype=np.float32) / 255
        a = (alpha * (0.35 + 0.65 * cam))[..., None]                   # 차가운 곳은 옅게 칠해 사진이 잘 보이게
        mix = (1 - a) * base + a * _jet(cam)
        return Image.fromarray((np.clip(mix, 0, 1) * 255).astype(np.uint8))


def shrink(img, max_side=1280):
    if max(img.size) <= max_side:
        return img
    s = max_side / max(img.size)
    return img.resize((round(img.width * s), round(img.height * s)), Image.BILINEAR)


def jpeg_bytes(img, quality=85):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


# ───────────────────────── 5. 판정 기록 ─────────────────────────
# 판정할 때마다 records 폴더에 자동으로 남습니다. 프로그램을 껐다 켜도 지워지지 않습니다.
#   records/records.db      판정 내용 (일시, 종류, 등급, 결함, 오염 면적, 규칙 버전, 모델, 사진 해시)
#   records/000001.jpg      판정한 사진 (긴 변 1280px로 줄인 사본)
#   records/000001_heat.jpg 히트맵
RECORD_DIR.mkdir(exist_ok=True)
DB_PATH = RECORD_DIR / "records.db"
db_lock = threading.Lock()


def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


with db() as _c:
    _c.execute("""CREATE TABLE IF NOT EXISTS records(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT, filename TEXT, sha256 TEXT,
        paper_type TEXT, type_conf REAL,
        grade TEXT, grade_after TEXT,
        defects TEXT, stain_ratio REAL, guides TEXT,
        rules_version TEXT, model TEXT, result TEXT)""")


def file_id(path):
    """모델 파일의 지문 (어떤 모델로 판정했는지 기록용)"""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


def save_record(result, filename, sha, photo, heat):
    with db_lock, db() as conn:
        cur = conn.execute(
            "INSERT INTO records(ts,filename,sha256,paper_type,type_conf,grade,grade_after,defects,stain_ratio,guides,rules_version,model,result)"
            " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (result["time"], filename, sha, result["paper_type"], result["paper_type_conf"],
             result["grade"], result["grade_after"],
             json.dumps(result["defect_counts"], ensure_ascii=False), result["stain_ratio"],
             json.dumps(result["guides"], ensure_ascii=False),
             result["rules_version"], result["model"], json.dumps(result, ensure_ascii=False)))
        rid = cur.lastrowid
    (RECORD_DIR / f"{rid:06d}.jpg").write_bytes(photo)
    if heat:
        (RECORD_DIR / f"{rid:06d}_heat.jpg").write_bytes(heat)
    return rid


# ───────────────────────── 6. 서버 ─────────────────────────
WEIGHTS = next((p for p in DET_CANDIDATES if p.exists()), None)
if WEIGHTS is None:
    raise SystemExit("best.pt 를 찾지 못했습니다. 이 파일과 같은 폴더에 best.pt 를 넣어 주세요.")
TYPE_WEIGHTS = next((p for p in TYPE_CANDIDATES if p.exists()), None)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024       # 20MB
model = YOLO(str(WEIGHTS))                                # 서버 시작 시 한 번만 불러옴
type_model = YOLO(str(TYPE_WEIGHTS)) if TYPE_WEIGHTS else None   # 없으면 종류는 '미분류'로 표시
MODEL_ID = file_id(WEIGHTS) + ("+" + file_id(TYPE_WEIGHTS) if TYPE_WEIGHTS else "")
lock = threading.Lock()
model.predict(Image.new("RGB", (IMGSZ, IMGSZ)), imgsz=IMGSZ, augment=TTA, verbose=False)   # 예열: 첫 사진도 바로 응답하도록
if type_model:
    type_model.predict(Image.new("RGB", (320, 320)), verbose=False)
cam = GradCAM(model, CAM_IMGSZ) if HEATMAP else None


def paper_type(img, dets):
    """종이 종류: 낱장(sheet) / 제본(bound).  스프링이 보이면 제본으로 확정"""
    need = RULES["class_confidence"].get("spring", RULES["confidence_threshold"])
    if any(d["class"] == "spring" and d["confidence"] >= need for d in dets):
        return "bound", 1.0
    if type_model is None:
        return None, None
    p = type_model.predict(img, verbose=False)[0].probs
    return type_model.names[int(p.top1)], round(float(p.top1conf), 3)


@app.get("/")
def index():
    return Response(PAGE, mimetype="text/html")


@app.get("/health")
def health():
    return jsonify(status="ok", model=MODEL_ID, rules_version=RULES["version"])


@app.post("/predict")
def predict():
    f = request.files.get("file")
    if f is None:
        return jsonify(error="file 필드에 이미지를 담아 보내 주세요."), 400
    raw = f.read()
    try:
        img = Image.open(io.BytesIO(raw))
        img = ImageOps.exif_transpose(img).convert("RGB")   # 폰 사진 회전 보정 (화면과 좌표 일치)
    except Exception:
        return jsonify(error="이미지를 읽을 수 없습니다."), 400

    # ① 사실: 결함 탐지 + 종이 종류
    w, h = img.size
    with lock:
        r = model.predict(img, conf=PREDICT_CONF, imgsz=IMGSZ, augment=TTA, verbose=False)[0]
        dets = [
            {
                "class": r.names[int(b.cls)],
                "confidence": float(b.conf),
                "box": [round(float(v), 1) for v in b.xyxy[0].tolist()],
            }
            for b in r.boxes
        ]
        ptype, pconf = paper_type(img, dets)

    # ② 규칙: 등급
    g, after, findings, stain_ratio, note, review = grade(dets, RULES, w, h, ptype, pconf)

    small = shrink(img)
    heat_jpg = None
    if cam is not None:
        try:
            with lock:
                heat_jpg = jpeg_bytes(cam.overlay(small, class_names=set(RULES["classes"]), conf=RULES["confidence_threshold"]))
        except Exception as e:                              # 히트맵이 실패해도 판정은 그대로 보냄
            print("히트맵 생성 실패:", e)

    guides, counts = [], {}
    for fnd in findings:
        name = SHORT.get(fnd["type"], fnd["label"])
        counts[name] = counts.get(name, 0) + 1
        if fnd["instruction"] and fnd["instruction"] not in guides:   # 같은 지침은 한 번만
            guides.append(fnd["instruction"])

    result = dict(
        time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        # 등급
        grade=g,                                            # A / B / C / X(등급 외)
        grade_text=GRADE_KO[g],
        grade_note=note,
        grade_after=after,                                  # 전처리 후 예상 등급 (등급 외면 null)
        grade_after_text=GRADE_KO[after] if after else None,
        verdict_code={"A": "pass", "B": "preprocess", "C": "preprocess", "X": "reject"}[g],
        # 사실
        paper_type=ptype,                                   # sheet / bound / null
        paper_type_text=TYPE_KO.get(ptype, "미분류"),
        paper_type_conf=pconf,
        stain_ratio=stain_ratio,                            # 오염 면적 비율 (0~1)
        defect_counts=counts,
        review=review,
        image_size=[w, h],
        detections=[
            {
                "type": x["type"],
                "label": SHORT.get(x["type"], x["label"]),
                "label_full": x["label"],
                "kind": x["kind"],
                "confidence": x["confidence"],
                "box": x["box_xyxy"],
            }
            for x in findings
        ],
        guides=guides,
        rules_version=RULES["version"],
        model=MODEL_ID,
    )

    # ③ 기록 저장
    try:
        rid = save_record(result, f.filename or "", hashlib.sha256(raw).hexdigest(), jpeg_bytes(small), heat_jpg)
        result["id"] = rid
    except Exception as e:
        print("기록 저장 실패:", e)
        result["id"] = None
    result["heatmap"] = ("data:image/jpeg;base64," + base64.b64encode(heat_jpg).decode()) if heat_jpg else None
    return jsonify(result)


@app.get("/records")
def records():
    with db() as conn:
        rows = conn.execute("SELECT id,ts,paper_type,grade,grade_after,defects,stain_ratio FROM records ORDER BY id DESC LIMIT 300").fetchall()
        total = conn.execute("SELECT COUNT(*) FROM records").fetchone()[0]
    return jsonify(total=total, items=[
        dict(id=x["id"], time=x["ts"], paper_type_text=TYPE_KO.get(x["paper_type"], "미분류"),
             grade=x["grade"], grade_text=GRADE_KO.get(x["grade"], x["grade"]),
             defect_counts=json.loads(x["defects"] or "{}"))
        for x in rows
    ])


@app.get("/records/<int:rid>")
def record(rid):
    with db() as conn:
        row = conn.execute("SELECT result FROM records WHERE id=?", (rid,)).fetchone()
    if row is None:
        return jsonify(error="기록이 없습니다."), 404
    res = json.loads(row["result"])
    res["id"] = rid
    res["photo"] = f"/records/{rid}/photo"
    res["heatmap"] = f"/records/{rid}/heat" if (RECORD_DIR / f"{rid:06d}_heat.jpg").exists() else None
    return jsonify(res)


@app.get("/records/<int:rid>/<kind>")
def record_image(rid, kind):
    p = RECORD_DIR / (f"{rid:06d}_heat.jpg" if kind == "heat" else f"{rid:06d}.jpg")
    if kind not in ("photo", "heat") or not p.exists():
        return jsonify(error="사진이 없습니다."), 404
    return send_file(p, mimetype="image/jpeg")


@app.get("/records.csv")
def records_csv():
    """판정 기록 내보내기 (엑셀에서 바로 열리는 CSV)"""
    with db() as conn:
        rows = conn.execute("SELECT * FROM records ORDER BY id").fetchall()
    buf = io.StringIO()
    wr = csv.writer(buf)
    wr.writerow(["번호", "판정 일시", "파일명", "종류", "등급", "전처리 후 예상 등급", "결함", "오염 면적(%)", "전처리 지침", "사진 지문(SHA-256)", "규칙 버전", "모델"])
    for x in rows:
        defects = json.loads(x["defects"] or "{}")
        wr.writerow([
            x["id"], x["ts"], x["filename"], TYPE_KO.get(x["paper_type"], "미분류"),
            GRADE_KO.get(x["grade"], x["grade"]), GRADE_KO.get(x["grade_after"], "-") if x["grade_after"] else "-",
            ", ".join(f"{k} {v}" for k, v in defects.items()) or "없음",
            f'{(x["stain_ratio"] or 0) * 100:.1f}', " / ".join(json.loads(x["guides"] or "[]")),
            x["sha256"], x["rules_version"], x["model"],
        ])
    name = "ecorebook_records_" + datetime.now().strftime("%Y%m%d_%H%M") + ".csv"
    return Response("﻿" + buf.getvalue(), mimetype="text/csv; charset=utf-8",
                    headers={"Content-Disposition": f"attachment; filename={name}"})


# ───────────────────────── 7. 웹페이지 ─────────────────────────
# 디자인을 고칠 때는 이 안의 HTML만 수정하면 됩니다. (첫 줄의 긴 글자 덩어리는 ECO-REBOOK 로고 폰트)
PAGE = r'''<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>에코리북 AI 선별·등급화</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/variable/pretendardvariable-dynamic-subset.css">
<style>
@font-face{font-family:"KimWildGag";src:url(data:font/ttf;base64,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) format("truetype");font-weight:400;font-display:block}
:root{
  color-scheme:dark;
  --ink:#f2f4f3; --sub:#9a9fa6; --bg:#1b1c20; --row:#2a2c31; --line:#34363c;
  --lime:#32e8a5; --on:#06201a; --warn:#ffb54d; --bad:#ff7b7b;
}
*{box-sizing:border-box;-webkit-tap-highlight-color:transparent}
html,body{margin:0;height:100%}
body{background:#0c0d0f;color:var(--ink);font-family:"Pretendard Variable",Pretendard,"Noto Sans KR","Apple SD Gothic Neo",sans-serif;
  letter-spacing:-.02em;-webkit-font-smoothing:antialiased}
button{font:inherit;color:inherit;border:0;background:none;cursor:pointer;padding:0}
.app{position:relative;max-width:440px;height:100%;margin:0 auto;background:var(--bg);overflow:hidden;
  display:flex;flex-direction:column}
@media(min-width:480px){.app{height:min(900px,96vh);margin-top:2vh;border-radius:36px;box-shadow:0 0 0 1px var(--line),0 30px 80px rgba(0,0,0,.6)}}

/* splash */
.splash{position:absolute;inset:0;background:#000;z-index:20;display:flex;flex-direction:column;
  align-items:center;justify-content:center;gap:26px;transition:opacity .5s}
.splash.hide{opacity:0;pointer-events:none}
.logo{color:#fff;font-family:"KimWildGag","Pretendard Variable",sans-serif;font-size:36px;font-weight:400;letter-spacing:0}
.logo i{color:#32e8a5;font-style:normal}
.ring{width:30px;height:30px;border-radius:50%;border:2px solid #333;border-top-color:var(--lime);animation:sp 1s linear infinite}
@keyframes sp{to{transform:rotate(360deg)}}

/* screens */
.screen{display:none;flex:1;overflow-y:auto;padding:max(20px,env(safe-area-inset-top)) 20px 24px}
.screen.on{display:block}
.title{font-size:34px;font-weight:730;letter-spacing:-.04em;margin:18px 0 18px}
.title.brand{font-family:"KimWildGag","Pretendard Variable",sans-serif;font-weight:400;letter-spacing:0;font-size:36px}
.topbar{display:flex;align-items:center;height:44px;margin-top:4px}
.topbar h2{flex:1;text-align:center;font-size:16px;font-weight:640;margin:0;padding-right:36px}
.back{width:36px;height:36px;display:grid;place-items:center;border-radius:50%}

.banner{background:radial-gradient(120% 140% at 100% 0%,rgba(50,232,165,.28),transparent 55%),#24262b;border:1px solid var(--line);border-radius:20px;padding:20px;display:flex;justify-content:space-between;align-items:center;gap:12px}
.banner b{display:block;font-size:19px;line-height:1.3;letter-spacing:-.03em}
.banner span{font-size:12.5px;opacity:.75}

.drop{margin-top:16px;border-radius:20px;background:var(--row);border:1.5px dashed #44474e;min-height:250px;display:flex;flex-direction:column;
  align-items:center;justify-content:center;text-align:center;gap:6px;cursor:pointer;position:relative;overflow:hidden}
.drop.over{outline:2px solid var(--lime)}
.drop .plus{width:52px;height:52px;border-radius:50%;background:var(--lime);display:grid;place-items:center;margin-bottom:8px;box-shadow:0 0 24px rgba(50,232,165,.45)}
.drop b{font-size:16px}.drop small{color:var(--sub);font-size:13px}
.drop img{position:absolute;inset:0;width:100%;height:100%;object-fit:cover}
.btn{display:block;width:100%;height:54px;border-radius:27px;background:var(--lime);color:var(--on);font-weight:730;font-size:16px;margin-top:14px;box-shadow:0 8px 24px rgba(50,232,165,.25)}
.btn:disabled{background:var(--row);color:#62666d;box-shadow:none;cursor:not-allowed}
.btn.ghost{background:var(--row);color:var(--ink);box-shadow:none;margin-top:8px;height:46px;font-size:14px;font-weight:550}
.err{margin-top:12px;padding:12px 14px;border-radius:12px;background:rgba(255,123,123,.12);color:var(--bad);font-size:13.5px;display:none}

.sec{font-size:13px;font-weight:640;color:var(--sub);margin:26px 0 10px}
.how{display:grid;gap:8px}
.how div{background:var(--row);border-radius:12px;padding:13px 14px;font-size:14px;display:flex;gap:12px;align-items:center}
.how em{font-style:normal;font-weight:730;color:var(--lime);width:14px}

/* result */
.imgbox{position:relative;border-radius:18px;overflow:hidden;background:var(--row);margin-top:8px}
.imgbox img{display:block;width:100%}
.seg{display:flex;gap:6px;margin:10px 0 0}
.seg button{padding:7px 14px;border-radius:999px;background:var(--row);font-size:13px;font-weight:550}
.seg button.act{background:var(--lime);color:var(--on)}
.verdict{margin-top:14px;border-radius:20px;padding:16px 18px;display:flex;align-items:center;gap:16px}
.verdict.ok{background:rgba(50,232,165,.08);border:1px solid rgba(50,232,165,.25)}
.verdict.warn{background:rgba(255,181,77,.08);border:1px solid rgba(255,181,77,.25)}
.verdict.bad{background:rgba(255,123,123,.08);border:1px solid rgba(255,123,123,.25)}
.vbadge{width:60px;height:60px;border-radius:16px;display:grid;place-items:center;flex-shrink:0;font-size:30px;font-weight:800;letter-spacing:-.03em}
.verdict.ok .vbadge{background:rgba(50,232,165,.18);color:var(--lime)}
.verdict.warn .vbadge{background:rgba(255,181,77,.18);color:var(--warn)}
.verdict.bad .vbadge{background:rgba(255,123,123,.18);color:var(--bad)}
.vinfo{flex:1;min-width:0}
.vtitle{font-size:17px;font-weight:730;letter-spacing:-.02em}
.vdesc{font-size:13px;color:var(--sub);margin-top:3px;line-height:1.5}
.verdict.ok .vtitle{color:var(--lime)}.verdict.warn .vtitle{color:var(--warn)}.verdict.bad .vtitle{color:var(--bad)}
/* key-info panels */
.kpanels{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:10px}
.kp{background:var(--row);border-radius:14px;padding:12px 14px}
.kp.full{grid-column:1/-1}
.kp-label{font-size:12px;color:var(--sub);font-weight:550;margin-bottom:5px}
.kp-value{font-size:16px;font-weight:730;letter-spacing:-.02em;display:flex;align-items:center;gap:5px}
.kp-arrow{color:var(--sub);font-size:13px;font-weight:400}
.kp-after{color:var(--lime)}
.kp-pct{color:var(--warn)}
/* review warning */
.rw{margin-top:10px;padding:11px 14px;border-radius:12px;background:rgba(255,181,77,.1);border:1px solid rgba(255,181,77,.2);color:var(--warn);font-size:13px;display:flex;align-items:center;gap:8px}
.rw svg{flex-shrink:0}
.chips{display:flex;flex-wrap:wrap;gap:8px}
.chip{padding:8px 14px;border-radius:999px;background:var(--row);font-size:13.5px;font-weight:550;display:flex;align-items:center;gap:6px}
.chip i{width:8px;height:8px;border-radius:50%}
.list{display:grid;gap:8px}
.item{background:var(--row);border-radius:12px;padding:13px 14px;display:flex;align-items:center;gap:12px;font-size:14.5px}
.item .n{color:var(--lime);font-weight:730;width:14px}
.item .nm{flex:1;font-weight:550}
.item .pc{font-variant-numeric:tabular-nums;font-weight:640}
.item .pc.low{color:var(--warn)}
.empty{color:var(--sub);text-align:center;padding:50px 0;font-size:14px}
.spin{width:28px;height:28px;border-radius:50%;border:3px solid var(--line);border-top-color:var(--lime);animation:sp .9s linear infinite;margin:0 auto 14px}

/* history */
.hist{display:flex;gap:12px;align-items:center;background:var(--row);border-radius:14px;padding:10px;margin-bottom:8px;width:100%;text-align:left}
.hist img{width:56px;height:56px;border-radius:10px;object-fit:cover}
.hist b{display:block;font-size:15px}.hist small{color:var(--sub);font-size:12.5px}
a.btn{display:flex;align-items:center;justify-content:center;text-decoration:none;margin-top:0}
.item .pc.sub{color:var(--sub);font-weight:550}

/* tab bar */
.tabs{display:flex;border-top:1px solid var(--line);padding:8px 0 max(10px,env(safe-area-inset-bottom));background:#17181b}
.tabs button{flex:1;display:flex;flex-direction:column;align-items:center;gap:3px;font-size:11.5px;color:#6f747b;font-weight:550}
.tabs button.act{color:var(--ink)}
.tabs button.act svg{stroke:var(--lime)}
svg{display:block}.screen::-webkit-scrollbar{display:none}.screen{scrollbar-width:none}
:focus-visible{outline:3px solid var(--lime);outline-offset:2px}
@media(prefers-reduced-motion:reduce){.ring,.spin{animation-duration:3s}.splash{transition:none}}

/* 화면 크기에 따라 유동적으로 */
.screen{padding-left:max(clamp(14px,5vw,20px),env(safe-area-inset-left));padding-right:max(clamp(14px,5vw,20px),env(safe-area-inset-right))}
.title{font-size:clamp(26px,8.5vw,34px)}
.logo,.title.brand{font-size:clamp(27px,9vw,36px)}
.banner{padding:clamp(14px,5vw,20px)}
.banner b{font-size:clamp(16px,5vw,19px)}
.drop{min-height:clamp(170px,34vh,250px)}
.verdict{padding:clamp(12px,4vw,16px) clamp(14px,4.5vw,18px);gap:clamp(10px,4vw,16px)}
.vbadge{width:clamp(48px,15vw,60px);height:clamp(48px,15vw,60px);font-size:clamp(24px,7.5vw,30px)}
.vtitle{font-size:clamp(15px,4.6vw,17px)}
.kp-value{font-size:clamp(14px,4.3vw,16px);flex-wrap:wrap}
.hist>div{min-width:0}.hist small{display:block;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.item .nm{min-width:0;overflow-wrap:anywhere}
@media(max-width:479px){.app{max-width:none;height:100dvh}}
@media(max-width:330px){.kpanels{grid-template-columns:1fr}.kp.full{grid-column:auto}.chip{padding:7px 11px;font-size:12.5px}}
@media(max-height:560px){.title{margin:8px 0 10px}.drop{min-height:130px}.sec{margin-top:18px}.tabs button svg{width:20px;height:20px}}
@media(hover:none){.seg button{padding:9px 16px}.back{width:44px;height:44px}.topbar h2{padding-right:44px}}
</style>
</head>
<body>
<div class="app">
  <div class="splash" id="splash"><div class="logo">ECO-REBOOK<i>.</i></div><div class="ring"></div></div>

  <!-- 선별 -->
  <section class="screen on" id="s-home">
    <div class="title brand">ECO-REBOOK</div>
    <div class="banner">
      <div><b>종이 한 장씩<br>사진으로 확인하세요</b><span>결함을 찾아 등급과 손질 방법을 알려드려요</span></div>
    </div>
    <div class="drop" id="drop" tabindex="0" role="button" aria-label="사진 선택">
      <div class="plus"><svg width="22" height="22" viewBox="0 0 24 24" stroke="#06201a" stroke-width="2.4" stroke-linecap="round"><path d="M12 5v14M5 12h14"/></svg></div>
      <b>사진 올리기</b><small>끌어다 놓거나 눌러서 선택</small>
    </div>
    <input type="file" id="file" accept="image/*" hidden>
    <div class="err" id="err"></div>
    <button class="btn" id="run" disabled>선별 시작</button>
    <button class="btn ghost" id="reset" hidden>다른 사진 선택</button>
    <div class="sec">사진 찍는 방법</div>
    <div class="how">
      <div><em>1</em>종이를 평평한 바닥에 펼쳐요</div>
      <div><em>2</em>그림자가 지지 않게 밝은 곳에서 찍어요</div>
      <div><em>3</em>한 번에 한 면씩 가득 차게 찍어요</div>
    </div>
  </section>

  <!-- 결과 -->
  <section class="screen" id="s-result">
    <div class="topbar"><button class="back" id="back" aria-label="뒤로"><svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#f2f4f3" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M15 5l-7 7 7 7"/></svg></button><h2>선별 결과</h2></div>
    <div id="result"></div>
  </section>

  <!-- 기록 -->
  <section class="screen" id="s-hist">
    <div class="title">History</div>
    <a class="btn ghost" id="export" href="/records.csv" hidden>판정 기록 내보내기 (CSV)</a>
    <div class="sec" id="histCount" hidden></div>
    <div id="histList"><div class="empty">선별한 기록이 여기에 쌓여요.<br>프로그램을 껐다 켜도 남아 있어요.</div></div>
  </section>

  <nav class="tabs">
    <button class="act" data-go="home"><svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#6f747b" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 8V5h3M17 5h3v3M20 16v3h-3M7 19H4v-3M4 12h16"/></svg>선별</button>
    <button data-go="hist"><svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#6f747b" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="8"/><path d="M12 8v4l3 2"/></svg>기록</button>
  </nav>
</div>

<script>
/* ============ 여기만 수정하세요 ============ */
const CONFIG = {
  API_URL: "/predict",                       // 분석 API (같은 서버, POST multipart/form-data)
  FILE_FIELD: "file",                        // 이미지 필드 이름
  LOW_CONF: 0.5,                             // 미만이면 사람 검수 안내
  TIMEOUT_MS: 30000
};
/* 응답 예시 — /predict 기준 (이름이 다르면 normalize()만 수정)
{ "grade":"B", "grade_text":"B등급", "grade_after":"A", "grade_after_text":"A등급", "grade_note":"...",
  "paper_type_text":"제본", "paper_type_conf":0.97, "stain_ratio":0.0, "review":false, "image_size":[w,h],
  "detections":[{"type":"staple","label":"스테이플","confidence":0.91,"box":[x1,y1,x2,y2]}],
  "guides":["..."], "heatmap":"data:image/...", "id":12 } */
const COLORS={"오염":"#ffb54d","스프링":"#ff6b6b","스테이플":"#ff6b6b","테이프":"#ff6b6b","코팅":"#c9a0ff","개인정보":"#6aa8ff"};
const $=id=>document.getElementById(id);
let file=null, url=null;

setTimeout(()=>$("splash").classList.add("hide"),1300);

function normalize(d){
  return{
  id:d.id??null, time:d.time||"",
  grade:d.grade, gradeText:d.grade_text, note:d.grade_note||"",
  after:d.grade_after||null, afterText:d.grade_after_text||null,
  type:d.paper_type_text||"미분류", typeConf:d.paper_type_conf??null,
  ratio:d.stain_ratio||0, review:!!d.review, size:d.image_size||null,
  dets:(d.detections||[]).map(x=>({label:x.label,conf:x.confidence,box:x.box})),
  photo:d.photo||null, heat:d.heatmap||null, guides:d.guides||[]};}

/* 화면 전환 */
function show(n){
  document.querySelectorAll(".screen").forEach(s=>s.classList.toggle("on",s.id==="s-"+n));
  document.querySelectorAll(".tabs button").forEach(b=>{
    const on=b.dataset.go===(n==="result"?"home":n); b.classList.toggle("act",on);
    b.querySelector("svg").style.stroke=on?"#32e8a5":"#6f747b";});
  document.querySelector(".screen.on").scrollTop=0;
}
document.querySelectorAll(".tabs button").forEach(b=>b.onclick=()=>{show(b.dataset.go);if(b.dataset.go==="hist")loadHist()});
$("back").onclick=()=>show("home");

/* 사진 선택 */
const drop=$("drop"),inp=$("file");
drop.onclick=()=>inp.click();
drop.onkeydown=e=>{if(e.key==="Enter"||e.key===" "){e.preventDefault();inp.click()}};
["dragenter","dragover"].forEach(t=>drop.addEventListener(t,e=>{e.preventDefault();drop.classList.add("over")}));
["dragleave","drop"].forEach(t=>drop.addEventListener(t,e=>{e.preventDefault();drop.classList.remove("over")}));
drop.addEventListener("drop",e=>setFile(e.dataTransfer.files[0]));
inp.onchange=()=>setFile(inp.files[0]);
$("reset").onclick=()=>{file=null;inp.value="";drop.querySelectorAll("img").forEach(i=>i.remove());
  drop.querySelector(".plus").style.display="";drop.querySelectorAll("b,small").forEach(e=>e.style.display="");
  $("run").disabled=true;$("reset").hidden=true;err("")};
function setFile(f){
  if(!f||!f.type.startsWith("image/")){err("JPG, PNG 같은 이미지 파일만 올릴 수 있어요.");return}
  err(""); file=f; if(url)URL.revokeObjectURL(url); url=URL.createObjectURL(f);
  drop.querySelectorAll("img").forEach(i=>i.remove());
  const im=new Image(); im.src=url; im.alt="선택한 사진"; drop.appendChild(im);
  drop.querySelector(".plus").style.display="none"; drop.querySelectorAll("b,small").forEach(e=>e.style.display="none");
  $("run").disabled=false; $("reset").hidden=false;
}
function err(m){const e=$("err");e.textContent=m;e.style.display=m?"block":"none"}

/* 분석 */
$("run").onclick=async()=>{
  if(!file)return; err(""); $("run").disabled=true;
  show("result"); $("result").innerHTML='<div class="empty"><div class="spin"></div>분석하고 있어요</div>';
  const fd=new FormData(); fd.append(CONFIG.FILE_FIELD,file);
  const c=new AbortController(), t=setTimeout(()=>c.abort(),CONFIG.TIMEOUT_MS);
  try{
    const r=await fetch(CONFIG.API_URL,{method:"POST",body:fd,signal:c.signal});
    if(!r.ok)throw new Error("서버 응답 오류 "+r.status);
    const data=normalize(await r.json()); render(data); loadHist();
  }catch(e){
    show("home");
    err(e.name==="AbortError"?"응답 시간이 초과됐어요. 서버가 켜져 있는지 확인해 주세요."
      :"분석에 실패했어요. API 주소와 서버 실행 여부를 확인해 주세요. ("+e.message+")");
  }finally{clearTimeout(t);$("run").disabled=false}
};

const col=l=>COLORS[l]||"#32e8a5";
const esc=s=>String(s).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));

function verdict(r){
  const low=r.dets.some(d=>d.conf<CONFIG.LOW_CONF)||r.review;
  const typeLow=r.typeConf!=null&&r.typeConf<0.7;
  const cls=r.grade==="X"?"bad":(r.grade==="A"?"ok":"warn");
  const sub={A:"바로 원료로 쓸 수 있어요",B:"전처리하면 전량 쓸 수 있어요",C:"일부를 덜어내면 쓸 수 있어요",X:"원료로 쓸 수 없어요"}[r.grade]||"";
  return{cls,txt:r.gradeText,sub,badge:r.grade==="X"?"외":r.grade,review:low||typeLow};
}

function render(r){
  const v=verdict(r), cnt={}; r.dets.forEach(d=>cnt[d.label]=(cnt[d.label]||0)+1);
  const chips=Object.keys(cnt).map(k=>`<span class="chip"><i style="background:${col(k)}"></i>${esc(k)} ${cnt[k]}</span>`).join("");
  const kp=[];
  if(r.after&&r.grade!=="A")kp.push(`<div class="kp-label">전처리 후 예상</div><div class="kp-value">${esc(r.grade)}<span class="kp-arrow">→</span><span class="kp-after">${esc(r.afterText)}</span></div>`);
  else if(r.grade==="X")kp.push(`<div class="kp-label">전처리 후 예상</div><div class="kp-value">올릴 수 없음</div>`);
  kp.push(`<div class="kp-label">종이 종류</div><div class="kp-value">${esc(r.type)}</div>`);
  kp.push(`<div class="kp-label">결함</div><div class="kp-value">${r.dets.length?r.dets.length+"개":"없음"}</div>`);
  if(r.ratio>0)kp.push(`<div class="kp-label">오염 면적</div><div class="kp-value kp-pct">약 ${Math.round(r.ratio*100)}%</div>`);
  const panels=kp.map((h,i)=>`<div class="kp${kp.length%2&&i===kp.length-1?" full":""}">${h}</div>`).join("");
  const warn=v.review?`<div class="rw"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 9v4M12 17h.01M10.3 3.9L2.4 17.5A2 2 0 004.1 20.5h15.8a2 2 0 001.7-3L13.7 3.9a2 2 0 00-3.4 0z"/></svg>확신이 낮은 결과가 있어요. 사람이 한 번 더 확인해 주세요.</div>`:"";
  const rows=r.dets.map((d,i)=>`<div class="item"><span class="n">${i+1}</span><span class="nm">${esc(d.label)}</span><span class="pc ${d.conf<CONFIG.LOW_CONF?"low":""}">${(d.conf*100).toFixed(1)}%</span></div>`).join("");
  $("result").innerHTML=`
    <div class="imgbox" id="imgbox"></div>
    ${r.heat?'<div class="seg"><button class="act" data-m="heat">히트맵</button><button data-m="box">탐지 위치</button></div>':""}
    <div class="verdict ${v.cls}"><div class="vbadge">${esc(v.badge)}</div><div class="vinfo"><div class="vtitle">${esc(v.txt)}</div><div class="vdesc">${esc(v.sub)}</div></div></div>
    ${warn}
    <div class="kpanels">${panels}</div>
    ${r.grade==="X"&&r.note?`<div class="sec">판정 사유</div><div class="list"><div class="item"><span class="nm" style="font-weight:500">${esc(r.note)}</span></div></div>`:""}
    ${chips?`<div class="sec">검출된 결함</div><div class="chips">${chips}</div>`:""}
    ${r.guides.length?`<div class="sec">${r.grade==="X"?"처리 방법":"이렇게 손보면 등급이 올라가요"}</div><div class="list">${r.guides.map((g,i)=>`<div class="item"><span class="n">${i+1}</span><span class="nm" style="font-weight:500">${esc(g)}</span></div>`).join("")}</div>`:""}
    ${rows?`<div class="sec">결함별 확신도</div><div class="list">${rows}</div>`:""}
    <div class="sec">판정 기록</div><div class="list"><div class="item"><span class="nm" style="font-weight:500">${r.id!=null?"기록 번호 "+r.id:"저장되지 않음"}</span><span class="pc sub">${esc(r.time)}</span></div></div>
    <button class="btn" id="again">다른 사진 선별하기</button>`;
  draw(r,r.heat?"heat":"box");
  document.querySelectorAll(".seg button").forEach(b=>b.onclick=()=>{
    document.querySelectorAll(".seg button").forEach(x=>x.classList.remove("act"));b.classList.add("act");draw(r,b.dataset.m)});
  $("again").onclick=()=>{$("reset").click();show("home")};
}

function draw(r,mode){
  const box=$("imgbox"); box.innerHTML="";
  if(mode==="heat"&&r.heat){box.innerHTML=`<img src="${r.heat}" alt="히트맵">`;return}
  const im=new Image(); im.alt="탐지 결과"; im.src=r.photo||url;
  im.onload=()=>{
    const [W,H]=r.size||[im.naturalWidth,im.naturalHeight]; box.appendChild(im);
    const s=document.createElementNS("http://www.w3.org/2000/svg","svg");
    s.setAttribute("viewBox",`0 0 ${W} ${H}`); s.style.cssText="position:absolute;inset:0;width:100%;height:100%";
    const w=Math.max(2,W/260);
    r.dets.forEach(d=>{if(!d.box)return;const[x1,y1,x2,y2]=d.box,c=col(d.label);
      s.insertAdjacentHTML("beforeend",`<rect x="${x1}" y="${y1}" width="${x2-x1}" height="${y2-y1}" rx="${w*2}" fill="none" stroke="${c}" stroke-width="${w}"/><text x="${x1}" y="${Math.max(y1-w*2,w*7)}" fill="${c}" font-size="${w*6}" font-weight="800">${esc(d.label)} ${(d.conf*100).toFixed(0)}%</text>`)});
    box.appendChild(s);
  };
}

/* 기록 (서버의 records 폴더에 저장돼요. 껐다 켜도 남아 있어요) */
async function loadHist(){
  try{
    const d=await (await fetch("/records")).json();
    $("export").hidden=!d.total; $("histCount").hidden=!d.total; $("histCount").textContent="전체 "+d.total+"건";
    if(!d.total)return;
    $("histList").innerHTML=d.items.map(h=>{
      const df=Object.entries(h.defect_counts).map(([k,n])=>k+" "+n).join(" · ")||"결함 없음";
      return `<button class="hist" data-id="${h.id}"><img src="/records/${h.id}/photo" alt="" loading="lazy"><div><b>${esc(h.grade_text)} · ${esc(h.paper_type_text)}</b><small>${esc(h.time.slice(5,16))} · ${esc(df)}</small></div></button>`}).join("");
    document.querySelectorAll(".hist").forEach(b=>b.onclick=async()=>{
      const r=normalize(await (await fetch("/records/"+b.dataset.id)).json()); render(r); show("result")});
  }catch(e){}
}
loadHist();
</script>
</body>
</html>
'''


if __name__ == "__main__":
    print(f"\n  에코리북 AI  →  http://localhost:{PORT}\n")
    app.run(host=HOST, port=PORT, threaded=True)
