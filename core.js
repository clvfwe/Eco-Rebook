/* 에코리북 브라우저판 — 등급 규칙과 모델 출력 해석 (ecorebook_app.py 의 RULES · grade() 와 같은 내용)
   구매처 기준이 바뀌면 RULES 만 고치면 됩니다. 파이썬판도 함께 고쳐 주세요. */
(function (g) {
  const NAMES = ["spring", "staple", "tape", "vinyl", "stain", "pii"];   // best.onnx 의 클래스 순서
  const TYPE_NAMES = ["bound", "sheet"];                                 // type.onnx 의 클래스 순서
  const SHORT = { spring: "스프링", staple: "스테이플", tape: "테이프", vinyl: "코팅", stain: "오염", pii: "개인정보" };
  const TYPE_KO = { sheet: "낱장", bound: "제본" };
  const GRADE_KO = { A: "A등급", B: "B등급", C: "C등급", X: "등급 외" };

  const RULES = {
    version: "2026-11-18",
    confidence_threshold: 0.35,
    class_confidence: {},
    review_below: 0.5,
    stain_reject_ratio: 0.30,
    coated_sheet_reject: true,
    coated_sheet_type_conf: 0.70,
    classes: {
      spring: { kind: "removable", label_ko: "금속 스프링 제본", instruction: "스프링을 분리하거나 제본부를 절단선 기준으로 잘라낸 뒤 투입하십시오." },
      staple: { kind: "removable", label_ko: "스테이플러 심·금속 이물질", instruction: "스테이플러 심을 제거하거나 제본부를 절단한 뒤 투입하십시오." },
      tape: { kind: "removable", label_ko: "테이프·접착제 잔여물", instruction: "테이프가 붙은 부분을 잘라내거나 해당 장을 분리한 뒤 투입하십시오." },
      pii: { kind: "removable", label_ko: "개인정보 기재 영역", instruction: "이름·학번 등 개인정보가 적힌 부분을 잘라 자체 파기한 뒤 나머지를 투입하십시오." },
      vinyl: { kind: "lossy", label_ko: "비닐·필름 코팅", instruction: "비닐 커버 또는 코팅된 표지를 분리하고 내지만 투입하십시오.",
        reject_instruction: "종이 전체가 코팅되어 있어 종이만 분리할 수 없습니다. 원료로 쓰지 말고 따로 배출하십시오." },
      stain: { kind: "lossy", label_ko: "오염 얼룩", instruction: "오염된 장을 골라내고 깨끗한 장만 투입하십시오.",
        reject_instruction: "오염 범위가 넓어 원료로 사용할 수 없습니다. 일반 폐지로 배출하십시오." }
    },
    grades: {
      A: "결함이 없어 바로 원료로 쓸 수 있습니다.",
      B: "떼어낼 수 있는 결함만 있습니다. 전처리하면 전량 쓸 수 있습니다.",
      C: "코팅 또는 부분 오염이 있습니다. 해당 부분을 덜어낸 나머지를 쓸 수 있습니다.",
      X: "오염이 넓어 원료로 쓸 수 없습니다.",
      X_coated: "낱장 전체가 코팅되어 있어 원료로 쓸 수 없습니다."
    }
  };

  const round = (v, n) => { const p = Math.pow(10, n); return Math.round(v * p) / p; };
  const need = (rules, cls) => (cls in rules.class_confidence ? rules.class_confidence[cls] : rules.confidence_threshold);

  function unionRatio(boxes, w, h, n = 200) {
    if (!boxes.length || w <= 0 || h <= 0) return 0;
    const grid = new Uint8Array(n * n);
    for (const [x1, y1, x2, y2] of boxes) {
      const a = Math.max(Math.floor(x1 / w * n), 0), b = Math.max(Math.floor(y1 / h * n), 0);
      const c = Math.min(Math.ceil(x2 / w * n), n), d = Math.min(Math.ceil(y2 / h * n), n);
      for (let y = b; y < d; y++) for (let x = a; x < c; x++) grid[y * n + x] = 1;
    }
    let s = 0; for (let i = 0; i < grid.length; i++) s += grid[i];
    return s / grid.length;
  }

  function grade(detections, rules, w, h, paper, paperConf) {
    const findings = [], kinds = new Set(), stainBoxes = [];
    for (const d of detections) {
      const rule = rules.classes[d.class];
      if (!rule || d.confidence < need(rules, d.class)) continue;
      findings.push({ type: d.class, label: rule.label_ko, kind: rule.kind, confidence: round(d.confidence, 3), box: d.box, instruction: rule.instruction });
      kinds.add(rule.kind);
      if (d.class === "stain") stainBoxes.push(d.box);
    }
    let review = findings.some(f => f.confidence < rules.review_below);
    const stainRatio = unionRatio(stainBoxes, w, h);
    const hasVinyl = findings.some(f => f.type === "vinyl");
    let coatedSheet = !!rules.coated_sheet_reject && hasVinyl && paper === "sheet";
    if (coatedSheet && (paperConf || 0) < (rules.coated_sheet_type_conf || 0)) { coatedSheet = false; review = true; }

    let g, after, note = null;
    if (stainRatio > rules.stain_reject_ratio) {
      g = "X"; after = null;
      findings.forEach(f => { if (f.type === "stain") f.instruction = rules.classes.stain.reject_instruction; });
    } else if (coatedSheet) {
      g = "X"; after = null; note = rules.grades.X_coated;
      findings.forEach(f => { if (f.type === "vinyl") f.instruction = rules.classes.vinyl.reject_instruction; });
    } else if (kinds.has("lossy")) { g = "C"; after = "A"; }
    else if (kinds.has("removable")) { g = "B"; after = "A"; }
    else { g = "A"; after = "A"; }
    if (g === "X") {
      const keep = note === null ? "stain" : "vinyl";
      findings.forEach(f => { if (f.type !== keep && f.type !== "pii") f.instruction = null; });
    }
    return { g, after, findings, stainRatio: round(stainRatio, 3), note: note || rules.grades[g], review };
  }

  /* 사진을 비율 그대로 size×size 안에 맞추는 배치 */
  function letterbox(w, h, size) {
    const s = Math.min(size / w, size / h);
    const nw = Math.max(1, Math.round(w * s)), nh = Math.max(1, Math.round(h * s));
    return { s, nw, nh, left: Math.floor((size - nw) / 2), top: Math.floor((size - nh) / 2), w, h };
  }

  /* 모델 출력 (4+nc)×N → 후보 목록.  out[c*N+i], 앞 4칸은 가운데 x·y, 너비, 높이 (입력 크기 기준 픽셀) */
  function candidates(out, N, nc, lb, conf) {
    const res = [];
    for (let i = 0; i < N; i++) {
      let best = 0, cls = -1;
      for (let c = 0; c < nc; c++) { const v = out[(4 + c) * N + i]; if (v > best) { best = v; cls = c; } }
      if (best < conf) continue;
      const cx = out[i], cy = out[N + i], bw = out[2 * N + i], bh = out[3 * N + i];
      const cl = (v, m) => Math.min(Math.max(v, 0), m);
      res.push({ cls, confidence: best, box: [
        cl((cx - bw / 2 - lb.left) / lb.s, lb.w), cl((cy - bh / 2 - lb.top) / lb.s, lb.h),
        cl((cx + bw / 2 - lb.left) / lb.s, lb.w), cl((cy + bh / 2 - lb.top) / lb.s, lb.h)] });
    }
    return res;
  }

  function iou(a, b) {
    const iw = Math.min(a[2], b[2]) - Math.max(a[0], b[0]), ih = Math.min(a[3], b[3]) - Math.max(a[1], b[1]);
    if (iw <= 0 || ih <= 0) return 0;
    const inter = iw * ih;
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter);
  }

  /* 겹치는 후보 중 가장 확실한 것만 남김 (같은 종류끼리만 비교) */
  function nms(cands, thr = 0.7, maxDet = 300) {
    const sorted = cands.slice().sort((a, b) => b.confidence - a.confidence), keep = [];
    for (const c of sorted) {
      if (keep.every(k => k.cls !== c.cls || iou(k.box, c.box) <= thr)) keep.push(c);
      if (keep.length >= maxDet) break;
    }
    return keep.map(k => ({ class: NAMES[k.cls], confidence: k.confidence, box: k.box.map(v => round(v, 1)) }));
  }

  /* 히트맵용 격자 (G×G, 0~1). 모델이 결함이라고 본 위치일수록 값이 큼.
     서버판의 Grad-CAM 과 달리, 각 후보의 점수를 그 후보가 가리킨 영역에 부드럽게 칠하는 방식 */
  function heatGrid(cands, w, h, conf, G = 160, floor = 0.2) {
    const grid = new Float32Array(G * G);
    if (!cands.some(c => c.confidence >= conf)) return grid;
    for (const c of cands) {
      const [x1, y1, x2, y2] = c.box;
      const cx = (x1 + x2) / 2 / w * G, cy = (y1 + y2) / 2 / h * G;
      const rx = Math.max((x2 - x1) / 2 / w * G, 1.5), ry = Math.max((y2 - y1) / 2 / h * G, 1.5);
      const xa = Math.max(Math.floor(cx - rx * 1.6), 0), xb = Math.min(Math.ceil(cx + rx * 1.6), G - 1);
      const ya = Math.max(Math.floor(cy - ry * 1.6), 0), yb = Math.min(Math.ceil(cy + ry * 1.6), G - 1);
      for (let y = ya; y <= yb; y++) for (let x = xa; x <= xb; x++) {
        const dx = (x + 0.5 - cx) / rx, dy = (y + 0.5 - cy) / ry;
        const v = c.confidence * Math.exp(-1.2 * (dx * dx + dy * dy));
        if (v > grid[y * G + x]) grid[y * G + x] = v;
      }
    }
    let m = 0; for (let i = 0; i < grid.length; i++) if (grid[i] > m) m = grid[i];
    if (m > 0) for (let i = 0; i < grid.length; i++) grid[i] = Math.min(Math.max((grid[i] / m - floor) / (1 - floor), 0), 1);
    return grid;
  }

  /* 종이 종류: 스프링이 보이면 제본으로 확정, 아니면 종류 모델의 판단 */
  function paperType(dets, probs) {
    if (dets.some(d => d.class === "spring" && d.confidence >= need(RULES, "spring"))) return ["bound", 1.0];
    if (!probs) return [null, null];
    let k = 0; for (let i = 1; i < probs.length; i++) if (probs[i] > probs[k]) k = i;
    return [TYPE_NAMES[k], round(probs[k], 3)];
  }

  /* 화면과 기록에 쓰는 결과 (파이썬판 /predict 응답과 같은 이름) */
  function buildResult(dets, w, h, ptype, pconf, time, model) {
    const r = grade(dets, RULES, w, h, ptype, pconf), guides = [], counts = {};
    for (const f of r.findings) {
      const name = SHORT[f.type] || f.label;
      counts[name] = (counts[name] || 0) + 1;
      if (f.instruction && !guides.includes(f.instruction)) guides.push(f.instruction);
    }
    return {
      time, grade: r.g, grade_text: GRADE_KO[r.g], grade_note: r.note,
      grade_after: r.after, grade_after_text: r.after ? GRADE_KO[r.after] : null,
      paper_type: ptype, paper_type_text: TYPE_KO[ptype] || "미분류", paper_type_conf: pconf,
      stain_ratio: r.stainRatio, defect_counts: counts, review: r.review, image_size: [w, h],
      detections: r.findings.map(x => ({ type: x.type, label: SHORT[x.type] || x.label, label_full: x.label, kind: x.kind, confidence: x.confidence, box: x.box })),
      guides, rules_version: RULES.version, model
    };
  }

  g.ECO = { NAMES, TYPE_NAMES, TYPE_KO, GRADE_KO, RULES, unionRatio, grade, letterbox, candidates, nms, heatGrid, paperType, buildResult };
})(typeof window !== "undefined" ? window : globalThis);
