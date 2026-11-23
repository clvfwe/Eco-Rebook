/* 에코리북 브라우저판 — 서버 없이 이 기기의 브라우저가 직접 판정합니다 (GitHub Pages 용)
   모델: models/best.onnx (결함), models/type.onnx (종이 종류).  등급 규칙은 core.js */
const CONFIG = {
  DET_MODEL: "models/best.onnx",
  TYPE_MODEL: "models/type.onnx",
  ORT_DIST: "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/",
  IMGSZ: 800,                // 결함 모델 입력 크기 (models/best.onnx 를 만들 때 정한 값)
  TYPE_IMGSZ: 320,           // 종류 모델 입력 크기
  PREDICT_CONF: 0.15,        // 모델이 후보를 뽑는 기준 (최종 기준은 core.js 의 RULES)
  LOW_CONF: 0.5,             // 미만이면 사람 검수 안내
  SAVE_SIDE: 960,            // 기록에 남기는 사진의 긴 변
  KEEP: 200,                 // 기기에 보관하는 기록 수 (넘으면 오래된 것부터 지움)
  MODEL_ID: "onnx-2026-11-24"
};
const COLORS={"오염":"#ffb54d","스프링":"#ff6b6b","스테이플":"#ff6b6b","테이프":"#ff6b6b","코팅":"#c9a0ff","개인정보":"#6aa8ff"};
const E=window.ECO, $=id=>document.getElementById(id);
let file=null, url=null;

setTimeout(()=>$("splash").classList.add("hide"),1300);

/* ───────── 모델 (페이지를 열면 미리 내려받음) ───────── */
let sessions=null;
function loadModels(){
  if(!sessions){
    ort.env.wasm.wasmPaths=CONFIG.ORT_DIST; ort.env.wasm.numThreads=1;
    sessions=Promise.all([CONFIG.DET_MODEL,CONFIG.TYPE_MODEL].map(p=>ort.InferenceSession.create(p,{executionProviders:["wasm"]})));
    sessions.catch(()=>{sessions=null});
  }
  return sessions;
}
if(window.ort)loadModels().catch(()=>{});

const loadImg=src=>new Promise((ok,no)=>{const im=new Image();im.onload=()=>ok(im);im.onerror=()=>no(new Error("사진을 읽을 수 없어요"));im.src=src});
function canvasOf(w,h){const c=document.createElement("canvas");c.width=w;c.height=h;return c}
function tensorOf(cv){
  const n=cv.width*cv.height, px=cv.getContext("2d").getImageData(0,0,cv.width,cv.height).data, a=new Float32Array(3*n);
  for(let i=0;i<n;i++){a[i]=px[i*4]/255;a[n+i]=px[i*4+1]/255;a[2*n+i]=px[i*4+2]/255}
  return new ort.Tensor("float32",a,[1,3,cv.height,cv.width]);
}
async function run(sess,t){const o=await sess.run({[sess.inputNames[0]]:t});return o[sess.outputNames[0]]}

/* 사진 위에 히트맵을 겹침 (서버판 overlay 와 같은 색·농도) */
function heatOverlay(photo,grid,G){
  const cv=canvasOf(photo.width,photo.height), ctx=cv.getContext("2d"); ctx.drawImage(photo,0,0);
  const im=ctx.getImageData(0,0,cv.width,cv.height), p=im.data, W=cv.width, H=cv.height, cl=v=>Math.min(Math.max(v,0),1);
  for(let y=0;y<H;y++){
    const gy=Math.min(Math.max((y+.5)/H*G-.5,0),G-1), y0=Math.floor(gy), y1=Math.min(y0+1,G-1), fy=gy-y0;
    for(let x=0;x<W;x++){
      const gx=Math.min(Math.max((x+.5)/W*G-.5,0),G-1), x0=Math.floor(gx), x1=Math.min(x0+1,G-1), fx=gx-x0;
      const v=(grid[y0*G+x0]*(1-fx)+grid[y0*G+x1]*fx)*(1-fy)+(grid[y1*G+x0]*(1-fx)+grid[y1*G+x1]*fx)*fy;
      const a=.5*(.35+.65*v), i=(y*W+x)*4;
      p[i]=(1-a)*p[i]+a*255*cl(1.5-Math.abs(4*v-3));
      p[i+1]=(1-a)*p[i+1]+a*255*cl(1.5-Math.abs(4*v-2));
      p[i+2]=(1-a)*p[i+2]+a*255*cl(1.5-Math.abs(4*v-1));
    }
  }
  ctx.putImageData(im,0,0); return cv;
}

async function analyze(f){
  const im=await loadImg(url), W=im.naturalWidth, H=im.naturalHeight;
  const [det,typ]=await loadModels();

  // ① 사실: 결함 탐지
  const S=CONFIG.IMGSZ, lb=E.letterbox(W,H,S), cv=canvasOf(S,S), ctx=cv.getContext("2d");
  ctx.fillStyle="rgb(114,114,114)"; ctx.fillRect(0,0,S,S); ctx.drawImage(im,lb.left,lb.top,lb.nw,lb.nh);
  const out=await run(det,tensorOf(cv));
  const cands=E.candidates(out.data,out.dims[2],out.dims[1]-4,lb,CONFIG.PREDICT_CONF), dets=E.nms(cands);

  // ① 사실: 종이 종류 (짧은 변을 맞춘 뒤 가운데를 잘라 봄)
  const T=CONFIG.TYPE_IMGSZ, tc=canvasOf(T,T), k=T/Math.min(W,H);
  tc.getContext("2d").drawImage(im,(T-W*k)/2,(T-H*k)/2,W*k,H*k);
  const [ptype,pconf]=E.paperType(dets,(await run(typ,tensorOf(tc))).data);

  // ② 규칙: 등급
  const d=new Date(), z=n=>String(n).padStart(2,"0");
  const time=`${d.getFullYear()}-${z(d.getMonth()+1)}-${z(d.getDate())} ${z(d.getHours())}:${z(d.getMinutes())}:${z(d.getSeconds())}`;
  const res=E.buildResult(dets,W,H,ptype,pconf,time,CONFIG.MODEL_ID);

  // 사진 사본 · 히트맵 · 목록용 작은 사진
  const sc=Math.min(1,CONFIG.SAVE_SIDE/Math.max(W,H)), ph=canvasOf(Math.round(W*sc),Math.round(H*sc));
  ph.getContext("2d").drawImage(im,0,0,ph.width,ph.height);
  const G=160, photo=ph.toDataURL("image/jpeg",.82);
  const heat=heatOverlay(ph,E.heatGrid(cands,W,H,E.RULES.confidence_threshold,G),G).toDataURL("image/jpeg",.82);
  const ts=112/Math.min(W,H), th=canvasOf(112,112);
  th.getContext("2d").drawImage(im,(112-W*ts)/2,(112-H*ts)/2,W*ts,H*ts);

  // ③ 기록 저장 (이 기기의 브라우저)
  res.filename=f.name||"";
  try{res.id=await saveRecord(res,th.toDataURL("image/jpeg",.7),photo,heat)}catch(e){res.id=null}
  res.photo=photo; res.heatmap=heat;
  return res;
}

/* ───────── 판정 기록 (IndexedDB: 이 기기의 브라우저에만 저장) ───────── */
let dbp=null;
function db(){
  return dbp||(dbp=new Promise((ok,no)=>{
    const q=indexedDB.open("ecorebook",1);
    q.onupgradeneeded=()=>{q.result.createObjectStore("records",{keyPath:"id",autoIncrement:true});q.result.createObjectStore("images",{keyPath:"id"})};
    q.onsuccess=()=>ok(q.result); q.onerror=()=>no(q.error);
  }));
}
const req=q=>new Promise((ok,no)=>{q.onsuccess=()=>ok(q.result);q.onerror=()=>no(q.error)});
async function saveRecord(res,thumb,photo,heat){
  const d=await db(), tx=d.transaction(["records","images"],"readwrite"), rs=tx.objectStore("records"), is=tx.objectStore("images");
  const id=await req(rs.add({...res,thumb}));
  is.put({id,photo,heat});
  const keys=await req(rs.getAllKeys());
  keys.slice(0,Math.max(0,keys.length-CONFIG.KEEP)).forEach(k=>{rs.delete(k);is.delete(k)});
  await new Promise((ok,no)=>{tx.oncomplete=ok;tx.onerror=()=>no(tx.error)});
  return id;
}
async function allRecords(){return req((await db()).transaction("records").objectStore("records").getAll())}
async function getRecord(id){
  const tx=(await db()).transaction(["records","images"]);
  const [r,im]=await Promise.all([req(tx.objectStore("records").get(id)),req(tx.objectStore("images").get(id))]);
  return r?{...r,photo:im&&im.photo,heatmap:im&&im.heat}:null;
}

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
  show("result"); $("result").innerHTML='<div class="empty"><div class="spin"></div>분석하고 있어요<br><small>처음에는 모델을 내려받느라 조금 더 걸려요</small></div>';
  await new Promise(r=>setTimeout(r,60));          // 안내 문구가 먼저 그려지도록
  try{
    if(!window.ort)throw new Error("분석 도구를 불러오지 못했어요");
    render(normalize(await analyze(file))); loadHist();
  }catch(e){
    show("home");
    err("분석에 실패했어요. 인터넷 연결을 확인하고 다시 시도해 주세요. ("+(e&&e.message||e)+")");
  }finally{$("run").disabled=false}
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

/* 기록 목록 */
async function loadHist(){
  try{
    const items=(await allRecords()).reverse(), n=items.length;
    $("export").hidden=!n; $("histCount").hidden=!n; $("histCount").textContent="전체 "+n+"건 (이 기기에 저장)";
    if(!n)return;
    $("histList").innerHTML=items.map(h=>{
      const df=Object.entries(h.defect_counts).map(([k,c])=>k+" "+c).join(" · ")||"결함 없음";
      return `<button class="hist" data-id="${h.id}"><img src="${h.thumb}" alt="" loading="lazy"><div><b>${esc(h.grade_text)} · ${esc(h.paper_type_text)}</b><small>${esc(h.time.slice(5,16))} · ${esc(df)}</small></div></button>`}).join("");
    document.querySelectorAll(".hist").forEach(b=>b.onclick=async()=>{
      const r=await getRecord(+b.dataset.id); if(r){render(normalize(r)); show("result")}});
  }catch(e){}
}
loadHist();

/* 판정 기록 내보내기 (엑셀에서 바로 열리는 CSV) */
$("export").onclick=async e=>{
  e.preventDefault();
  const q=v=>'"'+String(v??"").replace(/"/g,'""')+'"';
  const rows=[["번호","판정 일시","파일명","종류","등급","전처리 후 예상 등급","결함","오염 면적(%)","전처리 지침","규칙 버전","모델"]];
  (await allRecords()).forEach(x=>rows.push([x.id,x.time,x.filename,x.paper_type_text,x.grade_text,x.grade_after_text||"-",
    Object.entries(x.defect_counts).map(([k,c])=>k+" "+c).join(", ")||"없음",((x.stain_ratio||0)*100).toFixed(1),x.guides.join(" / "),x.rules_version,x.model]));
  const a=document.createElement("a"), d=new Date(), z=n=>String(n).padStart(2,"0");
  a.href=URL.createObjectURL(new Blob(["﻿"+rows.map(r=>r.map(q).join(",")).join("\r\n")],{type:"text/csv;charset=utf-8"}));
  a.download=`ecorebook_records_${d.getFullYear()}${z(d.getMonth()+1)}${z(d.getDate())}_${z(d.getHours())}${z(d.getMinutes())}.csv`;
  a.click(); setTimeout(()=>URL.revokeObjectURL(a.href),1000);
};
