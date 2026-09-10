const menu=document.getElementById('menu');
const burger=document.getElementById('burger');
burger.onclick=()=>menu.classList.toggle('open');

function go(id){
  menu.classList.remove('open');
  if(id==='home'){window.scrollTo({top:0,behavior:'smooth'});return;}
  const el=document.getElementById(id);
  const y=el.getBoundingClientRect().top+window.pageYOffset-64-16;
  window.scrollTo({top:y,behavior:'smooth'});
}

const links=document.querySelectorAll('.menu a[data-t]');
window.addEventListener('scroll',()=>{
  let cur='home';
  ['service','tool','collect'].forEach(id=>{
    if(document.getElementById(id).getBoundingClientRect().top-100<=0)cur=id;
  });
  if(window.scrollY<100)cur='home';
  links.forEach(a=>a.classList.toggle('on',a.dataset.t===cur));
},{passive:true});

function submitForm(){
  const name=document.getElementById('name').value.trim();
  const tel=document.getElementById('tel').value.trim();
  if(!name||!tel){alert('이름과 연락처를 입력해 주세요.');return;}
  const b=document.getElementById('submit');
  b.textContent='신청이 접수됐습니다. 곧 연락드리겠습니다.';
  b.disabled=true;
}
const GEMINI_API_KEY="AQ.Ab8RN6KBt3nEl01D4FI7tKB-eINcsxAfpvsgIkbIiN9147gY0w";
const MODEL="gemini-3.6-flash";

const file=document.getElementById('file'),dropInner=document.getElementById('dropInner'),drop=document.getElementById('drop');
const run=document.getElementById('run'),out=document.getElementById('out');
const tabUp=document.getElementById('tabUp'),tabCam=document.getElementById('tabCam');
const upMode=document.getElementById('upMode'),camMode=document.getElementById('camMode');
const video=document.getElementById('video'),camBox=document.getElementById('camBox');
const shot=document.getElementById('shot'),retake=document.getElementById('retake');
let img=null,stream=null;

function setImage(src){
  const i=new Image();
  i.onload=()=>{img=i;run.disabled=false;};
  i.src=src;
}
file.addEventListener('change',e=>{
  const f=e.target.files[0];if(!f)return;
  const r=new FileReader();
  r.onload=()=>{dropInner.innerHTML=`<img src="${r.result}" alt="">`;setImage(r.result);};
  r.readAsDataURL(f);
});
['dragover','dragenter'].forEach(ev=>drop.addEventListener(ev,e=>{e.preventDefault();drop.style.borderColor='var(--green)';}));
['dragleave','drop'].forEach(ev=>drop.addEventListener(ev,e=>{e.preventDefault();drop.style.borderColor='';}));
drop.addEventListener('drop',e=>{const f=e.dataTransfer.files[0];if(f){file.files=e.dataTransfer.files;file.dispatchEvent(new Event('change'));}});

tabUp.onclick=()=>{tabUp.classList.add('on');tabCam.classList.remove('on');upMode.style.display='block';camMode.style.display='none';stopCam();};
tabCam.onclick=async()=>{tabCam.classList.add('on');tabUp.classList.remove('on');upMode.style.display='none';camMode.style.display='block';await startCam();};

async function startCam(){
  const old=camBox.querySelector('img');if(old)old.remove();
  const note=camBox.querySelector('.cam-note');if(note)note.remove();
  video.style.display='block';
  try{
    stream=await navigator.mediaDevices.getUserMedia({video:{facingMode:'environment'}});
    video.srcObject=stream;shot.style.display='block';retake.style.display='none';
  }catch(e){
    video.style.display='none';shot.style.display='none';retake.style.display='none';
    camBox.innerHTML='<div class="cam-note" style="color:#ccc">이 환경에서는 카메라를 쓸 수 없어요. 파일을 다운로드해 브라우저에서 직접 열거나, 사진 올리기를 이용해 주세요.</div>';
  }
}
function stopCam(){if(stream){stream.getTracks().forEach(t=>t.stop());stream=null;}}
shot.onclick=()=>{
  const c=document.createElement('canvas');c.width=video.videoWidth;c.height=video.videoHeight;
  c.getContext('2d').drawImage(video,0,0);
  const url=c.toDataURL('image/jpeg',.92);
  setImage(url);
  video.style.display='none';shot.style.display='none';retake.style.display='block';
  const p=document.createElement('img');p.src=url;camBox.appendChild(p);
  stopCam();
};
retake.onclick=startCam;

function toBase64(i){
  const max=1024,w=i.naturalWidth,h=i.naturalHeight,s=Math.min(1,max/Math.max(w,h));
  const c=document.createElement('canvas');c.width=Math.round(w*s);c.height=Math.round(h*s);
  c.getContext('2d').drawImage(i,0,0,c.width,c.height);
  return c.toDataURL('image/jpeg',.92).split(',')[1];
}

const PROMPT=`당신은 에코리북의 원료 선별 AI입니다. 에코리북은 다 쓴 교육자료(문제집, 교과서, 참고서)를 파쇄해 종이 완충재를 만듭니다. 사진 속 자료가 완충재 원료로 적합한지 판정하세요.

판정 기준:
- 재질: 일반 종이는 적합. 비닐 코팅지, 라미네이팅, 합성지는 완충재 질감을 해치므로 분리 필요. 전체가 코팅이면 부적합.
- 오염: 곰팡이, 기름, 음식물, 심한 습기 자국은 위생 문제로 부적합. 가벼운 얼룩이나 변색은 허용.
- 이물질: 스프링, 스테이플, 테이프, 스티커, 비닐 커버, 플라스틱 인덱스는 떼어내면 쓸 수 있으므로 전처리 대상.
- 필기, 밑줄, 형광펜, 찢어짐, 구겨짐은 파쇄 예정이므로 문제 없음.

각 항목 status는 다음 셋 중 하나입니다.
- "pass": 문제 없음
- "pretreat": 전처리(제거·분리)하면 사용 가능
- "fail": 사용 불가

problem_areas에는 사진에서 발견된 모든 문제 위치를 넣으세요. 좌표는 이미지 전체 기준 0~100 백분율입니다. 스프링이나 긴 테이프처럼 길게 이어진 것은 그 선을 따라 5~10개 좌표를 배열하세요. 각 좌표마다 무엇인지(label), 어떻게 처리할지(action), 성격(kind: "pretreat" 또는 "fail")을 적으세요.

pretreatment_steps에는 작업자가 순서대로 할 전처리 작업을 짧은 문장으로 나열하세요. 없으면 빈 배열.

반드시 마크다운 없이 순수 JSON만 응답하세요:
{
  "is_paper_material": true,
  "material": {"status":"pass","reason":"..."},
  "contamination": {"status":"pass","reason":"..."},
  "foreign": {"status":"pretreat","reason":"..."},
  "problem_areas": [{"x_pct":20,"y_pct":30,"label":"스프링","action":"제거","kind":"pretreat"}],
  "pretreatment_steps": ["좌측 제본부 스프링 제거","표지 코팅지 분리"],
  "overall": "적합" | "전처리 후 적합" | "부적합",
  "summary": "한두 문장 요약"
}`;

run.onclick=async()=>{
  if(!img)return;
  run.disabled=true;run.textContent='선별 중…';
  out.innerHTML='<div class="spin"></div><p class="empty" style="padding:0 0 30px">AI가 원료를 살펴보고 있어요</p>';
  try{
    const res=await fetch(`https://generativelanguage.googleapis.com/v1beta/models/${MODEL}:generateContent?key=${GEMINI_API_KEY}`,{
      method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({contents:[{parts:[{text:PROMPT},{inline_data:{mime_type:'image/jpeg',data:toBase64(img)}}]}]})
    });
    if(!res.ok){const e=await res.json().catch(()=>({}));throw new Error(e.error?.message||'응답 오류 '+res.status);}
    const data=await res.json();
    let txt=data.candidates[0].content.parts[0].text.replace(/```json|```/g,'').trim();
    render(JSON.parse(txt));
  }catch(e){
    console.error(e);
    out.innerHTML=`<div class="err"><b>선별에 실패했어요.</b><br>${esc(e.message)}</div>`;
  }finally{run.disabled=false;run.textContent='선별 시작';}
};

function cls(v){return v==='적합'?'ok':v==='부적합'?'no':'pre';}
function scls(s){return s==='pass'?'ok':s==='fail'?'no':'pre';}
function stxt(s){return s==='pass'?'문제 없음':s==='fail'?'사용 불가':'전처리 필요';}

function render(r){
  const areas=Array.isArray(r.problem_areas)?r.problem_areas:[];
  const steps=Array.isArray(r.pretreatment_steps)?r.pretreatment_steps:[];
  const overall=r.overall||'전처리 후 적합';
  const sub=overall==='적합'?'바로 투입할 수 있습니다':overall==='부적합'?'투입 불가, 폐기 처리':'아래 전처리 작업 후 투입';
  let h=`
    <div class="vbox ${cls(overall)}"><div class="vt">${esc(overall)}</div><div class="vs">${sub}</div></div>
    <p class="summary">${esc(r.summary||'')}</p>`;
  if(areas.length)h+=`<div class="map" id="map"><canvas id="cv"></canvas></div>`;
  h+=`<div class="sec">항목별 판정</div>`;
  [['material','재질'],['contamination','오염'],['foreign','이물질']].forEach(([k,t])=>{
    const v=r[k]||{status:'fail',reason:''};
    h+=`<div class="item"><div><b>${t}</b><div class="r">${esc(v.reason||'')}</div></div><span class="tag ${scls(v.status)}">${stxt(v.status)}</span></div>`;
  });
  h+=`<div class="sec">전처리 작업</div>`;
  if(overall==='부적합'){
    h+=`<div class="none" style="background:var(--bad-light);color:var(--bad)">투입 불가 원료입니다. 폐기 처리해 주세요.</div>`;
  }else if(!steps.length&&!areas.filter(a=>a.kind!=='fail').length){
    h+=`<div class="none">전처리 없이 바로 투입할 수 있어요.</div>`;
  }else{
    const list=steps.length?steps.map((s,i)=>({label:s,action:'',kind:'pretreat'})):dedupe(areas);
    h+=`<ul class="todo">`+list.map((a,i)=>`<li><span class="num ${a.kind==='fail'?'no':''}">${i+1}</span><div><b>${esc(a.label)}</b>${a.action?`<span class="act">${esc(a.action)}</span>`:''}</div></li>`).join('')+`</ul>`;
  }
  out.innerHTML=h;
  if(areas.length)drawMap(areas);
}

function dedupe(areas){
  const seen=new Map();
  areas.forEach(a=>{const k=(a.label||'')+'|'+(a.action||'');if(!seen.has(k))seen.set(k,a);});
  return [...seen.values()];
}

function drawMap(areas){
  const map=document.getElementById('map'),cv=document.getElementById('cv');
  const maxW=640,s=Math.min(1,maxW/img.naturalWidth);
  cv.width=Math.round(img.naturalWidth*s);cv.height=Math.round(img.naturalHeight*s);
  const ctx=cv.getContext('2d');ctx.drawImage(img,0,0,cv.width,cv.height);

  const pts=areas.map(a=>({x:Number(a.x_pct)/100*cv.width,y:Number(a.y_pct)/100*cv.height}));
  const sigma=Math.max(cv.width,cv.height)*.035;
  const gw=120,gh=Math.max(1,Math.round(gw*cv.height/cv.width));
  const sc=document.createElement('canvas');sc.width=gw;sc.height=gh;
  const sctx=sc.getContext('2d'),d=sctx.createImageData(gw,gh);
  for(let gy=0;gy<gh;gy++)for(let gx=0;gx<gw;gx++){
    const px=gx/(gw-1)*cv.width,py=gy/(gh-1)*cv.height;
    let t=0;
    for(const q of pts){const dx=px-q.x,dy=py-q.y;t+=Math.exp(-(dx*dx+dy*dy)/(2*sigma*sigma));}
    t=Math.pow(Math.min(1,t),1.3);
    const [r,g,b]=jet(t);
    const idx=(gy*gw+gx)*4;
    d.data[idx]=r;d.data[idx+1]=g;d.data[idx+2]=b;d.data[idx+3]=Math.round(30+t*180);
  }
  sctx.putImageData(d,0,0);
  ctx.imageSmoothingEnabled=true;
  ctx.drawImage(sc,0,0,gw,gh,0,0,cv.width,cv.height);

  // 번호 핀 (같은 항목은 첫 좌표에만)
  const list=dedupe(areas);
  list.forEach((a,i)=>{
    const p=document.createElement('div');
    p.className='pin '+(a.kind==='fail'?'no':'pre');
    p.style.left=a.x_pct+'%';p.style.top=a.y_pct+'%';
    p.textContent=i+1;p.title=a.label;
    map.appendChild(p);
  });
}

function jet(t){
  const c=v=>Math.max(0,Math.min(1,v));
  return [Math.round(c(1.5-Math.abs(4*t-3))*255),Math.round(c(1.5-Math.abs(4*t-2))*255),Math.round(c(1.5-Math.abs(4*t-1))*255)];
}
function esc(s){const d=document.createElement('div');d.textContent=s;return d.innerHTML;}
