
// Robust canvas helpers and visible error reporting
if (window.CanvasRenderingContext2D && !CanvasRenderingContext2D.prototype.roundRect) {
  CanvasRenderingContext2D.prototype.roundRect = function(x, y, w, h, r) {
    r = Math.min(r || 0, Math.abs(w) / 2, Math.abs(h) / 2);
    this.beginPath();
    this.moveTo(x + r, y);
    this.arcTo(x + w, y, x + w, y + h, r);
    this.arcTo(x + w, y + h, x, y + h, r);
    this.arcTo(x, y + h, x, y, r);
    this.arcTo(x, y, x + w, y, r);
    this.closePath();
    return this;
  };
}
function showCanvasError(id, message){
  const canvas=document.getElementById(id); if(!canvas) return;
  const ctx=canvas.getContext('2d'), w=canvas.width, h=canvas.height;
  ctx.clearRect(0,0,w,h); ctx.fillStyle='#fbfffe'; ctx.fillRect(0,0,w,h);
  ctx.fillStyle='#b42318'; ctx.font='800 18px Inter, system-ui, sans-serif';
  ctx.fillText('Demo rendering error', 32, 54);
  ctx.fillStyle='#66758a'; ctx.font='700 13px Inter, system-ui, sans-serif';
  const lines=String(message || 'Unknown error').match(/.{1,82}/g) || [];
  lines.slice(0,6).forEach((line,i)=>ctx.fillText(line,32,86+i*22));
}
window.addEventListener('error', (e)=>{
  console.error(e.error || e.message);
  showCanvasError('conditionMapCanvas', e.message);
  showCanvasError('forwardCanvas', e.message);
});
window.addEventListener('unhandledrejection', (e)=>{
  console.error(e.reason);
  showCanvasError('conditionMapCanvas', e.reason?.message || e.reason);
  showCanvasError('forwardCanvas', e.reason?.message || e.reason);
});

const CELL_COLORS = {"Type 1":"#46bfe5","Type 2":"#5ecf91","Type 3":"#ff8a73","Type 4":"#9a89ec"};
const TRAJ_CELL_RADIUS = 8.6;
const TRAJ_GHOST_RADIUS = TRAJ_CELL_RADIUS;
const TRAJ_HALO_RADIUS = TRAJ_CELL_RADIUS;
const TRAJ_DIVIDE_RADIUS = 8.0;
const TRAJ_DIVIDE_SMALL_RADIUS = 7.2;
const ALPHA_STOPS = [
  {t:0, c:[117,101,220]},
  {t:0.55, c:[31,183,201]},
  {t:1, c:[255,138,115]}
];

function lerp(a,b,t){return a+(b-a)*t}
function clamp(x,a,b){return Math.max(a,Math.min(b,x))}
function smoothstep(t){return t*t*(3-2*t)}
function formatNum(x,d=3){if(!Number.isFinite(+x)) return "—"; const v=Math.abs(+x); if(v>=100) return (+x).toFixed(1); if(v>=10) return (+x).toFixed(2); return (+x).toFixed(d)}
function pad3(x){return String(x).padStart(3,"0")}
async function loadJson(path){const r=await fetch(path,{cache:"no-store"}); if(!r.ok) throw new Error(path); return r.json()}
async function loadJsonWithFallback(path,fallback){try{return await loadJson(path)}catch(e){return fallback}}

function colorForAlpha(value,min,max){
  const u = clamp((value-min)/(max-min || 1),0,1);
  let a = ALPHA_STOPS[0], b = ALPHA_STOPS[ALPHA_STOPS.length-1];
  for(let i=0;i<ALPHA_STOPS.length-1;i++){ if(u>=ALPHA_STOPS[i].t && u<=ALPHA_STOPS[i+1].t){ a=ALPHA_STOPS[i]; b=ALPHA_STOPS[i+1]; break; } }
  const local=(u-a.t)/(b.t-a.t || 1);
  const c=a.c.map((v,i)=>Math.round(lerp(v,b.c[i],local)));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

function setupTopbar(){
  const t=document.getElementById("topbar");
  const f=()=>t.classList.toggle("scrolled",scrollY>40);
  f(); addEventListener("scroll",f,{passive:true});
}

function drawGrid(ctx,w,h,labels={x:"X1",y:"X2"}){
  ctx.save();
  ctx.clearRect(0,0,w,h);
  ctx.fillStyle="#fbfffe"; ctx.fillRect(0,0,w,h);
  ctx.strokeStyle="rgba(31,64,87,.07)"; ctx.lineWidth=1;
  for(let x=70;x<w;x+=90){ctx.beginPath();ctx.moveTo(x,48);ctx.lineTo(x,h-60);ctx.stroke()}
  for(let y=70;y<h;y+=75){ctx.beginPath();ctx.moveTo(54,y);ctx.lineTo(w-54,y);ctx.stroke()}
  const g=ctx.createRadialGradient(w*.52,h*.45,20,w*.52,h*.45,w*.65);
  g.addColorStop(0,"rgba(31,183,201,.09)");
  g.addColorStop(.55,"rgba(107,211,155,.045)");
  g.addColorStop(1,"rgba(255,255,255,0)");
  ctx.fillStyle=g; ctx.fillRect(0,0,w,h);
  ctx.fillStyle="rgba(16,32,51,.72)"; ctx.font="800 17px Inter, system-ui, sans-serif";
  ctx.fillText(labels.y,28,40); ctx.fillText(labels.x,w-88,h-30);
  ctx.restore();
}
function getTrajectoryBounds(frames){
  const xs=[], ys=[];
  for(const f of frames){ xs.push(...f.x); ys.push(...f.y); }
  const minX=Math.min(...xs), maxX=Math.max(...xs), minY=Math.min(...ys), maxY=Math.max(...ys);
  const dx=Math.max(0.01,maxX-minX), dy=Math.max(0.01,maxY-minY);
  return {minX:minX-.08*dx,maxX:maxX+.08*dx,minY:minY-.08*dy,maxY:maxY+.08*dy};
}
function getEventTrajectoryBounds(transitions){
  const xs=[], ys=[];
  for(const tr of transitions){ xs.push(...tr.x0,...tr.x1); ys.push(...tr.y0,...tr.y1); }
  const minX=Math.min(...xs), maxX=Math.max(...xs), minY=Math.min(...ys), maxY=Math.max(...ys);
  const dx=Math.max(0.01,maxX-minX), dy=Math.max(0.01,maxY-minY);
  return {minX:minX-.08*dx,maxX:maxX+.08*dx,minY:minY-.08*dy,maxY:maxY+.08*dy};
}
function scaleXY(x,y,b,w,h){
  const pad=78;
  const s=Math.min((w-pad*2)/(b.maxX-b.minX || 1),(h-pad*2)/(b.maxY-b.minY || 1));
  const cx=(b.minX+b.maxX)/2, cy=(b.minY+b.maxY)/2;
  return [w/2+(x-cx)*s,h/2-(y-cy)*s];
}
function drawCell(ctx,x,y,r,color,alpha=.85){
  ctx.save(); ctx.globalAlpha=alpha;
  ctx.beginPath(); ctx.arc(x,y,r,0,Math.PI*2); ctx.fillStyle=color; ctx.fill();
  ctx.lineWidth=0.9; ctx.strokeStyle="rgba(255,255,255,.86)"; ctx.stroke();
  ctx.beginPath(); ctx.arc(x-r*.18,y-r*.14,r*.33,0,Math.PI*2); ctx.fillStyle="rgba(255,255,255,.42)"; ctx.fill();
  ctx.restore();
}
function drawDeathMarker(ctx,x,y,u,alpha=.9){
  const p=clamp((u-.08)/.82,0,1);
  const ring=1-Math.abs(p-.46)/.46;
  const burst=smoothstep(clamp((u-.18)/.54,0,1));
  ctx.save();
  ctx.globalAlpha=alpha;
  ctx.strokeStyle="rgba(255,82,82,.86)";
  ctx.lineCap="round";
  ctx.lineWidth=1.8;
  ctx.beginPath();
  ctx.arc(x,y,5+13*burst,0,Math.PI*2);
  ctx.strokeStyle=`rgba(255,82,82,${0.48*Math.max(0,ring)})`;
  ctx.stroke();

  ctx.strokeStyle=`rgba(180,35,24,${0.66*(1-p)})`;
  ctx.lineWidth=2.0;
  const arm=4+5*burst;
  ctx.beginPath();
  ctx.moveTo(x-arm,y-arm); ctx.lineTo(x+arm,y+arm);
  ctx.moveTo(x+arm,y-arm); ctx.lineTo(x-arm,y+arm);
  ctx.stroke();

  ctx.strokeStyle=`rgba(255,138,115,${0.42*(1-p)})`;
  ctx.lineWidth=1.25;
  for(let i=0;i<6;i++){
    const a=i*Math.PI/3;
    const r0=4+4*burst, r1=8+10*burst;
    ctx.beginPath();
    ctx.moveTo(x+Math.cos(a)*r0,y+Math.sin(a)*r0);
    ctx.lineTo(x+Math.cos(a)*r1,y+Math.sin(a)*r1);
    ctx.stroke();
  }
  ctx.restore();
}
function interpolateFrame(frames,timeMs){
  if(!frames || frames.length===0) return null;
  if(frames.length===1) return frames[0];
  const cycle=frames.length-1;
  const phase=(timeMs/680)%cycle;
  const k=Math.floor(phase), u=smoothstep(phase-k);
  const a=frames[k], b=frames[k+1];
  const n=Math.min(a.x.length,b.x.length);
  const out={x:new Array(n),y:new Array(n),z:new Array(n),weight:new Array(n),type:a.type||[],id:a.id||[]};
  for(let i=0;i<n;i++){
    out.x[i]=lerp(a.x[i],b.x[i],u);
    out.y[i]=lerp(a.y[i],b.y[i],u);
    out.z[i]=lerp(a.z?.[i]??0,b.z?.[i]??0,u);
    out.weight[i]=lerp(a.weight[i],b.weight[i],u);
  }
  out.stepLabel=`${a.step} → ${b.step}`;
  out.progress=(k+u)/(frames.length-1);
  out.stepValue=lerp(+a.step,+b.step,u);
  return out;
}
function weightAlpha(w,min,max){
  // Log scaling keeps low-mass cells visible while preserving weight-driven opacity.
  const eps=1e-12;
  const lo=Math.log(Math.max(min,eps)), hi=Math.log(Math.max(max,eps)), ww=Math.log(Math.max(w,eps));
  const u=clamp((ww-lo)/(hi-lo || 1),0,1);
  return 0.22+0.72*Math.sqrt(u);
}
function hashUnit(value){
  let h=2166136261;
  const s=String(value);
  for(let i=0;i<s.length;i++){
    h^=s.charCodeAt(i);
    h=Math.imul(h,16777619);
  }
  return (h>>>0)/4294967295;
}
function visualOffset(id, magnitude=4.4){
  const angle=hashUnit(`${id}:angle`)*Math.PI*2;
  const radius=(.35+.65*hashUnit(`${id}:radius`))*magnitude;
  return [Math.cos(angle)*radius, Math.sin(angle)*radius];
}
function childIdsFor(tr,i,offspring){
  const raw=String(tr.children?.[i] || "");
  const ids=raw.split(";").filter(Boolean);
  while(ids.length<offspring) ids.push(`${tr.id?.[i] || i}:child:${ids.length}`);
  return ids.slice(0,offspring);
}
function transitionFrame(transitions,timeMs){
  if(!transitions || transitions.length===0) return null;
  const phase=(timeMs/680)%transitions.length;
  const k=Math.floor(phase);
  const u=smoothstep(phase-k);
  const tr=transitions[k];
  return {tr,u,stepValue:lerp(+tr.source_step,+tr.target_step,u)};
}
function dominantFate(c){
  const vals=[['Type 1',+c.type1_prop||0],['Type 2',+c.type2_prop||0],['Type 3',+c.type3_prop||0],['Type 4',+c.type4_prop||0]];
  vals.sort((a,b)=>b[1]-a[1]);
  return {type:vals[0][0], value:vals[0][1]};
}
function pct(value){return `${Math.round((+value || 0)*100)}%`}
function signedNum(value,d=2){const v=+value || 0; return `${v>=0?"+":""}${formatNum(v,d)}`}
function typeLegendName(type){return type || "Type 4"}

function drawTypeLegendOverlay(ctx,w,h,{x=w-190,y=142,scale=1.34}={}){
  const items=Object.entries(CELL_COLORS);
  const boxW=126*scale, rowH=22*scale, boxH=(32+items.length*22)*scale;
  ctx.save();
  ctx.fillStyle="rgba(255,255,255,.86)";
  ctx.strokeStyle="rgba(31,183,201,.18)";
  ctx.lineWidth=1;
  ctx.beginPath();
  ctx.roundRect(x,y,boxW,boxH,16*scale);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle="#08788a";
  ctx.font=`950 ${12*scale}px Inter, system-ui, sans-serif`;
  ctx.textAlign="left";
  ctx.textBaseline="alphabetic";
  ctx.fillText("Cell type",x+12*scale,y+21*scale);
  ctx.font=`900 ${12*scale}px Inter, system-ui, sans-serif`;
  for(let i=0;i<items.length;i++){
    const [type,color]=items[i];
    const cy=y+39*scale+i*rowH;
    ctx.beginPath();
    ctx.arc(x+16*scale,cy-4*scale,5.5*scale,0,Math.PI*2);
    ctx.fillStyle=color;
    ctx.fill();
    ctx.strokeStyle="rgba(255,255,255,.88)";
    ctx.stroke();
    ctx.fillStyle="#526478";
    ctx.fillText(type,x+28*scale,cy);
  }
  ctx.restore();
}

function fallbackConditionIndex(){
  const out=[]; let seed=13; const rand=()=>{seed=(seed*1664525+1013904223)%4294967296; return seed/4294967296};
  for(let i=1;i<=100;i++){
    const rho1=Math.pow(10, rand()*3), rho2=Math.pow(10, rand()*3), alpha_g=-0.39+rand()*0.29;
    out.push({condition:i,label:`Condition ${pad3(i)}`,rho1,rho2,alpha_g,mass_fold_change:0.65+rand()*0.9,mean_X1_shift:rand()-.5,mean_X2_shift:rand()-.5,mean_X3_shift:rand()-.5,type1_prop:.25,type2_prop:.25,type3_prop:.25,type4_prop:.25,trajectory:`data/trajectories/condition_${pad3(i)}.json`});
  }
  return out;
}
function fallbackTrajectory(condition){
  let seed=condition.condition*97+11; const rand=()=>{seed=(seed*1664525+1013904223)%4294967296; return seed/4294967296};
  const n=160, steps=10;
  const base=[];
  for(let i=0;i<n;i++) base.push({x:3.8+(rand()-.5)*.9,y:1.2+(rand()-.5)*.8,z:1+(rand()-.5)*.5,type:["Type 1","Type 2","Type 3","Type 4"][Math.floor(rand()*4)]});
  const e1=Math.tanh(Math.log10(condition.rho1)-1.4), e2=Math.tanh(Math.log10(condition.rho2)-1.2), ea=condition.alpha_g+.24;
  const frames=[];
  for(let s=0;s<steps;s++){
    const t=s/(steps-1), sm=smoothstep(t), f={step:s,t,id:[],x:[],y:[],z:[],weight:[],type:[]};
    for(let i=0;i<n;i++){
      const b=base[i]; f.id.push(i); f.x.push(b.x+sm*(.65*e1-.18*e2)+.12*Math.sin(b.y*2+sm)); f.y.push(b.y+sm*(.7*e2-.16*e1)+.12*Math.cos(b.x*2-sm)); f.z.push(b.z+sm*(-1.2*ea)); f.weight.push(Math.exp(sm*(-1.6*ea+.1*Math.sin(i)))/n); f.type.push(b.type);
    }
    frames.push(f);
  }
  return {condition:condition.condition,label:condition.label,params:{rho1:condition.rho1,rho2:condition.rho2,alpha_g:condition.alpha_g},frames};
}

function normalizeConditionPayload(payload){
  if(Array.isArray(payload)) return {conditions:payload, trajectory_bounds:null, weight_bounds:null};
  return {
    conditions: payload.conditions || [],
    trajectory_bounds: payload.trajectory_bounds || payload.bounds || null,
    weight_bounds: payload.weight_bounds || null
  };
}

function setupConditionAtlasDemo(index, meta={}){
  index = (Array.isArray(index) ? index : []).filter(c => {
    const cid = Number(c.condition ?? c.samples);
    const rho1 = Number(c.rho1);
    const rho2 = Number(c.rho2);
    const alpha = Number(c.alpha_g);
    return Number.isFinite(cid) && cid > 0 && Number.isFinite(rho1) && Number.isFinite(rho2) && Number.isFinite(alpha) && rho1 > 0 && rho2 > 0;
  }).sort((a,b)=>Number(a.condition)-Number(b.condition));
  if(index.length===0) throw new Error("No valid conditions found in data/demo1_condition_index.json");

  const mapCanvas = document.getElementById("conditionMapCanvas");
  const colorbarCanvas = document.getElementById("conditionColorbarCanvas");
  const trajCanvas = document.getElementById("forwardCanvas");
  const mapCtx = mapCanvas.getContext("2d");
  const colorbarCtx = colorbarCanvas.getContext("2d");
  const trajCtx = trajCanvas.getContext("2d");
  const select = document.getElementById("conditionSelect");
  const info = document.getElementById("conditionInfoPanel");
  const stepLabel = document.getElementById("trajectoryStepLabel");
  const outcome = document.getElementById("demo1Outcome");
  const mapHint = document.getElementById("demo1MapHint");

  let selected = index.find(c => Number(c.condition) === 37) || index[0];
  let trajectory = null;
  let trajBounds = null;
  let weightMin = 0;
  let weightMax = 1;
  let hover = null;
  let mapPoints = [];

  const alphaMin = Math.min(...index.map(c => +c.alpha_g));
  const alphaMax = Math.max(...index.map(c => +c.alpha_g));
  const xVals = index.map(c => Math.log10(+c.rho1));
  const yVals = index.map(c => Math.log10(+c.rho2));
  const mapBounds = {
    minX: Math.min(...xVals), maxX: Math.max(...xVals),
    minY: Math.min(...yVals), maxY: Math.max(...yVals)
  };

  select.innerHTML = "";
  for(const c of index){
    const o = document.createElement("option");
    o.value = c.condition;
    o.textContent = c.label || `Condition ${pad3(c.condition)}`;
    select.appendChild(o);
  }

  function mapGeometry(){
    const w = mapCanvas.width, h = mapCanvas.height;
    const base = {left: 82, right: 34, top: 34, bottom: 74};
    const availW = w - base.left - base.right;
    const availH = h - base.top - base.bottom;
    const size = Math.min(availW, availH);
    const padL = base.left + (availW - size) / 2;
    const padT = base.top + (availH - size) / 2;
    return {w, h, padL, padT, plotW: size, plotH: size, padR: w - padL - size, padB: h - padT - size};
  }

  function mapScale(logx, logy){
    const g = mapGeometry();
    const x = g.padL + (logx - mapBounds.minX) / (mapBounds.maxX - mapBounds.minX || 1) * g.plotW;
    const y = g.padT + (1 - (logy - mapBounds.minY) / (mapBounds.maxY - mapBounds.minY || 1)) * g.plotH;
    return [x, y];
  }

  function drawSoftMapBackground(ctx, w, h){
    ctx.clearRect(0,0,w,h);
    ctx.fillStyle = "#fbfffe";
    ctx.fillRect(0,0,w,h);
    const glow1 = ctx.createRadialGradient(w*.28,h*.26,10,w*.28,h*.26,w*.52);
    glow1.addColorStop(0,"rgba(31,183,201,.12)");
    glow1.addColorStop(1,"rgba(31,183,201,0)");
    ctx.fillStyle = glow1; ctx.fillRect(0,0,w,h);
    const glow2 = ctx.createRadialGradient(w*.72,h*.68,10,w*.72,h*.68,w*.56);
    glow2.addColorStop(0,"rgba(107,211,155,.14)");
    glow2.addColorStop(1,"rgba(107,211,155,0)");
    ctx.fillStyle = glow2; ctx.fillRect(0,0,w,h);
  }

  function drawMap(){
    const {w, h, padL, padT, plotW, plotH} = mapGeometry();
    drawSoftMapBackground(mapCtx,w,h);

    // grid
    mapCtx.save();
    mapCtx.strokeStyle = "rgba(31,64,87,.085)";
    mapCtx.lineWidth = 1;
    const ticks = 4;
    for(let i=0;i<=ticks;i++){
      const x = padL + i/ticks*plotW;
      const y = padT + i/ticks*plotH;
      mapCtx.beginPath(); mapCtx.moveTo(x,padT); mapCtx.lineTo(x,padT+plotH); mapCtx.stroke();
      mapCtx.beginPath(); mapCtx.moveTo(padL,y); mapCtx.lineTo(padL+plotW,y); mapCtx.stroke();
    }

    // axes
    mapCtx.strokeStyle = "rgba(16,32,51,.34)";
    mapCtx.lineWidth = 1.25;
    mapCtx.beginPath(); mapCtx.moveTo(padL,padT); mapCtx.lineTo(padL,padT+plotH); mapCtx.lineTo(padL+plotW,padT+plotH); mapCtx.stroke();

    // axis titles
    mapCtx.fillStyle = "#102033";
    mapCtx.font = "950 24px Inter, system-ui, sans-serif";
    mapCtx.textAlign = "center";
    mapCtx.textBaseline = "alphabetic";
    mapCtx.fillText("log₁₀ ρ₁", padL + plotW / 2, h - 18);
    mapCtx.save();
    mapCtx.translate(30, padT + plotH / 2);
    mapCtx.rotate(-Math.PI / 2);
    mapCtx.fillText("log₁₀ ρ₂", 0, 0);
    mapCtx.restore();

    // tick marks
    mapCtx.strokeStyle = "rgba(16,32,51,.20)";
    mapCtx.lineWidth = 1;
    for(let i=0;i<=ticks;i++){
      const x = padL + i/ticks*plotW;
      const y = padT + plotH - i/ticks*plotH;
      mapCtx.beginPath(); mapCtx.moveTo(x, padT+plotH); mapCtx.lineTo(x, padT+plotH+5); mapCtx.stroke();
      mapCtx.beginPath(); mapCtx.moveTo(padL-5, y); mapCtx.lineTo(padL, y); mapCtx.stroke();
    }

    // subtle density halos first
    mapPoints = [];
    for(const c of index){
      const [x,y] = mapScale(Math.log10(+c.rho1), Math.log10(+c.rho2));
      mapPoints.push({x,y,c});
      mapCtx.beginPath();
      mapCtx.arc(x,y,10,0,Math.PI*2);
      mapCtx.fillStyle = colorForAlpha(+c.alpha_g, alphaMin, alphaMax);
      mapCtx.globalAlpha = 0.09;
      mapCtx.fill();
    }
    mapCtx.globalAlpha = 1;

    // points
    for(const p of mapPoints){
      const c = p.c;
      const isSelected = Number(c.condition) === Number(selected.condition);
      const isHover = hover && Number(c.condition) === Number(hover.condition);
      const r = isSelected ? 7.2 : isHover ? 6.6 : 5.4;
      mapCtx.beginPath();
      mapCtx.arc(p.x,p.y,r,0,Math.PI*2);
      mapCtx.fillStyle = colorForAlpha(+c.alpha_g, alphaMin, alphaMax);
      mapCtx.globalAlpha = isHover || isSelected ? 1 : .88;
      mapCtx.fill();
      mapCtx.globalAlpha = 1;
      mapCtx.lineWidth = isHover ? 2.2 : 1.1;
      mapCtx.strokeStyle = isHover ? "rgba(16,32,51,.55)" : "rgba(255,255,255,.92)";
      mapCtx.stroke();
    }

    // hover ring
    if(hover && Number(hover.condition) !== Number(selected.condition)){
      const p = mapPoints.find(p => Number(p.c.condition) === Number(hover.condition));
      if(p){
        mapCtx.beginPath();
        mapCtx.arc(p.x,p.y,11.5,0,Math.PI*2);
        mapCtx.strokeStyle = "rgba(16,32,51,.54)";
        mapCtx.lineWidth = 2;
        mapCtx.stroke();
      }
    }

    // selected ring, white halo, and tiny crosshair
    const sp = mapPoints.find(p => Number(p.c.condition) === Number(selected.condition));
    if(sp){
      mapCtx.beginPath();
      mapCtx.arc(sp.x,sp.y,18.5,0,Math.PI*2);
      mapCtx.strokeStyle = "rgba(255,255,255,.96)";
      mapCtx.lineWidth = 7;
      mapCtx.stroke();
      mapCtx.beginPath();
      mapCtx.arc(sp.x,sp.y,18.5,0,Math.PI*2);
      mapCtx.strokeStyle = "#102033";
      mapCtx.lineWidth = 3.4;
      mapCtx.stroke();
      mapCtx.strokeStyle = "rgba(16,32,51,.45)";
      mapCtx.lineWidth = 1;
      mapCtx.beginPath(); mapCtx.moveTo(sp.x-26,sp.y); mapCtx.lineTo(sp.x-16,sp.y); mapCtx.moveTo(sp.x+16,sp.y); mapCtx.lineTo(sp.x+26,sp.y); mapCtx.moveTo(sp.x,sp.y-26); mapCtx.lineTo(sp.x,sp.y-16); mapCtx.moveTo(sp.x,sp.y+16); mapCtx.lineTo(sp.x,sp.y+26); mapCtx.stroke();
    }

    // minimal hover chip: condition label only, no explanatory captions
    if(hover){
      const hp = mapPoints.find(p => Number(p.c.condition) === Number(hover.condition));
      if(hp){
        const label = hover.label || `Condition ${pad3(hover.condition)}`;
        mapCtx.font = "950 12px Inter, system-ui, sans-serif";
        const tw = Math.max(110, mapCtx.measureText(label).width + 28);
        const tx = clamp(hp.x + 13, 82, w - tw - 12);
        const ty = clamp(hp.y - 42, 14, h - 62);
        mapCtx.fillStyle = "rgba(255,255,255,.94)";
        mapCtx.strokeStyle = "rgba(31,183,201,.18)";
        mapCtx.lineWidth = 1;
        mapCtx.beginPath(); mapCtx.roundRect(tx,ty,tw,34,12); mapCtx.fill(); mapCtx.stroke();
        mapCtx.fillStyle = "#102033";
        mapCtx.textAlign = "left"; mapCtx.textBaseline = "middle";
        mapCtx.fillText(label, tx+14, ty+18);
      }
    }
    mapCtx.restore();
  }

  function syncColorbarHeight(){
    // Keep the colorbar visually aligned with the square condition-map canvas.
    // CSS determines the map size from the current layout; this function makes
    // the colorbar follow that exact displayed height instead of stretching to
    // the full panel.
    const rect = mapCanvas.getBoundingClientRect();
    if(rect && rect.height > 0){
      const h = Math.round(rect.height);
      colorbarCanvas.style.height = `${h}px`;
      colorbarCanvas.parentElement.style.height = `${h}px`;
    }
  }

  function drawColorbar(){
    syncColorbarHeight();
    const w = colorbarCanvas.width, h = colorbarCanvas.height;
    colorbarCtx.clearRect(0,0,w,h);
    colorbarCtx.fillStyle = "#fbfffe";
    colorbarCtx.fillRect(0,0,w,h);
    const barX = 16, barY = 44, barW = 28, barH = h - 92;
    const grad = colorbarCtx.createLinearGradient(0,barY+barH,0,barY);
    for(let i=0;i<=40;i++){
      const u = i/40;
      grad.addColorStop(u, colorForAlpha(alphaMin + u*(alphaMax-alphaMin), alphaMin, alphaMax));
    }
    colorbarCtx.fillStyle = grad;
    colorbarCtx.beginPath();
    colorbarCtx.roundRect(barX,barY,barW,barH,11);
    colorbarCtx.fill();
    colorbarCtx.strokeStyle = "rgba(16,32,51,.18)";
    colorbarCtx.lineWidth = 1;
    colorbarCtx.stroke();

    colorbarCtx.fillStyle = "#102033";
    colorbarCtx.font = "950 25px Inter, system-ui, sans-serif";
    colorbarCtx.textAlign = "center";
    colorbarCtx.textBaseline = "alphabetic";
    colorbarCtx.fillText("α", barX + barW/2, 31);

    const ticks = [
      {v:alphaMax, y:barY, align:"top"},
      {v:(alphaMin+alphaMax)/2, y:barY+barH/2, align:"middle"},
      {v:alphaMin, y:barY+barH, align:"bottom"}
    ];
    colorbarCtx.font = "900 18px Inter, system-ui, sans-serif";
    colorbarCtx.fillStyle = "#607188";
    colorbarCtx.textAlign = "left";
    for(const t of ticks){
      colorbarCtx.strokeStyle = "rgba(16,32,51,.20)";
      colorbarCtx.beginPath(); colorbarCtx.moveTo(barX+barW+4,t.y); colorbarCtx.lineTo(barX+barW+10,t.y); colorbarCtx.stroke();
      colorbarCtx.textBaseline = t.align;
      colorbarCtx.fillText(t.v.toFixed(2), barX+barW+12, t.y);
    }
  }

  function updateInfo(){
    const label = selected.label || `Condition ${pad3(selected.condition)}`;
    info.innerHTML = `
      <div class="compact-condition-line">
        <strong>${label}</strong>
        <span>&rho;<sub>1</sub> <em>${formatNum(selected.rho1,3)}</em></span>
        <span>&rho;<sub>2</sub> <em>${formatNum(selected.rho2,3)}</em></span>
        <span>&alpha; <em>${formatNum(selected.alpha_g,3)}</em></span>
      </div>`;
  }
  function updateOutcome(){
    if(!outcome) return;
    const fate=dominantFate(selected);
    const mass=+selected.mass_fold_change || 1;
    outcome.innerHTML=`
      <div class="outcome-lede">
        <span>Selected endpoint</span>
        <b>${typeLegendName(fate.type)} dominates at ${pct(fate.value)}</b>
      </div>
      <div class="outcome-metrics">
        <span><em>mass</em><b>${formatNum(mass,2)}x</b></span>
        <span><em>X1 shift</em><b>${signedNum(selected.mean_X1_shift,2)}</b></span>
        <span><em>X2 shift</em><b>${signedNum(selected.mean_X2_shift,2)}</b></span>
      </div>`;
  }

  async function loadSelected(c,userDriven=false){
    selected = c;
    select.value = String(c.condition);
    if(userDriven) mapHint?.classList.add("is-dismissed");
    updateInfo();
    updateOutcome();
    drawMap();
    try { trajectory = await loadJson(c.trajectory || `data/trajectories/condition_${pad3(c.condition)}.json`); }
    catch(e) { trajectory = fallbackTrajectory(c); }
    const localBounds = trajectory.transitions?.length ? getEventTrajectoryBounds(trajectory.transitions) : getTrajectoryBounds(trajectory.frames);
    trajBounds = meta.trajectory_bounds || localBounds;
    const weights = trajectory.transitions?.length
      ? trajectory.transitions.flatMap(tr => [...(tr.w0 || []), ...(tr.w1 || [])])
      : trajectory.frames.flatMap(f => f.weight || []);
    weightMin = (meta.weight_bounds && Number.isFinite(+meta.weight_bounds.min)) ? +meta.weight_bounds.min : Math.min(...weights);
    weightMax = (meta.weight_bounds && Number.isFinite(+meta.weight_bounds.max)) ? +meta.weight_bounds.max : Math.max(...weights);
  }

  function nearestPoint(mx,my){
    let best=null, bestD=Infinity;
    for(const p of mapPoints){
      const d=(p.x-mx)**2+(p.y-my)**2;
      if(d<bestD){bestD=d; best=p;}
    }
    return bestD < 22*22 ? best.c : null;
  }

  mapCanvas.addEventListener("click", e => {
    const r = mapCanvas.getBoundingClientRect();
    const mx = (e.clientX-r.left) * mapCanvas.width / r.width;
    const my = (e.clientY-r.top) * mapCanvas.height / r.height;
    const c = nearestPoint(mx,my);
    if(c) {
      loadSelected(c,true);
    }
  });
  mapCanvas.addEventListener("mousemove", e => {
    const r = mapCanvas.getBoundingClientRect();
    const mx = (e.clientX-r.left) * mapCanvas.width / r.width;
    const my = (e.clientY-r.top) * mapCanvas.height / r.height;
    const c = nearestPoint(mx,my);
    mapCanvas.style.cursor = c ? "pointer" : "default";
    if((c?.condition || null) !== (hover?.condition || null)){
      hover = c;
      drawMap();
    }
  });
  mapCanvas.addEventListener("mouseleave", () => { hover = null; drawMap(); });
  select.addEventListener("change", () => {
    const c = index.find(x => String(x.condition) === select.value);
    if(c) loadSelected(c,true);
  });

  function drawTrajectoryAxes(ctx,w,h,bounds){
    ctx.clearRect(0,0,w,h);
    ctx.fillStyle = "#fbfffe";
    ctx.fillRect(0,0,w,h);
    const base = {left: 88, right: 42, top: 46, bottom: 78};
    const availW = w - base.left - base.right;
    const availH = h - base.top - base.bottom;
    const size = Math.min(availW, availH);
    const padL = base.left + (availW - size) / 2;
    const padT = base.top + (availH - size) / 2;
    const plotW = size, plotH = size;
    const padR = w - padL - plotW, padB = h - padT - plotH;
    const glow = ctx.createRadialGradient(w*.52,h*.46,20,w*.52,h*.46,w*.72);
    glow.addColorStop(0,"rgba(31,183,201,.075)");
    glow.addColorStop(.55,"rgba(107,211,155,.04)");
    glow.addColorStop(1,"rgba(255,255,255,0)");
    ctx.fillStyle = glow; ctx.fillRect(0,0,w,h);

    ctx.strokeStyle = "rgba(31,64,87,.075)";
    ctx.lineWidth = 1;
    const ticks=5;
    for(let i=0;i<=ticks;i++){
      const x=padL+i/ticks*plotW;
      const y=padT+i/ticks*plotH;
      ctx.beginPath(); ctx.moveTo(x,padT); ctx.lineTo(x,padT+plotH); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(padL,y); ctx.lineTo(padL+plotW,y); ctx.stroke();
    }
    ctx.strokeStyle = "rgba(16,32,51,.34)";
    ctx.lineWidth = 1.25;
    ctx.beginPath(); ctx.moveTo(padL,padT); ctx.lineTo(padL,padT+plotH); ctx.lineTo(padL+plotW,padT+plotH); ctx.stroke();
    ctx.fillStyle = "#102033";
    ctx.font = "950 19px Inter, system-ui, sans-serif";
    ctx.textAlign = "center"; ctx.textBaseline="alphabetic";
    ctx.fillText("X1", padL+plotW/2, h-22);
    ctx.save(); ctx.translate(30,padT+plotH/2); ctx.rotate(-Math.PI/2); ctx.fillText("X2",0,0); ctx.restore();
    return {padL,padR,padT,padB,plotW,plotH};
  }

  function pointXY(x,y,b,w,h,p){
    const sx = p.padL + (x-b.minX)/(b.maxX-b.minX || 1)*p.plotW;
    const sy = p.padT + (1-(y-b.minY)/(b.maxY-b.minY || 1))*p.plotH;
    return [sx,sy];
  }

  function drawBranchingTrajectory(time,w,h,pads){
    const tf = transitionFrame(trajectory.transitions,time);
    if(!tf) return;
    const {tr,u,stepValue} = tf;
    if(stepLabel) stepLabel.textContent = `step = ${Math.round(stepValue ?? 0)}`;

    const first = trajectory.transitions[0];
    for(let i=0;i<first.x0.length;i+=2){
      const [x0,y0] = pointXY(first.x0[i], first.y0[i], trajBounds, w, h, pads);
      drawCell(trajCtx,x0,y0,TRAJ_GHOST_RADIUS,"#b8c9d5",0.15);
    }

    const last = trajectory.transitions[trajectory.transitions.length-1];
    let haloIndex=0;
    for(let i=0;i<last.x1.length;i+=4){
      if((last.offspring_count?.[i] || 0) <= 0) continue;
      const childIds=childIdsFor(last,i,+(last.offspring_count?.[i] || 1));
      for(const childId of childIds){
        if(haloIndex++ % 4 !== 0) continue;
        const [x1,y1] = pointXY(last.x1[i], last.y1[i], trajBounds, w, h, pads);
        const [ox,oy] = visualOffset(childId);
        trajCtx.beginPath(); trajCtx.arc(x1+ox,y1+oy,TRAJ_HALO_RADIUS,0,Math.PI*2);
        trajCtx.strokeStyle = "rgba(143,122,230,.18)";
        trajCtx.lineWidth = 1.0;
        trajCtx.stroke();
      }
    }

    for(let i=0;i<tr.x0.length;i++){
      const event = tr.event[i] || "survive";
      const offspring = +(tr.offspring_count?.[i] ?? 1);
      const typ = tr.type?.[i] || "Type 4";
      const color = CELL_COLORS[typ] || "#9a89ec";
      const alpha = weightAlpha(lerp(+tr.w0[i],+tr.w1[i],u),weightMin,weightMax);
      const [sx0,sy0] = pointXY(tr.x0[i],tr.y0[i],trajBounds,w,h,pads);
      const [sx1,sy1] = pointXY(tr.x1[i],tr.y1[i],trajBounds,w,h,pads);
      const parentId=tr.id?.[i] || `${tr.source_step}:${i}`;
      const [pox,poy] = visualOffset(parentId);

      if(event === "die" || offspring <= 0){
        const fade = 1 - smoothstep(clamp((u-.12)/.88,0,1));
        const deathDrift = 0.22 * smoothstep(clamp(u/.42,0,1));
        const dx = lerp(sx0,sx1,deathDrift)+pox;
        const dy = lerp(sy0,sy1,deathDrift)+poy;
        drawDeathMarker(trajCtx,dx,dy,u,Math.max(.15,alpha));
        if(fade > .025) drawCell(trajCtx,dx,dy,TRAJ_CELL_RADIUS*(.45+.55*fade),"#ff5c52",alpha*fade*.96);
        continue;
      }

      const childIds=childIdsFor(tr,i,offspring);
      if(offspring === 1){
        const [cox,coy] = visualOffset(childIds[0]);
        drawCell(trajCtx,lerp(sx0+pox,sx1+cox,u),lerp(sy0+poy,sy1+coy,u),TRAJ_CELL_RADIUS,color,alpha);
        continue;
      }

      for(let j=0;j<offspring;j++){
        const [cox,coy] = visualOffset(childIds[j]);
        const pulse=Math.sin(Math.PI*u)*(3.2+Math.min(offspring,5));
        const angle=hashUnit(`${parentId}:${childIds[j]}:split`)*Math.PI*2;
        const x=lerp(sx0+pox,sx1+cox,u)+Math.cos(angle)*pulse;
        const y=lerp(sy0+poy,sy1+coy,u)+Math.sin(angle)*pulse;
        drawCell(trajCtx,x,y,offspring > 2 ? TRAJ_DIVIDE_SMALL_RADIUS : TRAJ_DIVIDE_RADIUS,color,alpha);
      }
    }
  }

  function drawTrajectory(time){
    const w = trajCanvas.width, h = trajCanvas.height;
    const pads = drawTrajectoryAxes(trajCtx,w,h,trajBounds);
    if(trajectory?.transitions?.length){
      drawBranchingTrajectory(time,w,h,pads);
    } else if(trajectory?.frames?.length){
      const f = interpolateFrame(trajectory.frames,time);
      if(stepLabel) stepLabel.textContent = `step = ${Math.round(f.stepValue ?? 0)}`;
      const n = trajectory.frames[0].x.length;
      const first = trajectory.frames[0];
      const last = trajectory.frames[trajectory.frames.length-1];

      // control ghost
      for(let i=0;i<n;i+=2){
        const [x0,y0] = pointXY(first.x[i], first.y[i], trajBounds, w, h, pads);
        drawCell(trajCtx,x0,y0,TRAJ_GHOST_RADIUS,"#b8c9d5",0.16);
      }

      // endpoint halo
      for(let i=0;i<n;i+=4){
        const [x1,y1] = pointXY(last.x[i], last.y[i], trajBounds, w, h, pads);
        trajCtx.beginPath(); trajCtx.arc(x1,y1,TRAJ_HALO_RADIUS,0,Math.PI*2);
        trajCtx.strokeStyle = "rgba(143,122,230,.18)";
        trajCtx.lineWidth = 1.0;
        trajCtx.stroke();
      }

      // trails
      trajCtx.save();
      trajCtx.lineWidth = 1.05;
      for(let i=0;i<n;i+=8){
        trajCtx.beginPath();
        for(let s=0;s<trajectory.frames.length;s++){
          const fr = trajectory.frames[s];
          const [x,y] = pointXY(fr.x[i], fr.y[i], trajBounds, w, h, pads);
          if(s===0) trajCtx.moveTo(x,y); else trajCtx.lineTo(x,y);
        }
        trajCtx.strokeStyle = i%16===0 ? "rgba(31,183,201,.16)" : "rgba(143,122,230,.12)";
        trajCtx.stroke();
      }
      trajCtx.restore();

      // current cells
      for(let i=0;i<f.x.length;i++){
        const [sx,sy] = pointXY(f.x[i],f.y[i],trajBounds,w,h,pads);
        const typ = f.type[i] || "Type 4";
        drawCell(trajCtx,sx,sy,TRAJ_CELL_RADIUS,CELL_COLORS[typ]||"#9a89ec",weightAlpha(f.weight[i],weightMin,weightMax));
      }
    }
    drawTypeLegendOverlay(trajCtx,w,h,{x:w-190,y:148,scale:1.34});
    requestAnimationFrame(drawTrajectory);
  }

  drawMap();
  drawColorbar();
  window.addEventListener("resize", () => {
    syncColorbarHeight();
    drawMap();
    drawColorbar();
  });
  loadSelected(selected);
  requestAnimationFrame(drawTrajectory);
}

function generateForwardData(){
  let seed=42; const rand=()=>{seed=(seed*1664525+1013904223)%4294967296; return seed/4294967296};
  const conditions=[{id:"condition_a",label:"Condition A",dx:1.35,dy:-.75},{id:"condition_b",label:"Condition B",dx:-1.12,dy:1.1},{id:"condition_c",label:"Condition C",dx:1.05,dy:1.05},{id:"condition_d",label:"Condition D",dx:-.55,dy:-1.25}];
  const types=["Type 1","Type 2","Type 3","Type 4"], base=[];
  for(let i=0;i<140;i++){const angle=rand()*Math.PI*2,r=.25+rand()*.85,x0=Math.cos(angle)*r+(rand()-.5)*.15,y0=Math.sin(angle)*r+(rand()-.5)*.15;base.push({cell_id:`cell_${i}`,cell_type:types[Math.floor(rand()*types.length)],x_control:x0,y_control:y0,seed:rand()})}
  const points=[]; for(const c of conditions){for(const b of base){const bendX=.26*Math.sin(b.y_control*2.8+c.dx*1.5),bendY=.26*Math.cos(b.x_control*2.8+c.dy*1.5);points.push({cell_id:b.cell_id,cell_type:b.cell_type,condition:c.id,x_control:b.x_control,y_control:b.y_control,x_target:b.x_control+c.dx+bendX+(b.seed-.5)*.16,y_target:b.y_control+c.dy+bendY+(b.seed-.5)*.16,mass:Math.max(.18,Math.min(2.2,1+.18*c.dy+(b.seed-.5)*.3))})}}
  return {conditions:conditions.map(({id,label})=>({id,label})),points};
}
function generateInverseData(){
  return {modes:["Cell-type design","High DEG","Low DEG"],candidates:[
    {mode:"Cell-type design",target:"Type 1",condition:"Design-α",score:.91,mass_fold_change:1.3,composition:{"Type 1":.82,"Type 2":.08,"Type 3":.06,"Type 4":.04},x_shift:-.95,y_shift:-.75},
    {mode:"Cell-type design",target:"Type 2",condition:"Design-β",score:.95,mass_fold_change:1.16,composition:{"Type 1":.07,"Type 2":.87,"Type 3":.03,"Type 4":.03},x_shift:1.15,y_shift:-.35},
    {mode:"Cell-type design",target:"Type 3",condition:"Design-γ",score:.92,mass_fold_change:.78,composition:{"Type 1":.06,"Type 2":.08,"Type 3":.81,"Type 4":.05},x_shift:-1.05,y_shift:1},
    {mode:"Cell-type design",target:"Type 4",condition:"Design-δ",score:.87,mass_fold_change:1.02,composition:{"Type 1":.12,"Type 2":.08,"Type 3":.1,"Type 4":.7},x_shift:.72,y_shift:1.08},
    {mode:"High DEG",target:"X1",condition:"Design-X1↑",score:.89,mass_fold_change:1.12,composition:{"Type 1":.68,"Type 2":.14,"Type 3":.08,"Type 4":.1},x_shift:-.85,y_shift:-.92},
    {mode:"High DEG",target:"X2",condition:"Design-X2↑",score:.93,mass_fold_change:.95,composition:{"Type 1":.1,"Type 2":.71,"Type 3":.07,"Type 4":.12},x_shift:1.05,y_shift:-.48},
    {mode:"High DEG",target:"X3",condition:"Design-X3↑",score:.84,mass_fold_change:.71,composition:{"Type 1":.14,"Type 2":.11,"Type 3":.57,"Type 4":.18},x_shift:1.18,y_shift:.88},
    {mode:"Low DEG",target:"X1",condition:"Design-X1↓",score:.88,mass_fold_change:.86,composition:{"Type 1":.13,"Type 2":.52,"Type 3":.14,"Type 4":.21},x_shift:1.06,y_shift:.62},
    {mode:"Low DEG",target:"X2",condition:"Design-X2↓",score:.86,mass_fold_change:1.2,composition:{"Type 1":.58,"Type 2":.1,"Type 3":.11,"Type 4":.21},x_shift:-1.1,y_shift:-.12},
    {mode:"Low DEG",target:"X3",condition:"Design-X3↓",score:.83,mass_fold_change:1.45,composition:{"Type 1":.36,"Type 2":.21,"Type 3":.09,"Type 4":.34},x_shift:-.22,y_shift:-1.18}
  ]};
}
function setupInverseDemo(forwardData,inverseData){
  const modeSelect=document.getElementById("modeSelect"),targetSelect=document.getElementById("targetSelect"),candidatePanel=document.getElementById("candidatePanel"),comp=document.getElementById("compositionBars"),canvas=document.getElementById("inverseCanvas"),ctx=canvas.getContext("2d"),basePoints=forwardData.points.filter(p=>p.condition===forwardData.conditions[0].id).slice(0,130);
  for(const m of inverseData.modes){const o=document.createElement("option");o.value=m;o.textContent=m;modeSelect.appendChild(o)}
  let selectedCandidate=null;
  function refreshTargets(){targetSelect.innerHTML="";const targets=[...new Set(inverseData.candidates.filter(c=>c.mode===modeSelect.value).map(c=>c.target))];targets.forEach(t=>{const o=document.createElement("option");o.value=t;o.textContent=t;targetSelect.appendChild(o)});refreshCandidates()}
  function refreshCandidates(){const cs=inverseData.candidates.filter(c=>c.mode===modeSelect.value&&c.target===targetSelect.value).sort((a,b)=>b.score-a.score);selectedCandidate=cs[0];candidatePanel.innerHTML="";cs.forEach((c,idx)=>{const el=document.createElement("div");el.className=`candidate ${idx===0?"active":""}`;el.innerHTML=`<strong>${c.condition}<em>${c.score.toFixed(2)}</em></strong><span>mass fold-change ${c.mass_fold_change.toFixed(2)}</span>`;el.onclick=()=>{selectedCandidate=c;[...candidatePanel.children].forEach(x=>x.classList.remove("active"));el.classList.add("active");updateComposition()};candidatePanel.appendChild(el)});updateComposition()}
  function updateComposition(){comp.innerHTML="";if(!selectedCandidate)return;Object.entries(selectedCandidate.composition).forEach(([type,value])=>{const row=document.createElement("div");row.className="bar-row";row.innerHTML=`<label><span>${type}</span><span>${Math.round(value*100)}%</span></label><div class="bar-track"><div class="bar-fill" style="--bar:${CELL_COLORS[type]};width:${value*100}%"></div></div>`;comp.appendChild(row)})}
  modeSelect.onchange=refreshTargets; targetSelect.onchange=refreshCandidates; refreshTargets();
  function frame(time){
    const w=canvas.width,h=canvas.height; if(!selectedCandidate) return requestAnimationFrame(frame);
    const loop=(Math.sin(time/1200)+1)/2,t=smoothstep(loop),shifted=basePoints.map((p,i)=>({...p,x_target:p.x_control+selectedCandidate.x_shift+.24*Math.sin(p.y_control*2.7+i*.03),y_target:p.y_control+selectedCandidate.y_shift+.24*Math.cos(p.x_control*2.7+i*.03)}));
    const b={minX:Math.min(...shifted.flatMap(p=>[p.x_control,p.x_target])),maxX:Math.max(...shifted.flatMap(p=>[p.x_control,p.x_target])),minY:Math.min(...shifted.flatMap(p=>[p.y_control,p.y_target])),maxY:Math.max(...shifted.flatMap(p=>[p.y_control,p.y_target]))};
    drawGrid(ctx,w,h,{x:"latent 1",y:"latent 2"});
    shifted.forEach(p=>{const[sx,sy]=scaleXY(p.x_control,p.y_control,b,w,h);drawCell(ctx,sx,sy,4.5,"#c6d0dc",.42)});
    shifted.forEach((p,i)=>{const x=lerp(p.x_control,p.x_target,t)+.04*Math.sin(time/620+i),y=lerp(p.y_control,p.y_target,t)+.04*Math.cos(time/750+i),[sx,sy]=scaleXY(x,y,b,w,h);drawCell(ctx,sx,sy,6.5,CELL_COLORS[p.cell_type]||"#9cc",.82)});
    ctx.fillStyle="rgba(16,32,51,.72)";ctx.font="800 16px Inter, system-ui, sans-serif";ctx.fillText(selectedCandidate.condition,28,64);ctx.fillStyle="rgba(82,100,120,.75)";ctx.font="700 13px Inter, system-ui, sans-serif";ctx.fillText(`${selectedCandidate.mode} · ${selectedCandidate.target}`,28,84);
    requestAnimationFrame(frame)
  }
  requestAnimationFrame(frame)
}

function setupDemo2(payload){
  const tasks = Array.isArray(payload?.tasks) ? payload.tasks : [];
  if(!tasks.length) return;

  const taskTypeButtons=document.getElementById("demo2TaskTypeButtons");
  const targetButtons=document.getElementById("demo2TargetButtons");
  const info=document.getElementById("demo2ConditionInfo");
  const canvas=document.getElementById("demo2Canvas");
  const stepLabel=document.getElementById("demo2StepLabel");
  const comp=document.getElementById("demo2CompositionBars");
  const means=document.getElementById("demo2MeanCards");
  const targetHint=document.getElementById("demo2TargetHint");
  if(!taskTypeButtons || !targetButtons || !info || !canvas || !comp || !means) return;

  const ctx=canvas.getContext("2d");
  const taskTypes=[
    {value:"cell_type",label:"Retain cell type"},
    {value:"deg",label:"DEG target"}
  ];
  const targetOrders={
    cell_type:["type2","type3","type4"],
    deg:["high_high","high_low","low_high","low_low"]
  };
  const targetLabels={
    type2:"Type 2",
    type3:"Type 3",
    type4:"Type 4",
    high_high:"X1 high · X2 high",
    high_low:"X1 high · X2 low",
    low_high:"X1 low · X2 high",
    low_low:"X1 low · X2 low"
  };

  let selectedTask=null;
  let trajectory=null;
  let bounds=payload.trajectory_bounds || null;
  let weightMin=payload.weight_bounds?.min ?? 0;
  let weightMax=payload.weight_bounds?.max ?? 1;
  let selectedTaskType="cell_type";

  taskTypeButtons.innerHTML="";
  for(const t of taskTypes){
    const btn=document.createElement("button");
    btn.type="button";
    btn.className="demo2-target-button demo2-task-type-button";
    btn.dataset.taskType=t.value;
    btn.textContent=t.label;
    btn.addEventListener("click",()=>{
      selectedTaskType=t.value;
      updateTaskTypeButtons();
      refreshTargets();
    });
    taskTypeButtons.appendChild(btn);
  }

  function taskSort(a,b){
    const order=targetOrders[a.task_type] || [];
    return order.indexOf(a.target_key)-order.indexOf(b.target_key);
  }

  function tasksForType(type){
    return tasks.filter(t=>t.task_type===type).sort(taskSort);
  }

  function refreshTargets(){
    const typed=tasksForType(selectedTaskType);
    targetButtons.innerHTML="";
    for(const t of typed){
      const btn=document.createElement("button");
      btn.type="button";
      btn.className="demo2-target-button";
      btn.dataset.targetKey=t.target_key;
      btn.textContent=targetLabels[t.target_key] || t.target_label;
      btn.addEventListener("click",()=>selectTask(t,true));
      targetButtons.appendChild(btn);
    }
    if(typed.length) selectTask(typed[0]);
  }

  function updateInfo(){
    info.innerHTML = `
      <div class="compact-condition-line">
        <span>&rho;<sub>1</sub> <em>${formatNum(selectedTask.rho1,3)}</em></span>
        <span>&rho;<sub>2</sub> <em>${formatNum(selectedTask.rho2,3)}</em></span>
        <span>&alpha; <em>${formatNum(selectedTask.alpha_g,3)}</em></span>
      </div>`;
  }

  function updateResults(){
    const rows=[
      ["Type 1",+selectedTask.type1_prop || 0],
      ["Type 2",+selectedTask.type2_prop || 0],
      ["Type 3",+selectedTask.type3_prop || 0],
      ["Type 4",+selectedTask.type4_prop || 0]
    ];
    comp.innerHTML="";
    for(const [type,value] of rows){
      const row=document.createElement("div");
      row.className=`bar-row ${selectedTask.target_type===type ? "target-row" : ""}`;
      row.innerHTML=`<label><span>${type}</span><span>${Math.round(value*100)}%</span></label><div class="bar-track"><div class="bar-fill" style="--bar:${CELL_COLORS[type]};width:${value*100}%"></div></div>`;
      comp.appendChild(row);
    }
    const x1Delta=(+selectedTask.final_mean_X1 || 0)-(+selectedTask.initial_mean_X1 || 0);
    const x2Delta=(+selectedTask.final_mean_X2 || 0)-(+selectedTask.initial_mean_X2 || 0);
    means.innerHTML=`
      <div class="demo2-mean-card"><span>X1 shift</span><b>${signedNum(x1Delta,2)}</b></div>
      <div class="demo2-mean-card"><span>X2 shift</span><b>${signedNum(x2Delta,2)}</b></div>`;
  }
  function updateTargetButtons(){
    for(const btn of targetButtons.querySelectorAll(".demo2-target-button")){
      btn.classList.toggle("active", btn.dataset.targetKey === selectedTask?.target_key);
    }
  }

  function updateTaskTypeButtons(){
    for(const btn of taskTypeButtons.querySelectorAll(".demo2-task-type-button")){
      btn.classList.toggle("active", btn.dataset.taskType === selectedTaskType);
    }
  }

  function drawDemo2Axes(ctx,w,h,b){
    ctx.clearRect(0,0,w,h);
    ctx.fillStyle="#fbfffe";
    ctx.fillRect(0,0,w,h);
    const base={left:82,right:38,top:40,bottom:72};
    const availW=w-base.left-base.right;
    const availH=h-base.top-base.bottom;
    const size=Math.min(availW,availH);
    const padL=base.left+(availW-size)/2;
    const padT=base.top+(availH-size)/2;
    const plotW=size, plotH=size;
    const glow=ctx.createRadialGradient(w*.50,h*.46,18,w*.50,h*.46,w*.72);
    glow.addColorStop(0,"rgba(31,183,201,.07)");
    glow.addColorStop(.55,"rgba(107,211,155,.04)");
    glow.addColorStop(1,"rgba(255,255,255,0)");
    ctx.fillStyle=glow;
    ctx.fillRect(0,0,w,h);

    ctx.strokeStyle="rgba(31,64,87,.075)";
    ctx.lineWidth=1;
    const ticks=5;
    for(let i=0;i<=ticks;i++){
      const x=padL+i/ticks*plotW;
      const y=padT+i/ticks*plotH;
      ctx.beginPath(); ctx.moveTo(x,padT); ctx.lineTo(x,padT+plotH); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(padL,y); ctx.lineTo(padL+plotW,y); ctx.stroke();
    }
    ctx.strokeStyle="rgba(16,32,51,.34)";
    ctx.lineWidth=1.25;
    ctx.beginPath(); ctx.moveTo(padL,padT); ctx.lineTo(padL,padT+plotH); ctx.lineTo(padL+plotW,padT+plotH); ctx.stroke();
    ctx.fillStyle="#102033";
    ctx.font="950 22px Inter, system-ui, sans-serif";
    ctx.textAlign="center";
    ctx.textBaseline="alphabetic";
    ctx.fillText("X1",padL+plotW/2,h-20);
    ctx.save();
    ctx.translate(30,padT+plotH/2);
    ctx.rotate(-Math.PI/2);
    ctx.fillText("X2",0,0);
    ctx.restore();
    return {padL,padT,plotW,plotH};
  }

  function pointXY(x,y,b,w,h,p){
    const sx=p.padL+(x-b.minX)/(b.maxX-b.minX || 1)*p.plotW;
    const sy=p.padT+(1-(y-b.minY)/(b.maxY-b.minY || 1))*p.plotH;
    return [sx,sy];
  }

  function drawDemo2Trajectory(time){
    const w=canvas.width,h=canvas.height;
    if(!trajectory?.transitions?.length){
      ctx.clearRect(0,0,w,h);
      ctx.fillStyle="#fbfffe";
      ctx.fillRect(0,0,w,h);
      ctx.fillStyle="#607188";
      ctx.font="900 18px Inter, system-ui, sans-serif";
      ctx.fillText("Loading trajectory",32,48);
      requestAnimationFrame(drawDemo2Trajectory);
      return;
    }
    const activeBounds=bounds || getEventTrajectoryBounds(trajectory.transitions);
    const pads=drawDemo2Axes(ctx,w,h,activeBounds);

    const tf=transitionFrame(trajectory.transitions,time);
    const {tr,u,stepValue}=tf;
    if(stepLabel) stepLabel.textContent=`step = ${Math.round(stepValue ?? 0)}`;

    const first=trajectory.transitions[0];
    for(let i=0;i<first.x0.length;i+=2){
      const [x0,y0]=pointXY(first.x0[i],first.y0[i],activeBounds,w,h,pads);
      drawCell(ctx,x0,y0,TRAJ_GHOST_RADIUS,"#b8c9d5",0.15);
    }

    const last=trajectory.transitions[trajectory.transitions.length-1];
    let haloIndex=0;
    for(let i=0;i<last.x1.length;i+=4){
      if((last.offspring_count?.[i] || 0)<=0) continue;
      const childIds=childIdsFor(last,i,+(last.offspring_count?.[i] || 1));
      for(const childId of childIds){
        if(haloIndex++ % 4 !== 0) continue;
        const [x1,y1]=pointXY(last.x1[i],last.y1[i],activeBounds,w,h,pads);
        const [ox,oy]=visualOffset(childId);
        ctx.beginPath();
        ctx.arc(x1+ox,y1+oy,TRAJ_HALO_RADIUS,0,Math.PI*2);
        ctx.strokeStyle="rgba(143,122,230,.18)";
        ctx.lineWidth=1.0;
        ctx.stroke();
      }
    }

    for(let i=0;i<tr.x0.length;i++){
      const event=tr.event[i] || "survive";
      const offspring=+(tr.offspring_count?.[i] ?? 1);
      const typ=tr.type?.[i] || "Type 4";
      const color=CELL_COLORS[typ] || "#9a89ec";
      const alpha=weightAlpha(lerp(+tr.w0[i],+tr.w1[i],u),weightMin,weightMax);
      const [sx0,sy0]=pointXY(tr.x0[i],tr.y0[i],activeBounds,w,h,pads);
      const [sx1,sy1]=pointXY(tr.x1[i],tr.y1[i],activeBounds,w,h,pads);
      const parentId=tr.id?.[i] || `${tr.source_step}:${i}`;
      const [pox,poy]=visualOffset(parentId);

      if(event==="die" || offspring<=0){
        const fade=1-smoothstep(clamp((u-.12)/.88,0,1));
        const deathDrift=.22*smoothstep(clamp(u/.42,0,1));
        const dx=lerp(sx0,sx1,deathDrift)+pox;
        const dy=lerp(sy0,sy1,deathDrift)+poy;
        drawDeathMarker(ctx,dx,dy,u,Math.max(.15,alpha));
        if(fade>.025) drawCell(ctx,dx,dy,TRAJ_CELL_RADIUS*(.45+.55*fade),"#ff5c52",alpha*fade*.96);
        continue;
      }

      const childIds=childIdsFor(tr,i,offspring);
      if(offspring===1){
        const [cox,coy]=visualOffset(childIds[0]);
        drawCell(ctx,lerp(sx0+pox,sx1+cox,u),lerp(sy0+poy,sy1+coy,u),TRAJ_CELL_RADIUS,color,alpha);
        continue;
      }
      for(let j=0;j<offspring;j++){
        const [cox,coy]=visualOffset(childIds[j]);
        const pulse=Math.sin(Math.PI*u)*(3.2+Math.min(offspring,5));
        const angle=hashUnit(`${parentId}:${childIds[j]}:split`)*Math.PI*2;
        const x=lerp(sx0+pox,sx1+cox,u)+Math.cos(angle)*pulse;
        const y=lerp(sy0+poy,sy1+coy,u)+Math.sin(angle)*pulse;
        drawCell(ctx,x,y,offspring>2 ? TRAJ_DIVIDE_SMALL_RADIUS : TRAJ_DIVIDE_RADIUS,color,alpha);
      }
    }
    drawTypeLegendOverlay(ctx,w,h,{x:w-190,y:148,scale:1.34});
    requestAnimationFrame(drawDemo2Trajectory);
  }

  async function selectTask(task,userDriven=false){
    selectedTask=task;
    selectedTaskType=task.task_type;
    if(userDriven) targetHint?.classList.add("is-dismissed");
    updateTaskTypeButtons();
    updateTargetButtons();
    updateInfo();
    updateResults();
    trajectory=await loadJson(task.trajectory || `data/demo2/condition_${pad3(task.condition)}.json`);
    bounds=payload.trajectory_bounds || getEventTrajectoryBounds(trajectory.transitions);
    const weights=trajectory.transitions.flatMap(tr=>[...(tr.w0 || []),...(tr.w1 || [])]);
    weightMin=Number.isFinite(+payload.weight_bounds?.min) ? +payload.weight_bounds.min : Math.min(...weights);
    weightMax=Number.isFinite(+payload.weight_bounds?.max) ? +payload.weight_bounds.max : Math.max(...weights);
  }

  updateTaskTypeButtons();
  refreshTargets();
  requestAnimationFrame(drawDemo2Trajectory);
}
async function main(){
  try {
    setupTopbar();
    const conditionPayload=await loadJsonWithFallback("data/demo1_condition_index.json",{conditions:fallbackConditionIndex()});
    const demo1Payload=normalizeConditionPayload(conditionPayload);
    console.log("Demo1 conditions", demo1Payload.conditions?.length, demo1Payload);
    setupConditionAtlasDemo(demo1Payload.conditions, demo1Payload);
    const demo2Payload=await loadJsonWithFallback("data/demo2_task_index.json",{tasks:[]});
    setupDemo2(demo2Payload);
  } catch (err) {
    console.error("U-Pert website initialization failed:", err);
    showCanvasError('conditionMapCanvas', err.message || err);
    showCanvasError('forwardCanvas', err.message || err);
    showCanvasError('demo2Canvas', err.message || err);
  }
}
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', main);
} else {
  main();
}
