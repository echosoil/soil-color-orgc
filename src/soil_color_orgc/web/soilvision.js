"use strict";
const el = id => document.getElementById(id);
const coords = ["x0","y0","x1","y1"].map(el);
const grayCoords = ["gx0","gy0","gx1","gy1"].map(el);
let grayInfo=null;
const defaultROI = [0.4,0.5,0.6,0.7];
const mediaTypes = ["image/jpeg","image/png","image/webp"];
let photo = null, previewURL = null, generation = 0, ready = false, busy = false, result = null, drag = null;
function say(text) { el("message").textContent = text; }
function roi(inputs=coords) {
  const values = inputs.map(input => input.valueAsNumber);
  const [x0,y0,x1,y1] = values;
  return values.every(Number.isFinite) && x0>=0 && y0>=0 && x1<=1 && y1<=1 && x0<x1 && y0<y1 ? values : null;
}
function drawBox(box, r) {
  if (!r) { box.hidden=true; return; }
  const [x0,y0,x1,y1]=r;
  box.hidden=false;
  box.style.left=(100*x0)+"%"; box.style.top=(100*y0)+"%";
  box.style.width=(100*(x1-x0))+"%"; box.style.height=(100*(y1-y0))+"%";
}
function sync() {
  drawBox(el("roi-box"),roi());
  el("analyze").disabled=!ready||busy||!roi()||(el("gray").checked&&el("manual-gray").checked&&!roi(grayCoords));
  el("detect-gray").disabled=!ready||busy;
  el("check-gray").disabled=!ready||busy||!el("manual-gray").checked||!roi(grayCoords);
  grayCoords.forEach(i=>i.disabled=busy||!el("manual-gray").checked);
}
function normalized(rect,w,h){return [rect[0]/w,rect[1]/h,rect[2]/w,rect[3]/h];}
function showGray(info,w,h){
  grayInfo=info;el("gray-patches").replaceChildren();
  if(!info){drawBox(el("gray-box"),null);return;}
  const region=normalized(info.rect,w,h);
  drawBox(el("gray-box"),region);
  if(!el("manual-gray").checked)grayCoords.forEach((input,i)=>input.value=Number(region[i].toFixed(6)));
  for(const rect of info.patch_rects||[]){
    const box=document.createElement("div");box.className="gray-patch";el("gray-patches").append(box);
    drawBox(box,normalized(rect,w,h));
  }
  const d=info.ramp_diagnostics||{};
  el("gray-info").textContent=info.source.replaceAll("_"," ")+": "+(info.usable?"ramp accepted for correction":"not accepted; inspect or select manually")+(Number.isFinite(d.corr)?" · correlation "+d.corr.toFixed(3)+" · contrast "+d.contrast.toFixed(1):"");
}
function editGray(){
  grayInfo=null;el("gray-patches").replaceChildren();drawBox(el("gray-box"),roi(grayCoords));
  el("gray-info").textContent="Manual strip not checked. Click Check manual strip or analyze with grey correction enabled.";
  invalidate();sync();
}
async function inspectGray(manual){
  if(!ready||busy)return;
  const region=manual?roi(grayCoords):null;if(manual&&!region)return;
  const id=generation;
  if(!manual)el("manual-gray").checked=false;
  busy=true;disableControls(true);invalidate();sync();say("Inspecting grey reference…");
  try{
    const suffix=manual?"?"+new URLSearchParams({gray_roi:region.join(",")}):"";
    const response=await post("/api/v1/gray-preview"+suffix,photo);
    const r=await response.json();if(id!==generation)return;
    showGray(r.gray_region,r.image_width,r.image_height);
    say("Grey region inspection finished. Review the overlay before analysis.");
  }catch(error){if(id===generation){showGray(null);el("gray-info").textContent="Grey inspection failed.";say(error.message);}}
  finally{if(id===generation){busy=false;disableControls(false);sync();}}
}
function invalidate() { result=null; el("results").hidden=true; el("placeholder").hidden=false; el("used-box").hidden=true; }
function setROI(r) { coords.forEach((input,i)=>input.value=Number(r[i].toFixed(3))); invalidate(); sync(); }
function revokePreview() { if(previewURL) URL.revokeObjectURL(previewURL); previewURL=null; el("preview").removeAttribute("src"); }
async function post(path, file) {
  const response=await fetch(path,{method:"POST",headers:{"Content-Type":file.type},body:file,cache:"no-store"});
  if(!response.ok) {
    let detail="";
    try { const error=await response.json(); if(typeof error.detail==="string") detail=error.detail; } catch {}
    throw new Error(response.status===503 ? "Service busy. Please try again." : "Request rejected ("+response.status+(detail?": "+detail.replaceAll("_"," "):"")+")");
  }
  return response;
}
el("photo").addEventListener("change",async()=>{
  const id=++generation; photo=null; ready=false; busy=false; drag=null; invalidate(); revokePreview(); showGray(null); el("manual-gray").checked=false; grayCoords.forEach(i=>i.value=""); el("gray-info").textContent="Use Detect automatically to inspect the suggested strip.";
  el("preview-area").hidden=true; sync();
  const file=el("photo").files[0];
  if(!file){say("Choose a photograph to begin.");return;}
  if(!mediaTypes.includes(file.type)||!file.size||file.size>8*1024*1024){say("Choose a JPEG, PNG or WebP image up to 8 MiB.");return;}
  photo=file; say("Preparing oriented preview…");
  try {
    const response=await post("/api/v1/preview",file);
    const blob=await response.blob();
    if(id!==generation)return;
    const url=URL.createObjectURL(blob); previewURL=url;
    el("preview").src=url;
    await el("preview").decode();
    if(id!==generation)return;
    el("preview-area").hidden=false; ready=true; setROI(defaultROI); say("Choose a region and analyze.");
  } catch(error){ if(id===generation){ready=false;revokePreview();say(error.message);sync();} }
});
coords.forEach(input=>input.addEventListener("input",()=>{invalidate();sync();}));
el("gray").addEventListener("change",()=>{invalidate();sync();});
el("manual-gray").addEventListener("change",()=>{
  if(el("manual-gray").checked)editGray();else{showGray(null);el("gray-info").textContent="Click Detect automatically to inspect the strip.";invalidate();sync();}
});
grayCoords.forEach(input=>input.addEventListener("input",editGray));
el("detect-gray").addEventListener("click",()=>inspectGray(false));
el("check-gray").addEventListener("click",()=>inspectGray(true));
el("reset").addEventListener("click",()=>setROI(defaultROI));
function point(event){const rect=el("stage").getBoundingClientRect();return [Math.max(0,Math.min(1,(event.clientX-rect.left)/rect.width)),Math.max(0,Math.min(1,(event.clientY-rect.top)/rect.height))];}
el("stage").addEventListener("pointerdown",event=>{
  if(!ready||busy||!event.isPrimary||event.button!==0)return;
  drag={start:point(event),id:event.pointerId,gray:el("manual-gray").checked}; el("stage").setPointerCapture(event.pointerId);
});
function updateDrag(event){
  if(!drag||drag.id!==event.pointerId)return;
  const [x,y]=point(event),[a,b]=drag.start;
  if(Math.abs(x-a)<.005||Math.abs(y-b)<.005)return;
  const r=[Math.min(x,a),Math.min(y,b),Math.max(x,a),Math.max(y,b)];
  if(drag.gray){grayCoords.forEach((input,i)=>input.value=Number(r[i].toFixed(4)));editGray();}else setROI(r);
}
el("stage").addEventListener("pointermove",updateDrag);
el("stage").addEventListener("pointerup",event=>{updateDrag(event);drag=null;});
el("stage").addEventListener("pointercancel",()=>{drag=null;});
function disableControls(disabled){coords.forEach(input=>input.disabled=disabled);el("gray").disabled=disabled;el("reset").disabled=disabled;el("manual-gray").disabled=disabled;grayCoords.forEach(input=>input.disabled=disabled||!el("manual-gray").checked);}
function swatch(id,value){const canvas=el(id),ctx=canvas.getContext("2d");ctx.clearRect(0,0,canvas.width,canvas.height);if(value&&/^#[0-9a-f]{6}$/i.test(value.hex)){ctx.fillStyle=value.hex;ctx.fillRect(0,0,canvas.width,canvas.height);}}
function metric(name,value){const row=document.createElement("div"),dt=document.createElement("dt"),dd=document.createElement("dd");dt.textContent=name;dd.textContent=value;row.append(dt,dd);el("metrics").append(row);}
function render(r){
  result=r; el("placeholder").hidden=true;el("results").hidden=false;
  el("status").textContent=({ok:"Analysis completed",review:"Review recommended",insufficient_soil_pixels:"Insufficient soil pixels"})[r.status]||r.status;
  swatch("extracted",r.swatches?.extracted); swatch("matched",r.swatches?.munsell);
  el("lab").textContent=r.lab?"L* "+r.lab[0].toFixed(2)+" · a* "+r.lab[1].toFixed(2)+" · b* "+r.lab[2].toFixed(2):"Unavailable";
  el("munsell").textContent=r.munsell||"No accepted match";
  el("swatch-note").textContent="Swatches are approximate sRGB display colours, not physical standards."+((r.swatches?.extracted?.gamut_clipped||r.swatches?.munsell?.gamut_clipped)?" One or more colours were clipped to the display gamut.":"");
  el("metrics").replaceChildren();
  metric("CIEDE2000 distance",r.delta_e_2000==null?"Unavailable":r.delta_e_2000.toFixed(2));
  metric("Selected pixels",r.selected_pixels+" / "+r.roi_pixel_count);
  metric("Selected fraction",(100*r.selected_fraction).toFixed(2)+"%");
  metric("Grey correction",r.gray_calibration.replaceAll("_"," "));
  if(r.gray_metrics){metric("Mean grey imbalance before",r.gray_metrics.mean_imbalance_before.toFixed(2));metric("Mean grey imbalance after",r.gray_metrics.mean_imbalance_after.toFixed(2));}
  if(r.gray_region){showGray(r.gray_region,r.image_width,r.image_height);metric("Grey region source",r.gray_region.source.replaceAll("_"," "));}
  el("gray-details").hidden=!r.gray_region?.patch_measurements;el("gray-report").textContent=(r.gray_region?.patch_measurements||[]).map(p=>p.scale_label+": "+[p.measured_R,p.measured_G,p.measured_B].map(v=>v.toFixed(1)).join(" / ")+" → "+[p.after_R,p.after_G,p.after_B].map(v=>v.toFixed(1)).join(" / ")+" (target "+p.target_gray.toFixed(1)+")").join("\n");
  metric("ICC handling",r.image_decoding.icc_status.replaceAll("_"," "));
  metric("Carbon prediction","Disabled");
  el("warnings").replaceChildren();
  const labels={gray_calibration_failed:"Grey correction failed; uncorrected image used.",gray_region_not_accepted:"Grey region was not accepted; check all 11 patches and select manually if appropriate.",gray_imbalance_not_improved:"Grey imbalance did not improve.",insufficient_soil_pixels:"Insufficient selected soil pixels.",high_lightness_check_roi:"High lightness; check the selected region.",no_close_munsell_match:"No reference colour within the matching threshold."};
  (r.warnings.length?r.warnings:["none"]).forEach(w=>{const li=document.createElement("li");li.textContent=w==="none"?"No algorithm warnings. This does not establish measurement validity.":labels[w]||w;el("warnings").append(li);});
  el("provenance").textContent=JSON.stringify({method:r.method_version,reference_sha256:r.reference_sha256,requested_roi:r.requested_roi,gray_requested:r.gray_requested,gray_region:r.gray_region,image_decoding:r.image_decoding,scientific_status:r.scientific_status},null,2);
  const [x0,y0,x1,y1]=r.roi_pixels;
  if(r.image_width===el("preview").naturalWidth&&r.image_height===el("preview").naturalHeight)drawBox(el("used-box"),[x0/r.image_width,y0/r.image_height,x1/r.image_width,y1/r.image_height]);
}
el("analyze").addEventListener("click",async()=>{
  const region=roi();if(!ready||busy||!region)return;
  const id=generation,file=photo,gray=el("gray").checked;
  busy=true;disableControls(true);invalidate();sync();say("Analyzing photograph…");
  try{
    const params=new URLSearchParams({roi:region.join(","),gray:String(gray)});
    if(gray&&el("manual-gray").checked)params.set("gray_roi",roi(grayCoords).join(","));
    const response=await post("/api/v1/analyze?"+params,file);
    const r=await response.json();
    if(id!==generation)return;
    r.requested_roi=region;r.gray_requested=gray;render(r);say("Analysis finished.");
  }catch(error){if(id===generation)say(error.message);}
  finally{if(id===generation){busy=false;disableControls(false);sync();}}
});
el("photo").addEventListener("change",()=>disableControls(false));
function download(text,type,name){const url=URL.createObjectURL(new Blob([text],{type}));const a=document.createElement("a");a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
el("json").addEventListener("click",()=>{if(result)download(JSON.stringify(result,null,2)+"\n","application/json","soilvision-result.json");});
function csvCell(value){let s=value==null?"":typeof value==="object"?JSON.stringify(value):String(value);if(typeof value==="string"&&/^[=+@-]/.test(s))s="'"+s;return '"'+s.replaceAll('"','""')+'"';}
el("csv").addEventListener("click",()=>{
  if(!result)return;
  const r={...result,L:rval(0),a:rval(1),b:rval(2)};
  function rval(i){return result.lab?.[i]??null;}
  download(Object.keys(r).map(csvCell).join(",")+"\r\n"+Object.values(r).map(csvCell).join(",")+"\r\n","text/csv;charset=utf-8","soilvision-result.csv");
});
window.addEventListener("pagehide",revokePreview);
