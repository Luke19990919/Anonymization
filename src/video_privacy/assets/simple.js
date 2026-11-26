'use strict';
// Pure estimator: recent two-minute throughput, no countdown while progress stalls.
function estimateProgress(samples,total,now){
 if(samples.length<2)return {seconds:null,rate:0,stalled:false};
 const first=samples[0],last=samples[samples.length-1];
 let changed=first;
 for(let i=1;i<samples.length;i++)if(samples[i].count>samples[i-1].count)changed=samples[i];
 const stalled=now-changed.time>60000;
 const seconds=(last.time-first.time)/1000,frames=last.count-first.count;
 const rate=seconds>0?frames/seconds:0;
 return {seconds:!stalled&&seconds>=15&&frames>=3&&total>last.count?(total-last.count)/rate:null,rate,stalled};
}

// A source HTML opened from disk has no processing backend. Keep the startup guide visible.
if(location.protocol==='http:' || location.protocol==='https:'){
const $=id=>document.getElementById(id);
const token=document.querySelector('meta[name="local-token"]').content;
let selected=null,busy=false,pollTimer=null,rateStart=null,lastAdvance=null,progressSamples=[];
function showError(message){busy=false;$('error').textContent=message;$('error').hidden=false;$('uploadView').hidden=false;$('progressView').hidden=true;$('doneView').hidden=true;$('start').disabled=!selected;$('dropzone').setAttribute('aria-disabled','false');}
function select(file){
 if(busy||!file)return;
 const ext=file.name.split('.').pop().toLowerCase();
 if(!['mp4','mov','m4v','webm','mkv','avi'].includes(ext)){showError('請選擇影片檔，例如 MP4 或 MOV。');return;}
 if(!file.size||file.size>2*1024*1024*1024){showError('請選擇不超過 2 GB 的影片。');return;}
 selected=file;$('error').hidden=true;$('fileTitle').textContent=file.name;
 $('fileHint').textContent=(file.size/1024/1024).toFixed(1)+' MB · 點一下可更換影片';$('start').disabled=false;
}
$('videoFile').addEventListener('change',()=>select($('videoFile').files[0]));
$('dropzone').addEventListener('keydown',event=>{if(['Enter',' '].includes(event.key)){event.preventDefault();if(!busy)$('videoFile').click();}});
$('dropzone').addEventListener('click',event=>{if(busy)event.preventDefault();});
document.addEventListener('dragover',event=>{event.preventDefault();if(!busy)$('dropzone').classList.add('is-dragging');});
document.addEventListener('dragleave',event=>{if(!event.relatedTarget)$('dropzone').classList.remove('is-dragging');});
document.addEventListener('drop',event=>{event.preventDefault();$('dropzone').classList.remove('is-dragging');if(event.dataTransfer.files.length>1){showError('一次請選擇一支影片。');return;}select(event.dataTransfer.files[0]);});
function phase(title,detail,percent,label){$('phaseTitle').textContent=title;$('phaseDetail').textContent=detail;$('progressText').textContent=label||'';if(percent===null)$('progress').removeAttribute('value');else $('progress').value=percent;}
async function poll(){
 try{
  const response=await fetch('/api/state',{credentials:'same-origin',cache:'no-store'});
  if(!response.ok)throw Error('連線失效，請重新整理此頁後再試。');
  const state=await response.json();
  if(state.phase==='error'){showError(state.error||'處理未完成，請再試一次。');return;}
  if(state.phase==='done'){
   busy=false;$('progressView').hidden=true;$('doneView').hidden=false;
   $('resultVideo').src='/api/video?result='+Date.now();$('resultVideo').load();
   return;
  }
  if(state.phase==='preparing')phase('第一次使用，正在準備所需檔案','約需下載 3 GB；你的影片不會傳到外部服務。',null,'準備完成後會自動開始處理。');
  else if(state.phase==='processing'){
   const count=state.frames||0,total=state.total_frames||0;
   const now=performance.now();
   if(!rateStart||count<rateStart.count){rateStart={count,time:now};lastAdvance={count,time:now};progressSamples=[];}
   if(count>lastAdvance.count)lastAdvance={count,time:now};
   progressSamples.push({count,time:now});
   while(progressSamples.length>2&&progressSamples[1].time<now-120000)progressSamples.shift();
   const estimate=estimateProgress(progressSamples,total,now);
   const elapsed=state.elapsed_seconds;
   let detail=Number.isFinite(elapsed)?`已用 ${Math.floor(elapsed/60)} 分 ${Math.floor(elapsed%60)} 秒。`:'正在依本次觀察估算速度。';
   if(estimate.stalled)detail+=' 超過一分鐘無新增完成畫面，暫停時間預估；正在等待辨識結果。';
   else if(estimate.seconds!==null){
    detail+=` 約 ${estimate.rate.toFixed(2)} 幀／秒，預估剩餘 ${Math.max(1,Math.floor(estimate.seconds*.75/60))}–${Math.max(1,Math.ceil(estimate.seconds*1.5/60))} 分鐘。`;
   }else detail+=' 正在收集足夠進度，暫不提供剩餘時間。';
   detail+=' 依近期速度更新，非保證完成時間；辨識、遮蔽與編碼同步進行。';
   phase('正在逐幀辨識、遮蔽與編碼',detail,total?Math.min(99,count/total*100):null,count?`已處理 ${count}${total?' / '+total:''} 個畫面`:'正在載入辨識模型與緩衝畫面，請稍候。');
  }else phase('正在接收影片','影片只會送入這台電腦上的處理程式。',null,'請保持視窗開啟。');
  pollTimer=setTimeout(poll,1200);
 }catch(error){showError(error.message||'無法連線，請確認本機程式仍在執行。');}
}
$('start').addEventListener('click',()=>{
 if(!selected||busy)return;rateStart=null;lastAdvance=null;busy=true;$('error').hidden=true;$('uploadView').hidden=true;$('progressView').hidden=false;$('start').disabled=true;$('dropzone').setAttribute('aria-disabled','true');
 phase('正在接收影片','影片只會送入這台電腦上的處理程式。',0,'');
 const request=new XMLHttpRequest();request.open('POST','/api/upload');
 request.setRequestHeader('X-Local-Token',token);request.setRequestHeader('X-Video-Extension','.'+selected.name.split('.').pop().toLowerCase());
 request.upload.onprogress=event=>{if(event.lengthComputable)phase('正在接收影片','影片只會送入這台電腦上的處理程式。',event.loaded/event.total*100,Math.round(event.loaded/event.total*100)+'%');};
 request.onerror=()=>showError('無法連線，請確認本機程式仍在執行。');
 request.onload=()=>{if(request.status===202)poll();else{let message='影片未送出，請再試一次。';try{message=JSON.parse(request.responseText).error||message;}catch(error){}showError(message);}};
 request.send(selected);
});
$('again').addEventListener('click',()=>{if(pollTimer)clearTimeout(pollTimer);selected=null;busy=false;$('videoFile').value='';$('resultVideo').pause();$('resultVideo').removeAttribute('src');$('resultVideo').load();$('doneView').hidden=true;$('progressView').hidden=true;$('uploadView').hidden=false;$('error').hidden=true;$('fileTitle').textContent='把影片拖到這裡';$('fileHint').textContent='或點一下選擇影片';$('dropzone').setAttribute('aria-disabled','false');$('start').disabled=true;});

$('startupNotice').hidden=true;
$('application').hidden=false;
// A fixed entry URL must restore the current job after a refresh or second launch.
(async()=>{
 try{
  const response=await fetch('/api/state',{credentials:'same-origin',cache:'no-store'});
  if(!response.ok)return;
  const state=await response.json();
  if(!selected&&!busy&&['uploading','preparing','processing','done'].includes(state.phase)){
   busy=true;$('uploadView').hidden=true;$('progressView').hidden=false;
   $('start').disabled=true;$('dropzone').setAttribute('aria-disabled','true');
   poll();
  }
 }catch(error){showError('無法連線，請確認本機程式仍在執行。');}
})();
}
