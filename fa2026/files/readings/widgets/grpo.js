/* Teaching explorers for handout 21. No network, model updates, or random samples. */
(function(){
'use strict';
const byId=id=>document.getElementById(id);
const fmt=n=>Math.abs(n)<1e-10?'0':n.toFixed(6);
function group(){
 const k=Number(byId('gr-k').value), p=k/8, sigma=Math.sqrt(p*(1-p));
 byId('gr-k-value').textContent=k;
 const mixed=k>0&&k<8;
 const success=mixed?(1-p)/sigma:0, failure=mixed?-p/sigma:0;
 byId('gr-group-result').innerHTML='<div class="gr-metrics">'+[
  ['Mean reward',p],['Population std',sigma],['Success advantage',success],['Failure advantage',failure]
 ].map(([label,n])=>'<div><span>'+label+'</span><strong>'+fmt(n)+'</strong></div>').join('')+'</div><p>'+
 (mixed? k+' success'+(k===1?'':'es')+' and '+(8-k)+' failure'+(k===7?'':'s')+'. Coefficients ignore the negligible stabilizer.':
 'No reward contrast: all observed advantages are zero. The absent outcome has no coefficient in this group.')+'</p>';
}
function curve(a,low,high){
 const W=Math.max(230,Math.min(560,byId('gr-curves').clientWidth||560)),H=190,L=38,R=12,T=24,B=40;
 const x=r=>L+(r-.4)/1.2*(W-L-R);
 const y=v=>T+(a>0?(1.7-v)/1.4:(-.3-v)/1.4)*(H-T-B);
 const value=r=>a>0?a*Math.min(r,1+high):a*Math.max(r,1-low);
 const cutoff=a>0?1+high:1-low;
 const path=(f)=>Array.from({length:121},(_,i)=>{let r=.4+i*.01;return(i?'L':'M')+x(r).toFixed(2)+','+y(f(r)).toFixed(2)}).join(' ');
 const ticks=(W<360?[.4,1,1.6]:[.4,.8,1,1.2,1.6]).map(r=>'<text x="'+x(r)+'" y="'+(H-22)+'" text-anchor="middle">'+r.toFixed(1)+'</text>').join('');
 const yTicks=(a>0?[.4,1,1.6]:[-1.6,-1,-.4]).map(v=>'<text x="'+(L-7)+'" y="'+(y(v)+4)+'" text-anchor="end">'+v.toFixed(1)+'</text>').join('');
 const start=a>0?x(cutoff):x(.4),end=a>0?x(1.6):x(cutoff);
 return '<div class="gr-curve"><h4>Advantage '+(a>0?'+1':'−1')+': '+(a>0?'reinforcement':'suppression')+'</h4><svg viewBox="0 0 '+W+' '+H+'" role="img" aria-label="'+(a>0?'Positive advantage: rising until upper threshold, then flat.':'Negative advantage: flat below lower threshold, then decreasing.')+'">'+
 '<rect class="plateau" x="'+start+'" y="'+T+'" width="'+(end-start)+'" height="'+(H-T-B)+'"/>'+
 '<path class="axis" d="M'+L+','+T+'V'+(H-B)+'H'+(W-R)+'"/>'+ticks+yTicks+
 '<path class="raw" d="'+path(r=>r*a)+'"/><path class="active" d="'+path(value)+'"/>'+
 '<line class="threshold" x1="'+x(cutoff)+'" x2="'+x(cutoff)+'" y1="'+T+'" y2="'+(H-B)+'"/>'+
 '<text x="'+x(cutoff)+'" y="15" text-anchor="middle">threshold '+cutoff.toFixed(2)+'</text><text x="'+((W+L-R)/2)+'" y="'+(H-3)+'" text-anchor="middle">Probability ratio r</text></svg></div>';
}
function clipping(){
 const low=Number(byId('gr-low').value),high=Number(byId('gr-high').value);
 byId('gr-low-value').textContent=low.toFixed(2);byId('gr-high-value').textContent=high.toFixed(2);
 byId('gr-curves').innerHTML=curve(1,low,high)+curve(-1,low,high);
 byId('gr-thresholds').textContent='For old probability 0.01: reinforcement plateaus above new probability '+(.01*(1+high)).toFixed(4)+'; suppression plateaus below '+(.01*(1-low)).toFixed(4)+'. These are incentive thresholds, not probability bounds.';
}
if(!byId('gr-k'))return;
byId('gr-k').addEventListener('input',group);
byId('gr-low').addEventListener('input',clipping);byId('gr-high').addEventListener('input',clipping);
window.addEventListener('resize',clipping);
group();clipping();
})();
