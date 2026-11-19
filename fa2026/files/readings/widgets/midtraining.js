/* Fixed-content RoPE/PI explorer: positions in tokens, angles in radians. */
(function(){
'use strict';
const id=x=>document.getElementById(x);if(!id('mt-distance'))return;
function rotate(angle){return [Math.cos(angle),Math.sin(angle)]}
function panel(i,query,key){
 const q=rotate(query),k=rotate(key),cx=140,cy=100,r=70;
 const line=(v,color,marker)=>'<line x1="'+cx+'" y1="'+cy+'" x2="'+(cx+r*v[0]).toFixed(5)+'" y2="'+(cy-r*v[1]).toFixed(5)+'" stroke="'+color+'" stroke-width="3" marker-end="url(#'+marker+')"/>';
 return '<div class="mt-plot"><h4>Pair '+(i+1)+' · ω = '+(i===0?'1':'0.1')+'</h4><svg viewBox="0 0 280 200" role="img" aria-label="Unit-length query and key arrows for frequency '+(i===0?'1':'0.1')+'"><defs><marker id="mt-q'+i+'" markerWidth="7" markerHeight="7" refX="5" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6" fill="#073875"/></marker><marker id="mt-k'+i+'" markerWidth="7" markerHeight="7" refX="5" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6" fill="#b65c00"/></marker></defs><circle cx="140" cy="100" r="70" stroke="#b8c1cd" fill="none"/><path d="M45 100 H235 M140 15 V185" stroke="#d4dae2" fill="none"/>'+line(k,'#b65c00','mt-k'+i)+line(q,'#073875','mt-q'+i)+'</svg></div>';
}
function update(){
 const distance=Number(id('mt-distance').value),shift=Number(id('mt-shift').value),scale=Number(id('mt-scale').value),m=shift+distance,n=shift;
 id('mt-distance-value').textContent=distance;id('mt-shift-value').textContent=shift;id('mt-scale-value').textContent=scale;
 id('mt-positions').textContent='Query position m = '+m+'; key position n = '+n+'; relative offset n − m = '+(-distance)+'. With PI, use m/s = '+(m/scale).toFixed(3)+' and n/s = '+(n/scale).toFixed(3)+'.';
 let html='',dot=0,baseline=0;
 [1,.1].forEach((w,i)=>{const qa=rotate(m*w/scale),ka=rotate(n*w/scale);dot+=qa[0]*ka[0]+qa[1]*ka[1];baseline+=Math.cos(distance*w/scale);html+=panel(i,m*w/scale,n*w/scale)});
 id('mt-plots').innerHTML=html;
 const raw=Math.cos(distance)+Math.cos(.1*distance),residual=Math.abs(dot-baseline);
 id('mt-results').innerHTML='<div class="mt-metrics">'+[['Dot before PI',raw],['Dot after PI',dot],['Scaled logit after PI',dot/2],['Joint-shift residual',residual]].map(([label,value])=>'<div><span>'+label+'</span><strong>'+value.toFixed(6)+'</strong></div>').join('')+'</div>';
}
['mt-distance','mt-shift','mt-scale'].forEach(x=>id(x).addEventListener('input',update));update();
})();

/* Adapted self-study ideas; all numbers below are explicit toy calculations. */
(function(){
'use strict';
const id=x=>document.getElementById(x);
const metrics=items=>'<div class="mt-metrics">'+items.map(([k,v])=>'<div><span>'+k+'</span><strong>'+v+'</strong></div>').join('')+'</div>';
function mask(){
 const a=Number(id('mask-answer').value),sft=id('mask-mode').value==='sft';
 id('mask-answer-value').textContent=a.toFixed(1);
 const tokens=[['BOS',null,false],['&lt;user&gt;',2,false],['2+2?',3,false],['&lt;assistant&gt;',1,false],['4',a,true],['&lt;eos&gt;',.3,true],['&lt;pad&gt;',null,false]];
 id('mask-tokens').innerHTML=tokens.map(([label,nll,response])=>{const active=nll!==null&&(!sft||response);return '<div class="mt-token '+(active?'mt-active':'mt-inactive')+'"><strong>'+label+'</strong><small>'+(nll===null?(label==='BOS'?'Initial context':'Padding excluded'):(active?'Loss: ':'Masked NLL: ')+nll.toFixed(1))+'</small></div>'}).join('');
 const mid=(6.3+a)/5,answer=(a+.3)/2;
 id('mask-results').innerHTML=metrics([['Selected target count',sft?2:5],['Mid-training mean NLL',mid.toFixed(3)],['Response-only mean NLL',answer.toFixed(3)]]);
 id('mask-results').dataset.mid=mid;id('mask-results').dataset.sft=answer;
}
['mask-answer','mask-mode'].forEach(x=>id(x).addEventListener('input',mask));mask();
function spectrum(){
 const d=Number(id('spec-d').value),base=Number(id('spec-base').value),s=Number(id('spec-factor').value),method=id('spec-method').value;
 const old=Array.from({length:d/2},(_,i)=>2*Math.PI*base**(2*i/d));
 const next=old.map((v,i)=>method==='pi'?v*s:method==='base'?v*s**(2*i/d):v);
 const n0=4096,n1=n0*s,turn0=old.filter(v=>n0-1>=v).length,turn1=next.filter(v=>n1-1>=v).length;
 const width=Math.max(260,Math.min(760,id('spec-plot').clientWidth||560)),height=300,l=44,r=12,t=16,b=42;
 const ymax=Math.ceil(Math.log10(Math.max(n1,...next,...old))),ymin=0;
 const x=i=>l+i/(old.length-1)*(width-l-r),y=v=>t+(ymax-Math.log10(v))/(ymax-ymin)*(height-t-b);
 let svg='<svg viewBox="0 0 '+width+' '+height+'" role="img" aria-label="Log wavelength by rotary pair index, original and modified schedules">';
 for(let j=0;j<=ymax;j++)svg+='<path d="M'+l+' '+y(10**j)+' H'+(width-r)+'" stroke="#e0e4e9" fill="none"/><text x="'+(l-6)+'" y="'+(y(10**j)+4)+'" text-anchor="end">'+j+'</text>';
 [[n0,'#22764d','4096'],[n1,'#ad5700',String(n1)]].forEach(([n,color,label])=>{svg+='<path d="M'+l+' '+y(n)+' H'+(width-r)+'" stroke="'+color+'" stroke-dasharray="2 4" fill="none"/><text x="'+(l+4)+'" y="'+(y(n)-5)+'" fill="'+color+'">N = '+label+'</text>'});
 const path=vs=>vs.map((v,i)=>(i?'L':'M')+x(i).toFixed(2)+' '+y(v).toFixed(2)).join(' ');
 svg+='<path d="'+path(old)+'" fill="none" stroke="#777" stroke-width="3" stroke-dasharray="6 4"/><path d="'+path(next)+'" fill="none" stroke="#073875" stroke-width="2"/>';
 [0,Math.floor((old.length-1)/2),old.length-1].forEach(i=>svg+='<text x="'+x(i)+'" y="'+(height-b+18)+'" text-anchor="middle">'+i+'</text>');
 svg+='<text x="'+(width/2)+'" y="'+(height-4)+'" text-anchor="middle">Pair index i · y = log₁₀ λᵢ</text></svg>';
 id('spec-plot').innerHTML=svg;
 id('spec-results').innerHTML=metrics([['First wavelength (modified)',next[0].toFixed(2)],['Last wavelength (modified)',next.at(-1).toFixed(2)],['First / last wavelength multiplier',(next[0]/old[0]).toFixed(3)+' / '+(next.at(-1)/old.at(-1)).toFixed(3)],['Pairs completing ≥1 turn: old / target',turn0+' / '+turn1+' of '+old.length]]);
 id('spec-results').dataset.values=JSON.stringify({old,next,turn0,turn1});
}
['spec-d','spec-base','spec-factor','spec-method'].forEach(x=>id(x).addEventListener('input',spectrum));spectrum();window.addEventListener('resize',spectrum);
})();
