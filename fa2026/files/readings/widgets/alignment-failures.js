/* Exact two-response selection counterexample. No random sampling or model calls. */
(function(){
'use strict';
const id=x=>document.getElementById(x);
if(!id('af-p'))return;
function kl(p,q){return (p?p*Math.log(p/q):0)+(p<1?(1-p)*Math.log((1-p)/(1-q)):0)}
function row(label,p,q){
 const correct=1-p;
 return '<section class="af-card" data-false="'+p+'"><h4>'+label+'</h4><div class="af-bar" role="img" aria-label="Correct '+(100*correct).toFixed(2)+' percent; false '+(100*p).toFixed(2)+' percent"><div class="af-correct" style="width:'+(100*correct)+'%"></div><div class="af-false" style="width:'+(100*p)+'%"></div></div><p class="af-prob">Correct A: '+(100*correct).toFixed(2)+'% · False B: '+(100*p).toFixed(2)+'%</p><div class="af-stats">'+[
 ['Expected proxy',1+p],['Correctness',correct],['KL to reference',kl(p,q)]
 ].map(([label,v])=>'<div><span>'+label+'</span><strong>'+v.toFixed(4)+'</strong></div>').join('')+'</div></section>';
}
function update(){
 const p=Number(id('af-p').value),n=Number(id('af-n').value),beta=Number(id('af-beta').value);
 id('af-p-value').textContent=p.toFixed(2);id('af-n-value').textContent=n;id('af-beta-value').textContent=beta.toFixed(2);
 const selected=1-Math.pow(1-p,n),z=Math.log(p/(1-p))+1/beta,opt=1/(1+Math.exp(-z));
 id('af-result').innerHTML=row('Reference generator q',p,p)+row('Best-of-'+n+' selection',selected,p)+row('Exact KL optimum (β = '+beta.toFixed(2)+')',opt,p);
}
['af-p','af-n','af-beta'].forEach(x=>id(x).addEventListener('input',update));update();
})();
