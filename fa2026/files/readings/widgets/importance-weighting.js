/* Two-response, fixed-state importance sampling; exact enumeration, no RNG. */
(function () {
    'use strict';
    var root = document.getElementById('w-importance');
    if (!root) return;
    var old = root.querySelector('#iw-old'), current = root.querySelector('#iw-current'), count = root.querySelector('#iw-count');
    var result = root.querySelector('#iw-result'), plot = root.querySelector('#iw-plot');
    function fmt(x) { return (Math.abs(x) < 1e-10 ? 0 : x).toFixed(4).replace(/\.?0+$/, ''); }
    function row(label, value, kind) {
        return '<div class="iw-bar-row"><span>'+label+'</span><div class="iw-track"><div class="iw-bar '+kind+'" style="width:'+(100*value)+'%"></div></div><span>'+fmt(value)+'</span></div>';
    }
    function update() {
        var q=Number(old.value)/100, p=Number(current.value)/100, k=Number(count.value), z=k/20;
        root.querySelector('#iw-old-value').textContent=fmt(q);
        root.querySelector('#iw-current-value').textContent=fmt(p);
        root.querySelector('#iw-count-value').textContent=k+' of 20';
        old.setAttribute('aria-valuetext',fmt(q)); current.setAttribute('aria-valuetext',fmt(p)); count.setAttribute('aria-valuetext',k+' of 20 responses');
        var rA=p/q, rB=(1-p)/(1-q), expected=4*p-(1-p), corrected=4*q*rA-(1-q)*rB;
        var sample=4*z*rA-(1-z)*rB, unweighted=4*z-(1-z), meanRatio=z*rA+(1-z)*rB;
        plot.innerHTML='<div><strong>Response A · f(A) = 4</strong>'+row('Old q(A)',q,'iw-old')+row('Current p(A)',p,'iw-current')+row('Batch fraction',z,'iw-sample')+'<p>Weight p(A)/q(A) = <strong>'+fmt(rA)+'</strong></p></div>'+
            '<div><strong>Response B · f(B) = −1</strong>'+row('Old q(B)',1-q,'iw-old')+row('Current p(B)',1-p,'iw-current')+row('Batch fraction',1-z,'iw-sample')+'<p>Weight p(B)/q(B) = <strong>'+fmt(rB)+'</strong></p></div>';
        var rows=[['Exact current expectation',expected],['Exact old-distribution weighted expectation',corrected],['This batch: unweighted mean',unweighted],['This batch: importance-weighted mean',sample],['This batch: mean importance ratio',meanRatio]];
        result.innerHTML='<div class="tbl-scroll" tabindex="0" role="region" aria-label="Importance-weighted expectations and batch estimates"><table class="tbl"><thead><tr><th>Quantity</th><th>Value</th></tr></thead><tbody>'+rows.map(function(r){return '<tr><td>'+r[0]+'</td><td>'+fmt(r[1])+'</td></tr>';}).join('')+'</tbody></table></div>'+
            '<p>The two exact expectations agree. This chosen batch '+(Math.abs(sample-expected)<1e-9?'also matches them.':'differs by '+fmt(sample-expected)+' (weighted batch mean minus target).')+' Its mean ratio need not be 1.</p>'+
            (k===0?'<p><strong>No A was observed.</strong> Weighting cannot add a missing observation to this batch, even though q(A) is positive.</p>':k===20?'<p><strong>No B was observed.</strong> Weighting cannot add a missing observation to this batch, even though q(B) is positive.</p>':'')+
            '<p>A contributes '+fmt(4*rA)+' each time it appears, before dividing the batch sum by 20. This makes a rare old-policy response consequential when its ratio is large.</p>';
    }
    function load(q,p,k){old.value=q;current.value=p;count.value=k;update();}
    [old,current,count].forEach(function(el){el.addEventListener('input',update);el.addEventListener('change',update);});
    root.querySelector('#iw-example').addEventListener('click',function(){load(25,60,5);});
    root.querySelector('#iw-missing').addEventListener('click',function(){load(5,90,0);});
    root.querySelector('#iw-one').addEventListener('click',function(){load(5,90,1);});
    update();
}());
