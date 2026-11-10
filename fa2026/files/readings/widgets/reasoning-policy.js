/* One fixed state/action PPO contribution. No model update or rollout simulation. */
(function () {
    'use strict';
    var root = document.getElementById('w-ppo-clipping');
    if (!root) return;
    var fields = ['rp-ratio', 'rp-adv', 'rp-eps'].map(function (id) { return document.getElementById(id); });
    var limits = [[0.05, 2.5], [-10, 10], [0, 0.8]];
    var result = document.getElementById('rp-result'), plot = document.getElementById('rp-plot');
    function fmt(x) { return (Math.abs(x) < 1e-10 ? 0 : x).toLocaleString('en-US', {maximumFractionDigits: 6}); }
    function clip(r,e) { return Math.max(1-e, Math.min(1+e,r)); }
    function read() {
        var v = fields.map(function (el) { return Number(el.value); });
        var valid = fields.map(function (el,i) {
            var ok = el.value.trim() !== '' && Number.isFinite(v[i]) && v[i] >= limits[i][0] && v[i] <= limits[i][1];
            el.setAttribute('aria-invalid', String(!ok)); return ok;
        });
        return valid.every(function (ok) { return ok; }) ? v : null;
    }
    function contribution(r,a,e) { return Math.min(r*a,clip(r,e)*a); }
    function draw(r,a,e) {
        var low = Math.min(0,2.5*a), high = Math.max(0,2.5*a);
        if (a === 0) { low = -1; high = 1; }
        var margin = (high-low)*0.14; low -= margin; high += margin;
        var left=62, right=610, top=24, bottom=258;
        function x(v) { return left + v/2.5*(right-left); }
        function y(v) { return bottom - (v-low)/(high-low)*(bottom-top); }
        function path(clipped) {
            var out=''; for(var i=0;i<=250;i++) { var rr=i/100; out+=(i?'L':'M')+x(rr).toFixed(2)+','+y(clipped?contribution(rr,a,e):rr*a).toFixed(2); } return out;
        }
        var ticks = [0,1,2.5].map(function (t) { return '<text x="'+x(t)+'" y="279" text-anchor="middle">'+t+'</text>'; }).join('');
        plot.innerHTML='<svg viewBox="0 0 640 315" role="img" aria-labelledby="rp-plot-title rp-plot-desc"><title id="rp-plot-title">Raw and clipped surrogate versus probability ratio</title><desc id="rp-plot-desc">Dashed blue is ratio times advantage. Solid orange is the minimum of raw and bounded branches. '+(a>0?'The solid curve is flat above the upper threshold.':a<0?'The solid curve is flat below the lower threshold.':'All contributions are zero.')+'</desc>'+
            '<rect width="640" height="315" fill="white"/>'+
            '<path d="M'+left+','+top+'V'+bottom+'H'+right+'" stroke="#6b7480" fill="none"/>'+
            '<path d="M'+left+','+y(0)+'H'+right+'" stroke="#c6cdd6" fill="none"/>'+
            '<path d="M'+x(1-e)+','+top+'V'+bottom+'M'+x(1+e)+','+top+'V'+bottom+'" stroke="#8896a8" stroke-dasharray="3 4" fill="none"/>'+
            '<path d="'+path(false)+'" stroke="#002d72" stroke-width="3" stroke-dasharray="7 5" fill="none"/>'+
            '<path d="'+path(true)+'" stroke="#b94f00" stroke-width="3" fill="none"/>'+
            '<circle cx="'+x(r)+'" cy="'+y(contribution(r,a,e))+'" r="5" fill="#b94f00"/>'+ticks+
            '<text x="'+(left-10)+'" y="'+(y(0)+5)+'" text-anchor="end">0</text>'+
            '<text x="336" y="306" text-anchor="middle">Probability ratio r (not a probability)</text>'+
            '<text x="18" y="145" transform="rotate(-90 18 145)" text-anchor="middle">Surrogate contribution</text></svg>'+
            '<p class="rp-legend"><span>┄ Dashed blue: raw rA</span><span>━ Solid orange: clipped surrogate</span></p>';
    }
    function update() {
        var v = read();
        if (!v) { result.textContent='Enter ratio 0.05–2.5, advantage −10–10, and ε 0–0.8. All fields are required.'; plot.innerHTML=''; return; }
        var r=v[0],a=v[1],e=v[2],bounded=clip(r,e)*a,raw=r*a,value=Math.min(raw,bounded);
        var threshold=a>0?1+e:1-e;
        var kink=a!==0 && Math.abs(r-threshold)<1e-10;
        var flat=a===0 || (a>0 && r>1+e) || (a<0 && r<1-e);
        var slope=flat?0:a;
        var description=a===0?'Zero advantage: no contribution or gradient.':kink?'At the active clipping kink: there is no unique ordinary derivative.':flat?'Favorable change beyond the threshold: this term is flat.':'This term keeps its ordinary slope; it has not saturated on the favorable side.';
        var derivative=kink?(a>0?fmt(a)+' / 0':'0 / '+fmt(a))+' (left / right)':fmt(slope);
        result.innerHTML='<div class="tbl-scroll" tabindex="0" role="region" aria-label="Clipped surrogate values"><table class="tbl"><thead><tr><th>Quantity</th><th>Value</th></tr></thead><tbody>'+
            '<tr><td>Clipping thresholds</td><td>'+fmt(1-e)+' to '+fmt(1+e)+'</td></tr>'+
            '<tr><td>Raw branch rA</td><td>'+fmt(raw)+'</td></tr>'+
            '<tr><td>Bounded branch clip(r)A</td><td>'+fmt(bounded)+'</td></tr>'+
            '<tr><td>Clipped surrogate ℓ = minimum</td><td>'+fmt(value)+'</td></tr>'+
            '<tr><td>Loss for minimization −ℓ</td><td>'+fmt(-value)+'</td></tr>'+
            '<tr><td>Surrogate slope dℓ/dr</td><td>'+derivative+'</td></tr>'+
            '<tr><td>Loss derivative vs new log-probability</td><td>'+(kink?'Boundary convention required':fmt(-r*slope))+'</td></tr></tbody></table></div>'+
            '<p><strong>'+description+'</strong></p><p>The ratio remains '+fmt(r)+'. Evaluating this objective does not change or clamp the policy probability.</p>';
        draw(r,a,e);
    }
    function load(r,a,e) { [r,a,e].forEach(function (v,i) { fields[i].value=String(v); }); update(); }
    fields.forEach(function (el) { el.addEventListener('input',update); el.addEventListener('change',update); });
    document.getElementById('rp-positive').addEventListener('click',function () { load(1.5,2,0.2); });
    document.getElementById('rp-negative').addEventListener('click',function () { load(0.6,-2,0.2); });
    document.getElementById('rp-reset').addEventListener('click',function () { load(1.5,2,0.2); });
    update();
}());
