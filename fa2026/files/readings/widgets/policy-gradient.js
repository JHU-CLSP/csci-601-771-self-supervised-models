/* Offline finite two-response model. All steps update theta = logit(P(A)). */
(function () {
    'use strict';
    var root = document.getElementById('w-policy-gradient');
    if (!root) return;
    var ids = ['pg-p', 'pg-q', 'pg-ra', 'pg-rb', 'pg-beta', 'pg-rate'];
    var inputs = ids.map(function (id) { return document.getElementById(id); });
    var selected = document.getElementById('pg-outcome');
    var result = document.getElementById('pg-result'), status = document.getElementById('pg-status');
    var exact = document.getElementById('pg-exact'), sample = document.getElementById('pg-sample');
    var bounds = [[0.01, 0.99], [0.01, 0.99], [-20, 20], [-20, 20], [0, 20], [0.001, 1]];
    function fmt(x) { return x.toLocaleString('en-US', {maximumFractionDigits: 6}); }
    function logit(p) { return Math.log(p) - Math.log1p(-p); }
    function sigmoid(t) { return t >= 0 ? 1 / (1 + Math.exp(-t)) : Math.exp(t) / (1 + Math.exp(t)); }
    function read() {
        var v = inputs.map(function (el) { return Number(el.value); });
        var validFields = inputs.map(function (el, i) {
            var valid = el.value.trim() !== '' && Number.isFinite(v[i]) && v[i] >= bounds[i][0] && v[i] <= bounds[i][1];
            el.setAttribute('aria-invalid', String(!valid));
            return valid;
        });
        return validFields.every(function (valid) { return valid; }) ? v : null;
    }
    function calculate(v) {
        var p = v[0], q = v[1], ra = v[2], rb = v[3], beta = v[4];
        var la = Math.log(p / q), lb = Math.log((1 - p) / (1 - q));
        var kl = p * la + (1 - p) * lb, reward = p * ra + (1 - p) * rb;
        var ga = (ra - beta * la) * (1 - p), gb = (rb - beta * lb) * (-p);
        return {la:la, lb:lb, kl:kl, reward:reward, objective:reward-beta*kl,
            ga:ga, gb:gb, gradient:p*ga+(1-p)*gb};
    }
    function update() {
        var v = read(); exact.disabled = sample.disabled = !v;
        if (!v) {
            result.textContent = 'Use probabilities 0.01–0.99, rewards −20–20, β 0–20, and step size 0.001–1. All fields are required.';
            return;
        }
        var c = calculate(v), chosen = selected.value === 'A' ? c.ga : c.gb;
        result.innerHTML = '<p><strong>Current P(A): ' + fmt(v[0]) + '</strong>; reference P(A): ' + fmt(v[1]) + '.</p>' +
            '<div class="pg-track" aria-hidden="true"><span style="width:' + (100*v[0]) + '%">A</span><span style="width:' + (100*(1-v[0])) + '%">B</span></div>' +
            '<div class="tbl-scroll" tabindex="0" role="region" aria-label="Policy calculations"><table class="tbl"><thead><tr><th>Quantity</th><th>Value</th></tr></thead><tbody>' +
            '<tr><td>Expected predicted reward</td><td>' + fmt(c.reward) + '</td></tr>' +
            '<tr><td>Forward KL (nats)</td><td>' + fmt(c.kl) + '</td></tr>' +
            '<tr><td>Regularized objective F</td><td>' + fmt(c.objective) + '</td></tr>' +
            '<tr><td>Exact dF/dθ</td><td>' + fmt(c.gradient) + '</td></tr>' +
            '<tr><td>A / B sequence log-ratios</td><td>' + fmt(c.la) + ' / ' + fmt(c.lb) + '</td></tr>' +
            '<tr><td>A / B sampled gradients</td><td>' + fmt(c.ga) + ' / ' + fmt(c.gb) + '</td></tr>' +
            '<tr><td>Selected ' + selected.value + ' sample gradient</td><td>' + fmt(chosen) + '</td></tr></tbody></table></div>' +
            '<p>Policy-weighted sample mean: ' + fmt(v[0]) + ' × ' + fmt(c.ga) + ' + ' + fmt(1-v[0]) + ' × ' + fmt(c.gb) + ' = ' + fmt(c.gradient) + '. One selected sample can point in a different direction from this mean.</p>' +
            (v[4] > 0 ? '<p>Unconstrained optimum for these two fixed rewards: P(A) = ' + fmt(sigmoid(logit(v[1])+(v[2]-v[3])/v[4])) + '. The input controls cover only 0.01–0.99.</p>' :
                '<p>With β = 0, the reward-only optimum puts all mass on the higher-reward response; equal rewards give a constant objective.</p>');
    }
    function step(useExact) {
        var v = read(); if (!v) return;
        var c = calculate(v), g = useExact ? c.gradient : selected.value === 'A' ? c.ga : c.gb;
        var p = sigmoid(logit(v[0]) + v[5]*g);
        if (p < 0.01 || p > 0.99) {
            status.textContent = 'Step blocked: it would leave the displayed probability range. Reduce the step size or move P(A) away from the boundary.';
            return;
        }
        inputs[0].value = String(p); update();
        var after = calculate(read());
        status.textContent = (useExact ? 'Exact-gradient' : 'Selected ' + selected.value + ' sample') + ' step: θ += ' + fmt(v[5]) + ' × ' + fmt(g) +
            '; P(A) ' + fmt(v[0]) + ' → ' + fmt(p) + '; F ' + fmt(c.objective) + ' → ' + fmt(after.objective) + '. The table now describes the new policy.';
    }
    function load(values, text) { inputs.forEach(function (el,i) { el.value = String(values[i]); }); selected.value = 'A'; update(); status.textContent = text; }
    inputs.concat([selected]).forEach(function (el) { el.addEventListener('input', function () { update(); status.textContent = 'Inputs changed. Predict the direction before taking a step.'; }); });
    exact.addEventListener('click', function () { step(true); });
    sample.addEventListener('click', function () { step(false); });
    document.getElementById('pg-reset').addEventListener('click', function () { load([0.25,0.5,3,1,0.5,0.1], 'Reset to the initial example.'); });
    document.getElementById('pg-kl-example').addEventListener('click', function () { load([0.8,0.5,3,1,1,0.1], 'Loaded §5: F ≈ 2.407255 and exact gradient ≈ 0.098193.'); });
    update();
}());
