/* Offline Bradley-Terry pair-loss explorer. Updates hypothetical scalar scores. */
(function () {
    'use strict';
    var root = document.getElementById('w-preference-loss');
    if (!root) return;
    var a = document.getElementById('pref-a'), b = document.getElementById('pref-b');
    var shift = document.getElementById('pref-shift'), label = document.getElementById('pref-label');
    var result = document.getElementById('pref-result'), step = document.getElementById('pref-step');
    function sigmoid(x) { return x >= 0 ? 1 / (1 + Math.exp(-x)) : Math.exp(x) / (1 + Math.exp(x)); }
    function softplus(x) { return Math.max(x, 0) + Math.log1p(Math.exp(-Math.abs(x))); }
    function fmt(x) { return x.toLocaleString('en-US', {maximumFractionDigits: 6}); }
    function values() {
        if ([a, b, shift].some(function (el) { var x = Number(el.value); return el.value.trim() === '' || !Number.isFinite(x) || x < -1000 || x > 1000; })) return null;
        return [Number(a.value), Number(b.value), Number(shift.value), Number(label.value)];
    }
    function update() {
        var v = values(); step.disabled = !v;
        if (!v) { result.textContent = 'Enter finite scores and an offset between -1000 and 1000.'; return; }
        var delta = v[0] - v[1], p = sigmoid(delta), g = p - v[3];
        var loss = softplus((1 - 2 * v[3]) * delta);
        result.innerHTML = '<p><strong>Effective scores:</strong> A = ' + fmt(v[0] + v[2]) + ', B = ' + fmt(v[1] + v[2]) + '. Difference A − B = ' + fmt(delta) + '.</p>' +
            '<p><strong>P(A preferred to B): ' + fmt(p) + '</strong></p><div class="pref-track" aria-hidden="true"><div style="width:' + (100 * p) + '%"></div></div>' +
            '<p>Observed winner: ' + (v[3] ? 'A' : 'B') + '. Probability assigned to that observation: ' + fmt(v[3] ? p : 1 - p) + '.</p>' +
            '<div class="tbl-scroll"><table class="tbl"><thead><tr><th>Quantity</th><th>Value</th></tr></thead><tbody><tr><td>Pair loss (nats)</td><td>' + fmt(loss) + '</td></tr><tr><td>Gradient with respect to A score</td><td>' + fmt(g) + '</td></tr><tr><td>Gradient with respect to B score</td><td>' + fmt(-g) + '</td></tr><tr><td>A − B after one score step, rate 0.1</td><td>' + fmt(delta - 0.2 * g) + '</td></tr></tbody></table></div>' +
            '<p>A score-gradient step raises the labeled winner and lowers the other score, unless the gradient has numerically saturated. A common offset changes none of these pair quantities.</p>';
    }
    [a, b, shift, label].forEach(function (el) { el.addEventListener('input', update); el.addEventListener('change', update); });
    step.addEventListener('click', function () {
        var v = values(); if (!v) return;
        var g = sigmoid(v[0] - v[1]) - v[3];
        var nextA = v[0] - 0.1 * g, nextB = v[1] + 0.1 * g;
        if (Math.abs(nextA) > 1000 || Math.abs(nextB) > 1000) { result.textContent = 'This step would leave the allowed score range. Move the scores away from the boundary first.'; return; }
        a.value = String(Number(nextA.toFixed(6))); b.value = String(Number(nextB.toFixed(6))); update();
    });
    document.getElementById('pref-reset').addEventListener('click', function () { a.value = '1.2'; b.value = '0.8'; shift.value = '0'; label.value = '1'; update(); });
    update();
}());

/* User-controlled forward/backward trace of one independent-score update. */
(function () {
    'use strict';
    var root = document.getElementById('pref-stages');
    if (!root) return;
    var fields = ['pref-a', 'pref-b', 'pref-shift', 'pref-label'].map(function (id) { return document.getElementById(id); });
    var play = document.getElementById('pref-play'), next = document.getElementById('pref-next');
    var text = document.getElementById('pref-trace');
    var stage = 0, snapshot = null, playing = false, timer = null, committing = false;
    function fmt(x) { return x.toLocaleString('en-US', {maximumFractionDigits: 6}); }
    function valid() { return fields.slice(0, 3).every(function (el) { return el.value.trim() !== '' && Number.isFinite(Number(el.value)) && Math.abs(Number(el.value)) <= 1000; }); }
    function buttons() {
        play.textContent = playing ? 'Pause trace' : stage >= 6 ? 'Play another step' : stage > 0 ? 'Resume trace' : 'Play one step';
        play.setAttribute('aria-pressed', String(playing));
        play.disabled = next.disabled = !valid();
        next.textContent = stage >= 6 ? 'Start another step' : 'Next stage';
    }
    function stop() { playing = false; clearTimeout(timer); timer = null; buttons(); }
    function clearTrace() {
        stop(); stage = 0; snapshot = null;
        root.querySelectorAll('li').forEach(function (el) { el.className = ''; el.removeAttribute('aria-current'); });
        text.textContent = valid() ? 'Choose scores and a winner, then play or advance the trace.' : 'Enter valid scores before tracing a step.';
        buttons();
    }
    function capture() {
        var a = Number(fields[0].value), b = Number(fields[1].value), shift = Number(fields[2].value), y = Number(fields[3].value);
        var d = a - b, p = d >= 0 ? 1 / (1 + Math.exp(-d)) : Math.exp(d) / (1 + Math.exp(d));
        var z = (1 - 2 * y) * d;
        return {a:a, b:b, shift:shift, y:y, d:d, p:p, loss:Math.max(z, 0) + Math.log1p(Math.exp(-Math.abs(z))), g:p-y};
    }
    function advance() {
        if (!valid()) { clearTrace(); return; }
        if (!snapshot || stage >= 6) { stage = 0; snapshot = capture(); }
        stage += 1;
        var v = snapshot, winner = v.y ? 'A' : 'B';
        var descriptions = [
            'Read both scores for the same prompt: A = ' + fmt(v.a + v.shift) + ', B = ' + fmt(v.b + v.shift) + '. The observed winner is ' + winner + '.',
            'Subtract: delta = A − B = ' + fmt(v.d) + '. The shared offset cancels here.',
            'Forward: sigmoid(delta) = ' + fmt(v.p) + ' is P(A preferred to B). The observed winner ' + winner + ' receives probability ' + fmt(v.y ? v.p : 1-v.p) + '.',
            'Forward: stable binary cross-entropy with label ' + v.y + ' gives pair loss ' + fmt(v.loss) + ' nats. The loss measures fit to the comparison, not absolute answer quality.',
            'Backward: gradient of loss with respect to delta = probability − label = ' + fmt(v.g) + '. Subtraction sends gradient ' + fmt(v.g) + ' to A and ' + fmt(-v.g) + ' to B. Both come from the SAME pre-update scores.',
            ''
        ];
        if (stage === 6) {
            var na = v.a - 0.1 * v.g, nb = v.b + 0.1 * v.g;
            committing = true;
            try { document.getElementById('pref-step').click(); } finally { committing = false; }
            descriptions[5] = Math.abs(na) > 1000 || Math.abs(nb) > 1000 ? 'Update blocked: this step would exceed the allowed score range. Move the scores away from the boundary and try again.' :
                'Update ONCE: subtract 0.1 × each gradient. A: ' + fmt(v.a + v.shift) + ' → ' + fmt(Number(fields[0].value) + v.shift) + '; B: ' + fmt(v.b + v.shift) + ' → ' + fmt(Number(fields[1].value) + v.shift) + '. Inputs and the live table now show the updated scores. The trace above describes the completed step.';
            stop();
        }
        text.textContent = 'Stage ' + stage + ' of 6. ' + descriptions[stage-1];
        root.querySelectorAll('li').forEach(function (el, i) {
            el.className = i + 1 === stage ? 'current' + (stage >= 5 ? ' backward' : '') : i + 1 < stage ? 'visited' : '';
            if (i + 1 === stage) el.setAttribute('aria-current', 'step'); else el.removeAttribute('aria-current');
        });
        buttons();
    }
    function tick() { advance(); if (playing) timer = setTimeout(tick, 1800); }
    play.addEventListener('click', function () { if (playing) { stop(); return; } if (!valid()) return; playing = true; tick(); });
    next.addEventListener('click', function () { stop(); advance(); });
    document.getElementById('pref-rewind').addEventListener('click', clearTrace);
    fields.forEach(function (el) { el.addEventListener('input', clearTrace); el.addEventListener('change', clearTrace); });
    document.getElementById('pref-reset').addEventListener('click', clearTrace);
    document.getElementById('pref-step').addEventListener('click', function () { if (!committing) clearTrace(); });
    document.addEventListener('visibilitychange', function () { if (document.hidden) stop(); });
    window.addEventListener('beforeprint', stop);
    window.addEventListener('pagehide', stop);
    buttons();
}());
