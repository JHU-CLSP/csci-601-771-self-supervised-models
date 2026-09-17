/* Learning-rate explorer: gradient descent on a 1-D loss surface, drawn to
   scale, so the three regimes (too small, about right, divergent) are
   something you watch rather than something you are told about.
   No dependencies. */
(function () {
    'use strict';

    var mount = document.getElementById('w-gradient-descent');
    if (!mount) return;

    /* Two surfaces: a clean convex bowl, and one with a local minimum. */
    var SURFACES = {
        bowl: {
            name: 'Convex bowl',
            f: function (x) { return 0.5 * x * x; },
            df: function (x) { return x; },
            domain: [-5, 5], start: -4.2, yMax: 13
        },
        wiggle: {
            name: 'Two minima',
            f: function (x) { return 0.06 * Math.pow(x, 4) - 0.5 * x * x + 0.6 * x + 3; },
            df: function (x) { return 0.24 * Math.pow(x, 3) - x + 0.6; },
            domain: [-5, 5], start: -4.2, yMax: 13
        }
    };

    mount.innerHTML =
        '<div class="w-row">' +
        '  <span class="w-stat-label" style="align-self:center">Surface:</span>' +
        '  <button class="w-btn on" data-surf="bowl" type="button">Convex bowl</button>' +
        '  <button class="w-btn" data-surf="wiggle" type="button">Two minima</button>' +
        '</div>' +
        '<div class="w-row">' +
        '  <label for="wgd-lr" style="flex:1 1 100%;margin-bottom:2px">' +
        '    Learning rate <span id="wgd-lr-val" style="font-family:ui-monospace,Menlo,monospace;' +
        'color:#002d72">0.30</span>' +
        '  </label>' +
        '  <input type="range" id="wgd-lr" min="0.02" max="2.4" step="0.02" value="0.3" ' +
        '         style="flex:1 1 260px">' +
        '  <button class="w-btn" id="wgd-step" type="button">Step</button>' +
        '  <button class="w-btn" id="wgd-run" type="button">Run 25</button>' +
        '  <button class="w-btn" id="wgd-reset" type="button">Reset</button>' +
        '</div>' +
        '<svg id="wgd-svg" viewBox="0 0 620 260" width="100%" ' +
        '     style="display:block;margin:4px 0 0" role="img" ' +
        '     aria-label="A loss curve with the current parameter value marked, and the path taken by ' +
        'gradient descent."></svg>' +
        '<div class="w-stats">' +
        '  <div><div class="w-stat-label">Steps</div><div class="w-stat-val" id="wgd-n">0</div></div>' +
        '  <div><div class="w-stat-label">&theta;</div><div class="w-stat-val" id="wgd-x">&mdash;</div></div>' +
        '  <div><div class="w-stat-label">Loss</div><div class="w-stat-val" id="wgd-loss">&mdash;</div></div>' +
        '  <div><div class="w-stat-label">Gradient</div><div class="w-stat-val" id="wgd-grad">&mdash;</div></div>' +
        '</div>' +
        '<p class="w-note" id="wgd-note"></p>';

    var $svg = document.getElementById('wgd-svg');
    var $lr = document.getElementById('wgd-lr');
    var $lrVal = document.getElementById('wgd-lr-val');
    var $n = document.getElementById('wgd-n');
    var $x = document.getElementById('wgd-x');
    var $loss = document.getElementById('wgd-loss');
    var $grad = document.getElementById('wgd-grad');
    var $note = document.getElementById('wgd-note');

    var surf = SURFACES.bowl;
    var path = [surf.start];
    var diverged = false;

    var W = 620, H = 260, PAD = 34;

    function sx(x) {
        var d = surf.domain;
        return PAD + (x - d[0]) / (d[1] - d[0]) * (W - 2 * PAD);
    }
    function sy(y) {
        return H - PAD - Math.max(0, Math.min(1, y / surf.yMax)) * (H - 2 * PAD);
    }

    function draw() {
        var d = surf.domain, pts = [];
        for (var i = 0; i <= 240; i++) {
            var x = d[0] + (d[1] - d[0]) * i / 240;
            pts.push(sx(x).toFixed(1) + ',' + sy(surf.f(x)).toFixed(1));
        }

        var cur = path[path.length - 1];
        var inRange = cur >= d[0] && cur <= d[1] && isFinite(cur);

        var hops = '';
        for (var k = 1; k < path.length; k++) {
            var a = path[k - 1], b = path[k];
            if (![a, b].every(function (v) { return isFinite(v) && v >= d[0] && v <= d[1]; })) continue;
            hops += '<line x1="' + sx(a).toFixed(1) + '" y1="' + sy(surf.f(a)).toFixed(1) +
                    '" x2="' + sx(b).toFixed(1) + '" y2="' + sy(surf.f(b)).toFixed(1) +
                    '" stroke="#b02a55" stroke-width="1.4" stroke-dasharray="3 2" opacity="0.75"/>';
            hops += '<circle cx="' + sx(a).toFixed(1) + '" cy="' + sy(surf.f(a)).toFixed(1) +
                    '" r="2.6" fill="#b02a55" opacity="0.5"/>';
        }

        $svg.innerHTML =
            '<line x1="' + PAD + '" y1="' + (H - PAD) + '" x2="' + (W - PAD) + '" y2="' + (H - PAD) +
            '" stroke="#d8dde4" stroke-width="1"/>' +
            '<polyline points="' + pts.join(' ') + '" fill="none" stroke="#68ACE5" stroke-width="2.2"/>' +
            hops +
            (inRange
                ? '<circle cx="' + sx(cur).toFixed(1) + '" cy="' + sy(surf.f(cur)).toFixed(1) +
                  '" r="6" fill="#002d72"/>'
                : '') +
            '<text x="' + PAD + '" y="' + (H - 12) + '" font-family="roboto, sans-serif" ' +
            'font-size="11" fill="#6b7480">&theta;</text>' +
            '<text x="' + (W - PAD) + '" y="18" text-anchor="end" font-family="roboto, sans-serif" ' +
            'font-size="11" fill="#6b7480">loss surface</text>';
    }

    function readout() {
        var cur = path[path.length - 1];
        $n.textContent = path.length - 1;
        if (!isFinite(cur) || Math.abs(cur) > 1e6) {
            $x.textContent = '→ ∞';
            $loss.textContent = '→ ∞';
            $grad.textContent = '→ ∞';
        } else {
            $x.textContent = cur.toFixed(3);
            $loss.textContent = surf.f(cur).toFixed(3);
            $grad.textContent = surf.df(cur).toFixed(3);
        }

        var lr = Number($lr.value);

        /* Judge the regime from what actually happened, not from a threshold on
           the learning rate -- the threshold depends on the surface. */
        var regime = 'descending';
        if (path.length >= 3) {
            var losses = path.slice(-6).map(surf.f);
            var first = losses[0], last = losses[losses.length - 1];
            var signs = 0;
            for (var i = 2; i < path.length; i++) {
                if ((path[i] - path[i - 1]) * (path[i - 1] - path[i - 2]) < 0) signs++;
            }
            if (last > first * 1.0001) regime = 'diverging';
            else if (signs > (path.length - 2) * 0.5) regime = 'oscillating';
        }

        if (path.length === 1) {
            $note.innerHTML = 'Predict the trajectory, then press Step or Run 25. The curve shows ' +
                'the loss; the marker starts at &theta; = -4.2.';
        } else if (surf === SURFACES.bowl && Math.abs(lr - 2) < 1e-10) {
            $note.innerHTML = '<b>Oscillating without progress.</b> At &eta; = 2, each update ' +
                'flips the sign of &theta; without changing its magnitude or loss.';
        } else if (diverged) {
            $note.innerHTML = '<b style="color:#b02a55">Run stopped: values grew beyond the safety limit.</b> ' +
                'Reset and try a smaller learning rate. The convex bowl converges for 0 &lt; &eta; &lt; 2; ' +
                'the other surface has different curvature and no such shared threshold.';
        } else if (regime === 'diverging') {
            $note.innerHTML = '<b style="color:#b02a55">Loss has increased over recent steps.</b> ' +
                'Try a smaller learning rate. On the convex bowl, &eta; &gt; 2 makes the magnitude ' +
                'grow on every step; increasing loss over a few steps is not a general proof of divergence.';
        } else if (regime === 'oscillating') {
            $note.innerHTML = 'The updates are <b>oscillating</b>. Compare the recent losses: ' +
                'oscillation can accompany shrinking steps, a cycle, or instability.';
        } else if (Math.abs(surf.df(cur)) < 1e-8) {
            $note.innerHTML = 'The gradient is approximately zero at the current point. On the bowl ' +
                'this is its minimum; on a nonconvex surface, a small gradient alone does not certify a global minimum.';
        } else if (lr < 0.08) {
            $note.innerHTML = 'Small learning rate: updates can be slow. Compare progress after the ' +
                'same number of steps rather than assuming a smaller rate is always better.';
        } else if (surf === SURFACES.wiggle) {
            $note.innerHTML = 'There are two minima. Gradient descent may reach a local or global ' +
                'minimum depending on initialization and step size; local improvement does not guarantee the global minimum.';
        } else {
            $note.innerHTML = 'For this bowl with 0 &lt; &eta; &lt; 2, the magnitude shrinks on each ' +
                'step until numerical precision limits further progress.';
        }
    }

    function step() {
        if (diverged) return;
        var cur = path[path.length - 1];
        var next = cur - Number($lr.value) * surf.df(cur);
        path.push(next);
        if (!isFinite(next) || Math.abs(next) > 1e6) diverged = true;
        if (path.length > 400) path.shift();
        draw();
        readout();
    }

    function reset() {
        path = [surf.start];
        diverged = false;
        draw();
        readout();
    }

    document.getElementById('wgd-step').addEventListener('click', step);
    document.getElementById('wgd-run').addEventListener('click', function () {
        for (var i = 0; i < 25 && !diverged; i++) step();
    });
    document.getElementById('wgd-reset').addEventListener('click', reset);
    $lr.addEventListener('input', function () {
        $lrVal.textContent = Number($lr.value).toFixed(2);
        reset();
    });
    mount.querySelectorAll('[data-surf]').forEach(function (btn) {
        btn.addEventListener('click', function () {
            surf = SURFACES[btn.dataset.surf];
            mount.querySelectorAll('[data-surf]').forEach(function (b) {
                b.classList.toggle('on', b === btn);
            });
            reset();
        });
    });

    reset();
})();
