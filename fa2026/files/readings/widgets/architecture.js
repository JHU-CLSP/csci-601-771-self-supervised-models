/* Original offline component experiments for Handout 13. */
(function () {
    'use strict';
    var scale = document.getElementById('norm-scale');
    var shift = document.getElementById('norm-shift');
    function updateNorm() {
        var a = Number(scale.value), c = Number(shift.value), eps = 1e-6;
        var x = [1, 2, 3, 4].map(function (v) { return a * v + c; });
        var mean = x.reduce(function (s, v) { return s + v; }, 0) / 4;
        var variance = x.reduce(function (s, v) { return s + Math.pow(v - mean, 2); }, 0) / 4;
        var second = x.reduce(function (s, v) { return s + v * v; }, 0) / 4;
        var ln = x.map(function (v) { return (v - mean) / Math.sqrt(variance + eps); });
        var rms = x.map(function (v) { return v / Math.sqrt(second + eps); });
        document.getElementById('scale-value').value = a.toFixed(1);
        document.getElementById('shift-value').value = c.toFixed(1);
        var body = document.getElementById('norm-values');
        body.replaceChildren();
        [['Input', x], ['LayerNorm', ln], ['RMSNorm', rms]].forEach(function (pair) {
            var tr = document.createElement('tr'), th = document.createElement('th');
            th.scope = 'row'; th.textContent = pair[0]; tr.appendChild(th);
            pair[1].forEach(function (v) {
                var td = document.createElement('td'); td.textContent = v.toFixed(3); tr.appendChild(td);
            });
            body.appendChild(tr);
        });
        document.getElementById('norm-message').textContent =
            'Mean = ' + mean.toFixed(3) + '; variance = ' + variance.toFixed(3) +
            '; second moment = ' + second.toFixed(3) + '. ' +
            (a === 0 ? 'A constant vector centers to zero under LN. RMSNorm can keep a nonzero constant.' :
             'Change only the shift: LN is unchanged, while RMSNorm generally changes.');
    }
    if (scale && shift) {
        scale.addEventListener('input', updateNorm); shift.addEventListener('input', updateNorm);
        document.getElementById('norm-reset').addEventListener('click', function () {
            scale.value = 1; shift.value = 0; updateNorm();
        });
        updateNorm();
    }
    var width = document.getElementById('budget-width'), mode = document.getElementById('budget-mode');
    function updateBudget() {
        var d = Number(width.value), denseF = 4 * d;
        var gatedF = mode.value === 'matched' ? 8 * d / 3 : denseF;
        var dense = 2 * d * denseF, gated = 3 * d * gatedF;
        document.getElementById('budget-result').textContent =
            'Dense: F = ' + denseF + ', ' + dense.toLocaleString('en-US') + ' matrix parameters. ' +
            'Gated: F = ' + gatedF + ', ' + gated.toLocaleString('en-US') + ' matrix parameters. ' +
            'Gated/dense ratio = ' + (gated / dense).toFixed(2) + '.';
    }
    if (width && mode) {
        width.addEventListener('change', updateBudget); mode.addEventListener('change', updateBudget);
        updateBudget();
    }
})();
