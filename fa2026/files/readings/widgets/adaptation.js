/* Offline accounting model, not a GPU-memory profiler. */
(function () {
    'use strict';
    var root = document.getElementById('w-adaptation-memory');
    if (!root) return;
    var rank = document.getElementById('adapt-rank');
    var target = document.getElementById('adapt-target');
    var headAct = document.getElementById('adapt-head-act');
    var loraAct = document.getElementById('adapt-lora-act');
    var result = document.getElementById('adapt-result');
    var gib = Math.pow(2, 30);
    function fmt(x) { return x.toLocaleString('en-US', {maximumFractionDigits: 3}); }
    function update() {
        var r = Number(rank.value), ha = Number(headAct.value), la = Number(loraAct.value);
        if ([headAct, loraAct].some(function (el) {
            return el.value.trim() === '' || !Number.isFinite(Number(el.value)) || Number(el.value) < 0 || Number(el.value) > 64;
        })) {
            result.textContent = 'Enter activation-memory budgets between 0 and 64 GiB. These are hypothetical inputs, not measured predictions.';
            return;
        }
        var d = 4096, layers = 32, v = 32000;
        var perLayer = target.value === 'qv' ? 4 * d : 8 * d;
        if (target.value === 'all') perLayer += 3 * (d + 11008);
        var n = layers * r * perLayer;
        var h = d * v;
        var base = 7e9 * 2 / gib;
        var hs = 12 * h / gib;
        var ls = 14 * n / gib;
        var ht = base + hs + ha, lt = base + ls + la;
        var scale = Math.max(hs + ha, ls + la, 0.001);
        function bar(state, act) {
            return '<div class="memory-bar" aria-hidden="true"><span class="memory-state" style="width:' + (100 * state / scale) + '%"></span><span class="memory-act" style="width:' + (100 * act / scale) + '%"></span></div>';
        }
        result.innerHTML = '<p><strong>Trainable parameters:</strong> head ' + fmt(h / 1e6) + 'M; LoRA ' + fmt(n / 1e6) + 'M.</p>' +
            '<div class="tbl-scroll"><table class="tbl"><caption>Accounted memory (GiB); omissions listed below</caption><thead><tr><th>Component</th><th>Head</th><th>LoRA</th></tr></thead><tbody>' +
            '<tr><td>Shared base weights</td><td>' + fmt(base) + '</td><td>' + fmt(base) + '</td></tr>' +
            '<tr><td>Extra parameter-related memory</td><td>' + fmt(hs) + '</td><td>' + fmt(ls) + '</td></tr>' +
            '<tr><td>Assumed activations</td><td>' + fmt(ha) + '</td><td>' + fmt(la) + '</td></tr>' +
            '<tr><th>Accounted total</th><td>' + fmt(ht) + '</td><td>' + fmt(lt) + '</td></tr></tbody></table></div>' +
            '<p class="adapt-legend"><span class="memory-state"></span>Extra parameter-related memory <span class="memory-act"></span>Assumed activations (shared base omitted from bars)</p>' +
            '<p>Head</p>' + bar(hs, ha) + '<p>LoRA</p>' + bar(ls, la) +
            '<p><strong>' + (Math.abs(ht - lt) < 1e-9 ? 'Equal accounted memory.' : (ht < lt ? 'Head tuning' : 'LoRA') + ' uses ' + fmt(Math.abs(ht - lt)) + ' GiB less in this accounting model.') + '</strong> ' +
            'LoRA is lower when its activation budget minus the head activation budget is less than ' + fmt(hs - ls) + ' GiB.</p>';
    }
    [rank, target, headAct, loraAct].forEach(function (el) { el.addEventListener('input', update); el.addEventListener('change', update); });
    update();
}());

/* Schematic gradient trace; outcomes match the deterministic CPU lab. */
(function () {
    'use strict';
    var root = document.getElementById('w-adaptation-flow');
    if (!root) return;
    var mode = document.getElementById('flow-mode');
    var button = document.getElementById('flow-trace');
    var graph = document.getElementById('flow-graph');
    var result = document.getElementById('flow-result');
    var traced = false;
    function update() {
        var head = mode.value === 'head';
        var broken = mode.value === 'broken';
        var names = [head ? 'Early projection' : 'Early projection + adapter', 'Frozen bridge', head ? 'Late projection' : 'Late projection + adapter', 'Output head', 'Loss'];
        var training = [!head, false, !head, head, false];
        var reached = [!head && !broken, !head && !broken, !head, true, true];
        graph.innerHTML = '<p class="flow-direction">Forward: left to right (top to bottom on phones). Backward: reverse direction.</p><ol class="flow-stages">' + names.map(function (name, i) {
            var cut = traced && broken && i === 1;
            var state = cut ? 'cut' : traced && reached[i] ? 'reached' : '';
            var status = i === 4 ? 'Scalar objective' : training[i] ? 'Trainable parameters' : 'Frozen parameters';
            var back = !traced ? 'Backward not traced' : cut ? 'STOP: detached output' : reached[i] ? (i === 4 ? 'Backward starts here' : 'Backward crosses this stage') : 'No backward through this stage';
            return '<li class="flow-stage ' + state + '"><strong>' + name + '</strong><span>' + status + '</span><span class="flow-back">' + back + '</span></li>';
        }).join('') + '</ol>';
        button.textContent = traced ? 'Reset to forward' : 'Trace backward';
        if (!traced) {
            result.textContent = 'Predict: which trainable parameters get a gradient? Does the frozen bridge need an input gradient?';
            return;
        }
        var description = head ? 'The head learns from frozen features. Its backward pass needs those features, but no backbone backward pass.' : broken ? 'The late adapter still learns, so loss can decrease. Detaching the bridge disconnects the early adapter: its gradients are missing, not zero tensors.' : 'Backward crosses the frozen head and bridge to reach both adapters. Frozen weights get no parameter gradients, but their operations transmit input gradients.';
        function row(name, value) { return '<tr><td>' + name + '</td><td>' + value + '</td></tr>'; }
        result.innerHTML = '<p><strong>' + description + '</strong></p><div class="tbl-scroll"><table class="tbl"><caption>First-backward parameter gradients in the CPU lab</caption><thead><tr><th>Parameters</th><th>Gradient state</th></tr></thead><tbody>' +
            row('Early adapter A / B', head ? 'Not present' : broken ? 'None / None (disconnected)' : 'Zero tensor / nonzero tensor') +
            row('Bridge weight', 'None (frozen)') +
            row('Late adapter A / B', head ? 'Not present' : 'Zero tensor / nonzero tensor') +
            row('Head weight', head ? 'Nonzero tensor' : 'None (frozen)') +
            '</tbody></table></div><p>A starts with a zero task gradient because B starts at zero (§3.2). A zero tensor is different from a missing gradient.</p>';
    }
    mode.addEventListener('change', function () { traced = false; update(); });
    button.addEventListener('click', function () { traced = !traced; update(); });
    update();
}());
