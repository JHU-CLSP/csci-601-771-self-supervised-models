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
