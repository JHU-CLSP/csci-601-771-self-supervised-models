/* Offline arithmetic explorer. The static worked examples remain in print. */
(function () {
    'use strict';
    var root = document.getElementById('w-training-plan');
    if (!root) return;
    var names = ['d', 'b', 'k', 'p', 't'];
    var controls = names.map(function (name) { return document.getElementById('plan-' + name); });
    var result = document.getElementById('plan-result');
    function fmt(n) { return n.toLocaleString('en-US', { maximumFractionDigits: 0 }); }
    function row(label, value) { return '<dt>' + label + '</dt><dd>' + value + '</dd>'; }
    function update() {
        var values = controls.map(function (input) { return Number(input.value); });
        var valid = controls.every(function (input, i) {
            var v = values[i];
            return input.value.trim() !== '' && Number.isSafeInteger(v) && v >= Number(input.min) && v <= Number(input.max);
        });
        if (!valid) { result.textContent = 'Enter a positive whole number within the stated bounds for every field.'; return; }
        var d = values[0] * 1e6;
        var sequences = values[1] * values[2] * values[3];
        var tokens = sequences * values[4];
        if (!Number.isSafeInteger(tokens) || !Number.isSafeInteger(d)) {
            result.textContent = 'This combination is too large to count tokens exactly. Reduce the batch dimensions.';
            return;
        }
        var updates = Math.ceil(d / tokens);
        var finalTokens = d - (updates - 1) * tokens;
        result.innerHTML = '<dl>' + row('Sequences / full update', fmt(sequences)) +
            row('Tokens / full update', fmt(tokens)) + row('Optimizer updates', fmt(updates)) +
            row('Tokens in final update', fmt(finalTokens)) + '</dl>';
    }
    controls.forEach(function (input) { input.addEventListener('input', update); });
    update();
}());
