(function () {
    'use strict';
    var root = document.getElementById('attention-explorer');
    if (!root) return;
    var initial = [Math.log(2), 0, 0], values = [[2, 0], [0, 2], [4, 4]];
    var inputs = [], bars = [], labels = [];
    var query = root.querySelector('.query-position');
    var controls = root.querySelector('.score-controls');
    var holder = root.querySelector('.attention-bars');
    initial.forEach(function (score, i) {
        var label = document.createElement('label');
        label.textContent = 'Score for key ' + i;
        var input = document.createElement('input');
        input.type = 'number'; input.step = '0.1'; input.min = '-20'; input.max = '20';
        input.value = String(score); label.appendChild(input); controls.appendChild(label);
        inputs.push(input);
        var row = document.createElement('div'); row.className = 'attention-bar-row';
        var name = document.createElement('span'); name.textContent = 'Key ' + i;
        var track = document.createElement('div'); track.className = 'attention-track';
        var fill = document.createElement('div'); fill.className = 'attention-fill';
        var number = document.createElement('span');
        track.appendChild(fill); row.appendChild(name); row.appendChild(track); row.appendChild(number);
        holder.appendChild(row); bars.push(fill); labels.push(number);
        input.addEventListener('input', update);
    });
    function update() {
        var last = query ? Number(query.value) : 2;
        var scores = inputs.map(function (input) { return Number(input.value); });
        if (inputs.some(function (input) { return input.value === '' || !input.validity.valid; }) || scores.some(function (v) { return !isFinite(v); })) {
            root.querySelector('.attention-result').textContent = 'Enter three finite scores between -20 and 20.';
            return;
        }
        var max = Math.max.apply(null, scores.slice(0, last + 1));
        var weights = scores.map(function (s, i) { return i <= last ? Math.exp(s-max) : 0; });
        var total = weights.reduce(function (a,b) {return a+b;},0);
        weights = weights.map(function (w) { return w/total; });
        var out = [0, 0];
        weights.forEach(function (w,i) { out[0] += w*values[i][0]; out[1] += w*values[i][1]; bars[i].style.width = (100*w)+'%'; labels[i].textContent = i > last ? 'masked' : w.toFixed(4); });
        root.querySelector('.attention-result').textContent = 'Weights [' + weights.map(function(w){return w.toFixed(4);}).join(', ') + ']; sum 1.0000; output [' + out.map(function(v){return v.toFixed(4);}).join(', ') + '].';
    }
    if (query) query.addEventListener('change', update);
    root.querySelector('.attention-reset').addEventListener('click',function(){ inputs.forEach(function(input,i){input.value=String(initial[i]);}); if(query)query.value='1'; update(); });
    update();
})();
