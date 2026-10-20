/* Synthetic corpus audit for Handout 15. No network or external dependencies. */
(function () {
    'use strict';
    var mount = document.getElementById('w-data-audit');
    if (!mount) return;
    var docs = [
        ['a', 'Water freezes at zero degrees Celsius.'],
        ['b', 'Water freezes at zero degrees Celsius.'],
        ['c', 'water   freezes at zero degrees celsius.'],
        ['d', 'Menu Home Login Subscribe'],
        ['e', 'def square(x): return x*x'],
        ['f', 'Plants use sunlight to make sugars.'],
        ['g', 'Water boils at one hundred degrees Celsius.'],
        ['h', 'PLANTS use sunlight to make sugars.']
    ];
    var minimum = document.getElementById('data-minimum');
    var mode = document.getElementById('data-mode');
    var rows = document.getElementById('data-rows');
    var result = document.getElementById('data-result');
    function update() {
        var seen = Object.create(null);
        var kept = 0, words = 0, short = 0, duplicates = 0;
        rows.textContent = '';
        docs.forEach(function (doc) {
            var count = doc[1].trim().split(/\s+/).length;
            var key = mode.value === 'normalized' ? doc[1].toLowerCase().trim().replace(/\s+/g, ' ') : doc[1];
            var decision;
            if (count < Number(minimum.value)) {
                short += 1;
                decision = 'Drop: short';
            } else if (mode.value !== 'none' && seen[key]) {
                duplicates += 1;
                decision = 'Drop: duplicate';
            } else {
                kept += 1;
                words += count;
                seen[key] = true;
                decision = 'Keep';
            }
            var row = document.createElement('tr');
            [doc[0], doc[1], String(count), decision].forEach(function (value) {
                var cell = document.createElement('td');
                cell.textContent = value;
                row.appendChild(cell);
            });
            rows.appendChild(row);
        });
        result.textContent = 'Kept ' + kept + ' of 8 documents; ' + words + ' of 45 whitespace words. Dropped ' + short + ' short documents and ' + duplicates + ' duplicates after filtering.';
    }
    minimum.addEventListener('change', update);
    mode.addEventListener('change', update);
    update();
})();
