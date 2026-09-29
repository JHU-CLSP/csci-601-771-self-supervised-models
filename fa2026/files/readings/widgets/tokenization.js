/* Handout 9: deterministic, local character-BPE demonstrations. No network. */
(function () {
    'use strict';
    var corpus = ['jhu', ' jhu', ' jhu', ' hopkins', ' hop', ' hops', ' hops'];
    function key(pair) { return JSON.stringify(pair); }
    function countPairs(chunks) {
        var stats = new Map();
        chunks.forEach(function (chunk) {
            for (var i = 0; i + 1 < chunk.length; i++) {
                var pair = [chunk[i], chunk[i + 1]], k = key(pair);
                if (!stats.has(k)) stats.set(k, { pair: pair, count: 0 });
                stats.get(k).count++;
            }
        });
        return Array.from(stats.values());
    }
    function mergeChunk(chunk, pair) {
        var out = [];
        for (var i = 0; i < chunk.length;) {
            if (i + 1 < chunk.length && chunk[i] === pair[0] && chunk[i + 1] === pair[1]) {
                out.push(pair[0] + pair[1]); i += 2;
            } else { out.push(chunk[i]); i++; }
        }
        return out;
    }
    function nextRule(tokens, rules) {
        var present = new Set(countPairs([tokens]).map(function (s) { return key(s.pair); }));
        for (var i = 0; i < rules.length; i++) if (present.has(key(rules[i]))) return i;
        return -1;
    }
    var examples = [
        { title: 'lowered: the slide rules', text: 'lowered', rules: [['l','o'],['lo','w'],['e','r'],['low','er']] },
        { title: 'abcd: longest-match uses a different path', text: 'abcd', rules: [['b','c'],['a','b'],['c','d']] },
        { title: 'abcbcbc: rank beats local frequency', text: 'abcbcbc', rules: [['a','b'],['b','c']] }
    ];
    // Expose the pure algorithm only for local numerical validation under Node.
    if (typeof module !== 'undefined' && module.exports) {
        module.exports = { corpus: corpus, countPairs: countPairs, mergeChunk: mergeChunk, nextRule: nextRule, examples: examples };
    }
    if (typeof document === 'undefined') return;
    function el(tag, text, cls) {
        var node = document.createElement(tag);
        if (text !== undefined) node.textContent = text;
        if (cls) node.className = cls;
        return node;
    }
    function visible(text) { return text.replace(/ /g, '·'); }
    function describe(pair) { return visible(pair[0]) + ' + ' + visible(pair[1]) + ' → ' + visible(pair.join('')); }
    function button(text, fn) {
        var b = el('button', text, 'w-btn'); b.type = 'button'; b.addEventListener('click', fn); return b;
    }
    function showChunks(host, chunks) {
        host.replaceChildren();
        chunks.forEach(function (chunk) {
            var group = el('span', undefined, 'bpe-chunk');
            group.setAttribute('aria-label', 'Chunk: ' + chunk.map(visible).join(' | '));
            chunk.forEach(function (token) { group.appendChild(el('span', visible(token), 'bpe-piece')); });
            host.appendChild(group);
        });
    }
    function mountTraining() {
        var host = document.getElementById('bpe-training-ui'); if (!host) return;
        var history;
        var note = el('p', 'Character BPE; · is one space. Chunk walls stay fixed.');
        var status = el('p', '', 'bpe-status'); status.setAttribute('aria-live', 'polite');
        var chunks = el('div', undefined, 'bpe-chunks');
        var label = el('label', 'Next maximum-frequency pair'); label.htmlFor = 'bpe-choice';
        var choice = el('select'); choice.id = 'bpe-choice';
        var controls = el('div', undefined, 'w-row');
        var apply = button('Apply selected merge', function () {
            var state = history[history.length - 1], pair = JSON.parse(choice.value);
            history.push({ chunks: state.chunks.map(function (c) { return mergeChunk(c, pair); }), rules: state.rules.concat([pair]) });
            render();
        });
        var back = button('Undo', function () { if (history.length > 1) history.pop(); render(); });
        var reset = button('Reset', function () { init(); render(); });
        controls.append(apply, back, reset);
        var tableWrap = el('div', undefined, 'tbl-scroll');
        var table = el('table', undefined, 'tbl');
        var caption = el('caption', 'Current within-chunk pair counts');
        var head = el('thead'); var hr = el('tr');
        ['Pair', 'Count'].forEach(function (s) { var th = el('th', s); th.scope = 'col'; hr.appendChild(th); });
        head.appendChild(hr); var body = el('tbody'); table.append(caption, head, body); tableWrap.appendChild(table);
        var learned = el('p', '', 'bpe-rules');
        host.append(note, status, chunks, label, choice, controls, tableWrap, learned);
        function init() { history = [{ chunks: corpus.map(function (c) { return Array.from(c); }), rules: [] }]; }
        function render() {
            var state = history[history.length - 1];
            var stats = countPairs(state.chunks).sort(function (a,b) { return b.count - a.count; });
            var n = state.chunks.reduce(function (s,c) { return s + c.length; }, 0);
            status.textContent = state.rules.length + ' merges · ' + n + ' corpus tokens · ' + (10 + state.rules.length) + ' vocabulary entries';
            showChunks(chunks, state.chunks); choice.replaceChildren(); body.replaceChildren();
            stats.forEach(function (s) {
                var row = el('tr'); row.append(el('td', visible(s.pair[0]) + ' + ' + visible(s.pair[1])), el('td', String(s.count))); body.appendChild(row);
                if (s.count === stats[0].count) { var opt = el('option', describe(s.pair) + ' (count ' + s.count + ')'); opt.value = key(s.pair); choice.appendChild(opt); }
            });
            apply.disabled = !stats.length; choice.disabled = !stats.length; back.disabled = history.length === 1;
            if (!stats.length) status.textContent += ' · no adjacent pairs remain';
            learned.textContent = state.rules.length ? 'Learned order: ' + state.rules.map(function (p,i) { return (i+1) + '. ' + describe(p); }).join('; ') : 'No learned merges yet. Three pairs tie at count 4.';
        }
        init(); render();
    }
    function mountInference() {
        var host = document.getElementById('bpe-inference-ui'); if (!host) return;
        var label = el('label', 'Frozen merge table'); label.htmlFor = 'bpe-example';
        var select = el('select'); select.id = 'bpe-example';
        examples.forEach(function (x,i) { var o = el('option', x.title); o.value = String(i); select.appendChild(o); });
        var rules = el('ol', undefined, 'bpe-rule-list');
        var stateView = el('div', undefined, 'bpe-chunks');
        var status = el('p', '', 'bpe-status'); status.setAttribute('aria-live', 'polite');
        var controls = el('div', undefined, 'w-row');
        var trace = el('pre', '', 'bpe-trace'); trace.setAttribute('aria-label', 'Encoding trace');
        var current, tokens, lines, steps;
        var next = button('Apply next ranked merge', function () {
            var index = nextRule(tokens, current.rules); if (index < 0) return;
            tokens = mergeChunk(tokens, current.rules[index]); steps++;
            lines.push('Rank ' + (index+1) + ': [' + tokens.join(' | ') + ']'); render();
        });
        var reset = button('Reset', function () { init(); }); controls.append(next, reset);
        host.append(label, select, rules, stateView, status, controls, trace);
        function init() {
            current = examples[Number(select.value)]; tokens = Array.from(current.text); steps = 0;
            lines = ['Start: [' + tokens.join(' | ') + ']']; rules.replaceChildren();
            current.rules.forEach(function (p) { rules.appendChild(el('li', describe(p))); }); render();
        }
        function render() {
            var index = nextRule(tokens, current.rules); showChunks(stateView, [tokens]);
            status.textContent = index < 0 ? 'Done: no learned rule applies. ' + tokens.length + ' tokens.' : 'After ' + steps + ' steps: next is rank ' + (index+1) + ', ' + describe(current.rules[index]) + '.';
            Array.from(rules.children).forEach(function (li,i) { li.classList.toggle('next-rule', i === index); });
            next.disabled = index < 0; trace.textContent = lines.join('\n');
        }
        select.addEventListener('change', init); init();
    }
    mountTraining(); mountInference();
})();
