/* Next-token explorer: maximum-likelihood n-gram models estimated in the
   browser from a small embedded corpus, with a selectable order.

   The point is not model quality. It is to make three things visible at once:
   p(x_t | x_<t) as an actual distribution, the chain rule accumulating in log
   space, and the trade-off between a higher order (sharper predictions) and
   sparsity (contexts you have never seen). No dependencies. */
(function () {
    'use strict';

    var mount = document.getElementById('w-next-token');
    if (!mount) return;

    /* A deliberately tiny, repetitive corpus so every count is checkable by
       hand. Students can read it and predict what the model will say. */
    var CORPUS = [
        "the cat sat on the mat",
        "the cat sat on the floor",
        "the cat slept on the mat",
        "the dog sat on the rug",
        "the dog barked at the cat",
        "the dog chased the cat",
        "a cat chased a mouse",
        "a mouse ran under the floor",
        "the mouse ran under the rug",
        "the student read the paper",
        "the student read the book",
        "the student wrote a paper about language",
        "a language model predicts the next word",
        "a language model predicts the next token",
        "the model predicts the next word",
        "the model reads the paper about language",
        "language models read a lot of text",
        "models read text and predict text"
    ];

    var BOS = '<s>', EOS = '</s>';

    /* ---- estimate counts for orders 1..3 ----
       ctx[n] maps an (n-1)-word context string to a Map of next-word counts.
       The order-1 context is the empty string, i.e. plain unigram counts. */
    var ctx = { 1: new Map(), 2: new Map(), 3: new Map() };
    var vocab = new Set();
    var uniTotal = 0;

    function bump(map, key, w) {
        var inner = map.get(key);
        if (!inner) { inner = new Map(); map.set(key, inner); }
        inner.set(w, (inner.get(w) || 0) + 1);
    }

    CORPUS.forEach(function (line) {
        var toks = [BOS, BOS].concat(line.split(/\s+/)).concat([EOS]);
        for (var i = 2; i < toks.length; i++) {
            vocab.add(toks[i]);
            uniTotal++;
            bump(ctx[1], '', toks[i]);
            bump(ctx[2], toks[i - 1], toks[i]);
            bump(ctx[3], toks[i - 2] + ' ' + toks[i - 1], toks[i]);
        }
    });

    function contextKey(history, n) {
        if (n === 1) return '';
        var h = [BOS, BOS].concat(history);
        return h.slice(h.length - (n - 1)).join(' ');
    }

    function normalize(counts) {
        var total = 0, out = [];
        counts.forEach(function (c) { total += c; });
        counts.forEach(function (c, w) {
            out.push({ token: w, p: c / total, count: c, total: total });
        });
        out.sort(function (a, b) { return b.p - a.p || (a.token < b.token ? -1 : 1); });
        return out;
    }

    /* mode is the order: 1, 2 or 3. A fixed-order model has no estimate at all
       when its context was never observed -- that is the point. */
    function predict(history, mode) {
        var key = contextKey(history, mode);
        var counts = ctx[mode].get(key);
        return {
            order: mode,
            key: key,
            dist: counts ? normalize(counts) : null   /* null = unseen context */
        };
    }

    /* Score the typed sequence with the chain rule, one factor per token.
       Returns null perplexity if any factor is zero -- which is the lesson. */
    function scoreSequence(tokens, mode) {
        var sumLog = 0, scored = 0, zeros = [];
        for (var i = 0; i < tokens.length; i++) {
            var res = predict(tokens.slice(0, i), mode);
            var p = 0;
            if (res.dist) {
                for (var j = 0; j < res.dist.length; j++) {
                    if (res.dist[j].token === tokens[i]) { p = res.dist[j].p; break; }
                }
            }
            if (p === 0) { zeros.push(tokens[i]); continue; }
            sumLog += Math.log(p);
            scored++;
        }
        return {
            sumLog: sumLog,
            scored: scored,
            zeros: zeros,
            ppl: zeros.length ? null : (scored ? Math.exp(-sumLog / scored) : null)
        };
    }

    /* ---- DOM ---- */
    mount.innerHTML =
        '<div class="w-row">' +
        '  <label for="wnt-input" style="flex:1 1 100%;margin-bottom:2px">' +
        '    Context so far (words from the corpus above)' +
        '  </label>' +
        '  <input type="text" id="wnt-input" value="the cat sat" ' +
        '         autocomplete="off" spellcheck="false" style="flex:1 1 320px">' +
        '  <button class="w-btn" id="wnt-clear" type="button">Clear</button>' +
        '</div>' +
        '<div class="w-row">' +
        '  <span class="w-stat-label" style="align-self:center">Order:</span>' +
        '  <button class="w-btn" data-mode="1" type="button">Unigram</button>' +
        '  <button class="w-btn" data-mode="2" type="button">Bigram</button>' +
        '  <button class="w-btn on" data-mode="3" type="button">Trigram</button>' +
        '</div>' +
        '<p class="w-note" id="wnt-order" style="margin:0 0 12px"></p>' +
        '<div class="w-bars" id="wnt-bars"></div>' +
        '<div class="w-stats">' +
        '  <div><div class="w-stat-label">Tokens scored</div><div class="w-stat-val" id="wnt-n">0</div></div>' +
        '  <div><div class="w-stat-label">&sum; log p</div><div class="w-stat-val" id="wnt-logp">0.00</div></div>' +
        '  <div><div class="w-stat-label">Avg. log p</div><div class="w-stat-val" id="wnt-avg">&mdash;</div></div>' +
        '  <div><div class="w-stat-label">Perplexity</div><div class="w-stat-val" id="wnt-ppl">&mdash;</div></div>' +
        '</div>' +
        '<p class="w-note" id="wnt-warn"></p>' +
        '<div class="w-row" style="margin:16px 0 0;padding-top:14px;border-top:1px solid #e2e6ec">' +
        '  <button class="w-btn" id="wnt-gen" type="button">Generate 14 words &rarr;</button>' +
        '  <span class="w-note" style="margin:0;align-self:center">samples forward from the context above</span>' +
        '</div>' +
        '<div id="wnt-sample"></div>' +
        '<div id="wnt-compare" style="margin-top:16px"></div>';

    var $input = document.getElementById('wnt-input');
    var $bars = document.getElementById('wnt-bars');
    var $order = document.getElementById('wnt-order');
    var $warn = document.getElementById('wnt-warn');
    var $n = document.getElementById('wnt-n');
    var $logp = document.getElementById('wnt-logp');
    var $avg = document.getElementById('wnt-avg');
    var $ppl = document.getElementById('wnt-ppl');
    var $cmp = document.getElementById('wnt-compare');
    var $sample = document.getElementById('wnt-sample');
    var mode = 3;

    function esc(s) {
        return String(s).replace(/[&<>"]/g, function (c) {
            return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c];
        });
    }

    function label(t) { return t === EOS ? '&lt;/s&gt;' : esc(t); }

    function describe(res) {
        var name = { 1: 'unigram', 2: 'bigram', 3: 'trigram' }[res.order];
        if (!res.dist) {
            return 'The ' + name + ' context <code>' + esc(res.key) + '</code> never appears in the ' +
                'corpus, so a fixed-order model has <b>no estimate at all</b> here. Every next word ' +
                'gets probability 0. This is sparsity, and it is the reason smoothing exists.';
        }
        return 'Using the <b>' + name + '</b> estimate' +
            (res.order > 1 ? ', context <code>' + esc(res.key) + '</code>' : '') +
            ', ' + res.dist[0].total + ' observation(s) &mdash; so each probability below is a count ' +
            'over ' + res.dist[0].total + '.';
    }

    /* Perplexity of the current sentence under every mode, side by side. */
    function renderCompare(tokens) {
        if (!tokens.length) { $cmp.innerHTML = ''; return; }
        var modes = [
            { m: 1, name: 'Unigram' },
            { m: 2, name: 'Bigram' },
            { m: 3, name: 'Trigram' }
        ];
        var rows = modes.map(function (o) {
            var s = scoreSequence(tokens, o.m);
            var ppl = s.ppl === null
                ? '<span style="color:#b02a55">&infin; (zero factor)</span>'
                : s.ppl.toFixed(2);
            return '<tr><td>' + o.name + '</td><td>' + s.scored + ' / ' + tokens.length +
                '</td><td>' + ppl + '</td></tr>';
        }).join('');
        $cmp.innerHTML =
            '<div class="w-stat-label" style="margin-bottom:6px">This sentence, under each model</div>' +
            '<div class="tbl-scroll"><table class="tbl" style="margin:0;font-size:13.5px">' +
            '<thead><tr><th>Model</th><th>Tokens scored</th><th>Perplexity</th></tr></thead>' +
            '<tbody>' + rows + '</tbody></table></div>';
    }

    function render() {
        var raw = $input.value.toLowerCase().trim();
        var tokens = raw.length ? raw.split(/\s+/) : [];

        var res = predict(tokens, mode);
        $order.innerHTML = describe(res);

        if (!res.dist) {
            $bars.innerHTML = '<p class="w-note" style="padding:8px 0">' +
                'No distribution: this context was never observed.</p>';
        } else {
            var shown = res.dist.slice(0, 8);
            var max = shown[0].p;
            $bars.innerHTML = shown.map(function (d) {
                return '<div class="w-bar-row">' +
                    '<div class="w-bar-tok">' + label(d.token) + '</div>' +
                    '<div class="w-bar-track"><div class="w-bar-fill" style="width:' +
                    (100 * d.p / max).toFixed(1) + '%"></div></div>' +
                    '<div class="w-bar-val">' + d.p.toFixed(3) + '</div>' +
                    '</div>';
            }).join('');
        }

        var s = scoreSequence(tokens, mode);
        $n.textContent = s.scored;
        $logp.textContent = s.scored ? s.sumLog.toFixed(2) : '0.00';
        $avg.textContent = s.scored ? (s.sumLog / s.scored).toFixed(3) : '—';
        $ppl.innerHTML = s.ppl === null
            ? (tokens.length ? '<span style="color:#b02a55">&infin;</span>' : '—')
            : s.ppl.toFixed(2);

        $warn.innerHTML = s.zeros.length
            ? 'Probability 0 for: ' +
              s.zeros.map(function (w) { return '<code>' + esc(w) + '</code>'; }).join(', ') +
              '. One zero factor sends the whole product to 0, so the log is $-\\infty$ and ' +
              'perplexity is undefined &mdash; no matter how well the model did on every other token.'
            : '';

        renderCompare(tokens);
    }

    /* Sample forward from the model, one token at a time, feeding each choice
       back in as context -- the recursive sampling loop from the lecture. */
    function generate(history, maxWords) {
        var out = [], h = history.slice();
        for (var i = 0; i < maxWords; i++) {
            var dist = predict(h, mode).dist;
            if (!dist || !dist.length) { out.push('\u2026 [no estimate]'); break; }
            var r = Math.random(), acc = 0, pick = dist[dist.length - 1].token;
            for (var j = 0; j < dist.length; j++) {
                acc += dist[j].p;
                if (r <= acc) { pick = dist[j].token; break; }
            }
            if (pick === EOS) { out.push('\u2038'); break; }
            out.push(pick);
            h.push(pick);
        }
        return out;
    }

    function renderSample() {
        var raw = $input.value.toLowerCase().trim();
        var history = raw.length ? raw.split(/\s+/) : [];
        var words = generate(history, 14);
        var name = { 1: 'unigram', 2: 'bigram', 3: 'trigram' }[mode];
        $sample.innerHTML =
            '<div class="w-pairs" style="margin-top:12px;padding:12px 14px;background:#f8fafd;' +
            'border-radius:5px;line-height:1.9">' +
            (history.length ? history.map(function (w) { return tok(w, 'ctx'); }).join('') : '') +
            words.map(function (w) { return tok(w, 'tgt'); }).join('') +
            '</div>' +
            '<p class="w-note">Sampled from the ' + name + ' model. Press again for another draw; ' +
            '\u2038 marks the end-of-sequence token.</p>';
    }

    function tok(text, cls) {
        return '<span class="tok ' + (cls || '') + '">' + esc(text) + '</span>';
    }

    document.getElementById('wnt-gen').addEventListener('click', renderSample);

    mount.querySelectorAll('[data-mode]').forEach(function (btn) {
        btn.addEventListener('click', function () {
            mode = Number(btn.dataset.mode);
            mount.querySelectorAll('[data-mode]').forEach(function (b) {
                b.classList.toggle('on', b === btn);
            });
            render();
            $sample.innerHTML = '';
        });
    });

    $input.addEventListener('input', function () { render(); $sample.innerHTML = ''; });
    document.getElementById('wnt-clear').addEventListener('click', function () {
        $input.value = '';
        render();
        $input.focus();
    });

    /* Let the surrounding prose print the corpus it was estimated from. */
    var dump = document.getElementById('wnt-corpus');
    if (dump) dump.textContent = CORPUS.join('\n');

    render();
})();
