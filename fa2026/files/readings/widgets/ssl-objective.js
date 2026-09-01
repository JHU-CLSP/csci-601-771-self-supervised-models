/* Objective constructor: shows how many (input, target) training examples a
   single unlabeled sentence yields under two self-supervised objectives.
   The claim from the slides -- "labels for free from unlabeled data" -- becomes
   a countable list. No dependencies. */
(function () {
    'use strict';

    var mount = document.getElementById('w-ssl-objective');
    if (!mount) return;

    var DEFAULT = 'the treaty of paris formally ended the seven years war';

    mount.innerHTML =
        '<div class="w-row">' +
        '  <label for="wso-input" style="flex:1 1 100%;margin-bottom:2px">' +
        '    Raw, unlabeled text &mdash; nobody annotated this' +
        '  </label>' +
        '  <input type="text" id="wso-input" autocomplete="off" spellcheck="false" ' +
        '         style="flex:1 1 100%">' +
        '</div>' +
        '<div class="w-row">' +
        '  <button class="w-btn on" id="wso-causal" type="button">Causal LM (next token)</button>' +
        '  <button class="w-btn" id="wso-masked" type="button">Masked LM (fill the blank)</button>' +
        '  <span style="flex:1"></span>' +
        '  <span class="w-stat-label">Examples produced:</span>' +
        '  <span class="w-stat-val" id="wso-count" style="font-size:16px">0</span>' +
        '</div>' +
        '<p class="w-note" id="wso-explain" style="margin:0 0 12px"></p>' +
        '<div class="w-pairs" id="wso-pairs"></div>';

    var $input = document.getElementById('wso-input');
    var $pairs = document.getElementById('wso-pairs');
    var $count = document.getElementById('wso-count');
    var $explain = document.getElementById('wso-explain');
    var $causal = document.getElementById('wso-causal');
    var $masked = document.getElementById('wso-masked');
    var mode = 'causal';

    $input.value = DEFAULT;

    function esc(s) {
        return String(s).replace(/[&<>"]/g, function (c) {
            return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c];
        });
    }

    function tok(text, cls) {
        return '<span class="tok ' + (cls || '') + '">' + esc(text) + '</span>';
    }

    /* A pseudo-random but deterministic mask choice, so the display is stable
       across re-renders of the same sentence (BERT masks ~15% of positions). */
    function maskPositions(n) {
        var k = Math.max(1, Math.round(0.15 * n));
        var picked = [], step = n / k;
        for (var i = 0; i < k; i++) picked.push(Math.min(n - 1, Math.floor(i * step + step / 2)));
        return picked;
    }

    function render() {
        var words = $input.value.trim().split(/\s+/).filter(Boolean);
        if (!words.length) {
            $pairs.innerHTML = '<p class="w-note">Type a sentence above.</p>';
            $count.textContent = '0';
            return;
        }

        var rows = [];

        if (mode === 'causal') {
            $explain.innerHTML =
                'One example per position: everything to the left is the input, the very ' +
                'next word is the target. A sentence of <b>' + words.length + '</b> words yields <b>' +
                words.length + '</b> supervised examples &mdash; and the whole sentence is processed in ' +
                'a single forward pass, because a causal mask stops each position from peeking right.';
            for (var t = 0; t < words.length; t++) {
                var ctx = t === 0
                    ? tok('<s>', 'pad')
                    : words.slice(0, t).map(function (w) { return tok(w, 'ctx'); }).join('');
                rows.push(
                    '<div class="w-pair"><span class="w-pair-idx">' + (t + 1) + '</span>' +
                    '<span>' + ctx + '</span>' +
                    '<span class="w-arrow">&rarr;</span>' +
                    '<span>' + tok(words[t], 'tgt') + '</span></div>'
                );
            }
            $count.textContent = String(words.length);
        } else {
            var pos = maskPositions(words.length);
            $explain.innerHTML =
                'Corrupt the input, then ask the model to restore it. Here <b>' + pos.length +
                '</b> of ' + words.length + ' positions (~15%) are replaced by <code>[MASK]</code>, and the ' +
                'targets are the words that were removed. The model sees context on <em>both</em> sides &mdash; ' +
                'great for representations, but it no longer defines a distribution you can sample text from.';
            var masked = words.map(function (w, i) {
                return pos.indexOf(i) >= 0 ? tok('[MASK]', 'msk') : tok(w, 'ctx');
            }).join('');
            rows.push(
                '<div class="w-pair"><span class="w-pair-idx">in</span><span>' + masked + '</span></div>'
            );
            pos.forEach(function (i, k) {
                rows.push(
                    '<div class="w-pair"><span class="w-pair-idx">' + (k + 1) + '</span>' +
                    '<span>position ' + (i + 1) + '</span>' +
                    '<span class="w-arrow">&rarr;</span>' +
                    '<span>' + tok(words[i], 'tgt') + '</span></div>'
                );
            });
            $count.textContent = String(pos.length);
        }

        $pairs.innerHTML = rows.join('');
    }

    function setMode(m) {
        mode = m;
        $causal.classList.toggle('on', m === 'causal');
        $masked.classList.toggle('on', m === 'masked');
        render();
    }

    $causal.addEventListener('click', function () { setMode('causal'); });
    $masked.addEventListener('click', function () { setMode('masked'); });
    $input.addEventListener('input', render);

    render();
})();
