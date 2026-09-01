/* Shared behaviour for CSCI 601.471/671 reading handouts.
   No dependencies. Everything degrades gracefully if JS is off. */
(function () {
    'use strict';

    /* ---- 1. Build the sidebar TOC from the document's own headings ---- */
    function buildToc() {
        var toc = document.querySelector('.hd-toc');
        var body = document.querySelector('.hd-body');
        if (!toc || !body) return [];

        var heads = body.querySelectorAll('h2[id], h3[id]');
        if (!heads.length) return [];

        var root = document.createElement('ol');
        var sub = null;
        heads.forEach(function (h) {
            var li = document.createElement('li');
            var a = document.createElement('a');
            a.href = '#' + h.id;
            // Strip the decorative section number from the TOC label.
            var num = h.querySelector('.hd-secnum');
            a.textContent = (num ? h.textContent.replace(num.textContent, '') : h.textContent).trim();
            li.appendChild(a);

            if (h.tagName === 'H2') {
                root.appendChild(li);
                sub = null;
            } else {
                if (!sub) {
                    sub = document.createElement('ol');
                    (root.lastElementChild || root).appendChild(sub);
                }
                sub.appendChild(li);
            }
        });
        toc.appendChild(root);
        return Array.prototype.slice.call(heads);
    }

    /* ---- 2. Highlight the heading currently in view ---- */
    function trackToc(heads) {
        if (!heads.length || !('IntersectionObserver' in window)) return;
        var links = {};
        document.querySelectorAll('.hd-toc a').forEach(function (a) {
            links[a.getAttribute('href').slice(1)] = a;
        });

        var visible = new Set();
        function repaint() {
            var current = null;
            for (var i = 0; i < heads.length; i++) {
                if (visible.has(heads[i].id)) { current = heads[i].id; break; }
            }
            if (!current) return;
            for (var id in links) links[id].classList.toggle('active', id === current);
        }

        var io = new IntersectionObserver(function (entries) {
            entries.forEach(function (e) {
                if (e.isIntersecting) visible.add(e.target.id);
                else visible.delete(e.target.id);
            });
            repaint();
        }, { rootMargin: '0px 0px -70% 0px' });

        heads.forEach(function (h) { io.observe(h); });
    }

    /* ---- 3. "I'm comfortable with this" checkboxes, persisted per handout ---- */
    function wireChecklist() {
        var boxes = document.querySelectorAll('.hd-check[data-key]');
        if (!boxes.length) return;
        var ns = 'ssm-handout:' + (document.body.dataset.handout || location.pathname) + ':';

        boxes.forEach(function (wrap) {
            var key = ns + wrap.dataset.key;
            var input = wrap.querySelector('input[type=checkbox]');
            if (!input) return;

            var stored = null;
            try { stored = localStorage.getItem(key); } catch (e) { /* private mode */ }
            input.checked = stored === '1';
            wrap.classList.toggle('done', input.checked);

            input.addEventListener('change', function () {
                wrap.classList.toggle('done', input.checked);
                try { localStorage.setItem(key, input.checked ? '1' : '0'); } catch (e) { /* ignore */ }
            });
        });
    }

    /* ---- 4. Expand every answer before printing, restore afterwards ---- */
    function wirePrint() {
        var wasOpen = [];
        window.addEventListener('beforeprint', function () {
            wasOpen = [];
            document.querySelectorAll('details').forEach(function (d) {
                wasOpen.push(d.open);
                d.open = true;
            });
        });
        window.addEventListener('afterprint', function () {
            document.querySelectorAll('details').forEach(function (d, i) {
                if (i < wasOpen.length) d.open = wasOpen[i];
            });
        });
    }

    /* ---- 5. Expand / collapse all answers in a section ---- */
    function wireBulkToggles() {
        document.querySelectorAll('[data-toggle-answers]').forEach(function (btn) {
            btn.addEventListener('click', function () {
                var scope = document.querySelector(btn.dataset.toggleAnswers) || document;
                var items = scope.querySelectorAll('details.ans');
                var open = btn.dataset.state !== 'open';
                items.forEach(function (d) { d.open = open; });
                btn.dataset.state = open ? 'open' : 'closed';
                btn.textContent = open ? 'Hide all answers' : 'Show all answers';
            });
        });
    }

    /* ---- 6. Math + code ---- */
    function renderMath() {
        if (typeof renderMathInElement !== 'function') return;
        renderMathInElement(document.querySelector('.hd-body') || document.body, {
            delimiters: [
                { left: '$$', right: '$$', display: true },
                { left: '\\[', right: '\\]', display: true },
                { left: '$', right: '$', display: false },
                { left: '\\(', right: '\\)', display: false }
            ],
            ignoredTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code', 'option'],
            throwOnError: false
        });
    }

    function highlightCode() {
        if (typeof hljs === 'undefined') return;
        document.querySelectorAll('pre code').forEach(function (block) {
            try { hljs.highlightElement(block); } catch (e) { /* ignore */ }
        });
    }

    function init() {
        trackToc(buildToc());
        wireChecklist();
        wirePrint();
        wireBulkToggles();
        renderMath();
        highlightCode();
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
