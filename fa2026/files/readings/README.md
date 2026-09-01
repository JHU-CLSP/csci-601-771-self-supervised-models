# Reading handouts

One self-contained HTML handout per session, linked from the "Additional Reading" column of the
schedule in `fa2026/index.html`.

## Current handouts

| # | File | Session |
|---|------|---------|
| 1 | `01.foundations.html` | Foundations & prerequisites (math/CS review, what self-supervision is) |
| 2 | `02.language-modeling.html` | Language modeling: formal setup, scoring/generation, n-grams, sparsity, smoothing, perplexity |
| 3 | `03.neural-nets.html` | From counting to learning: fixed-window neural LM, what a neural net is, history, batched algebra, gradient descent |
| 5 | `05.backprop-in-practice.html` | Backprop as chain rule on a graph, reverse mode, autograd, the LM loss in PyTorch |

(No Handout #4 yet — Session 4 is analytical backprop, currently covered only by the assigned
3Blue1Brown videos.)

Widgets: `ssl-objective.js` (Handout #1), `next-token.js` (Handout #2 — order selector, per-model
perplexity table, sampler), `gradient-descent.js` (Handout #3 — learning-rate explorer).

Backoff is deliberately **not** covered: the course skips it, so `next-token.js` offers only fixed
orders (unigram/bigram/trigram) and Handout #2 §4 covers add-$k$ smoothing and interpolation only.

## `471-671-quiz-samples-public/`

Past quizzes and homeworks, with solutions. Source material for handout exercises — several are
ported into Handout #1 §1.3/§1.5/§6 and Handout #2 §1.2/§5.2/§5.4, marked with an `ex-src`
provenance label. Not linked from the schedule.

Three arithmetic errors found in these while porting (handouts use corrected values):
- sp2025 quiz1 Q3.3: `H(s2)` is 3.902 / ppl 14.95, not the printed 3.24 / 9.447 (and the second line
  is mislabelled `H(s1)`).
- sp2025 quiz1 Q3.4: solution says s1 wins "because it contains unobserved patterns" — should be
  *observed*.
- sp2024 quiz1 Q2.4: `H` uses `log2 2/6` while the same page's table and `P(s1)` both use `1/6`.
  With 1/6, H = 1.581 and ppl = 2.99, not 1.33 / 2.51.

PyTorch snippets live in Handout #1 §1.5, Handout #3 §4/§5, and Handout #5 §3/§4.
Handout #2 §3.2 is pure-Python n-gram counting (deliberately not PyTorch).
Every `assert` in them has been run against torch 2.12 — keep it that way when editing.

**Check every numeric claim against the running widget.** A probability, a perplexity, a divergence
threshold, a status message — several of these were wrong on the first pass in both handouts, and
only a browser run caught them.

## Adding a session

1. `cp _template.html NN.slug.html` (e.g. `02.language-modeling.html`) and fill in the placeholders
   marked `TITLE`, `Session N`, `NN.slug`, `TOPIC`.
2. Write the body. Give every `<h2>`/`<h3>` an `id` — the sidebar TOC is generated from them by
   `handout.js`, so there is no TOC to maintain by hand.
3. Add a link in the Session-N row of `../../index.html`.

## Conventions

- **Math**: KaTeX, `$…$` inline and `$$…$$` display. Write `<` as `&lt;` inside math
  (`x_{&lt;t}`) — otherwise the browser parses `<t}` as a tag and silently eats the rest of the
  document.
- **Code**: `<pre><code class="language-python">`, highlighted by highlight.js.
- **Answers**: `<details class="ans"><summary>…</summary><div class="ans-body">…</div></details>`.
  A `<button data-toggle-answers="#selector">` expands or collapses all of them within a scope, and
  everything is force-expanded before printing.
- **Checkboxes**: `<label class="hd-check" data-key="unique">` persists to `localStorage`, namespaced
  by the `data-handout` attribute on `<body>`.
- **Widgets**: one file per widget in `widgets/`, mounting into a `<div id="w-…">`. Plain ES5-ish
  JS, no build step, no framework.

## No network dependencies

KaTeX and highlight.js are vendored in `vendor/`. Handouts must work from `file://` and offline —
do not add a CDN `<script>` or `<link>`.

## `CS_601_471_671_spring2025___homework.pdf`

The old graded Homework 1 (background review + an IMDB classifier). Handout #1 ports its §1–§4 and
part of §5.1 as ungraded self-check exercises. **Source material only — not linked from the
schedule and not intended for students**, since it ships full solutions. Its §5.2 material
(embeddings, `DataLoader`, `nn.Module`, the IMDB classifier) is still waiting to be folded into a
later handout.

Because this directory is served publicly, an unlinked PDF here is still reachable by URL. Delete it
before committing if it should not be published.

Three answers in that PDF are wrong; the corrected values are in Handout #1 §1.1a and §1.2b:
`α(x+y) = [6,6,12]` (printed `16`), `Var[X] = 3.96 − 3.24` (printed `6.552 − 1.8²`), and
`E[1/(2+X)] = 0.2792` (printed `0.355` — the sum dropped its `k=0` term).
