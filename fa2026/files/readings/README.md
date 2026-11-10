# Reading handouts

One self-contained HTML handout per session, linked from the "Additional Reading" column of the
schedule in `fa2026/index.html`.

## Purpose and authoring approach

Handouts complement the lecture slides as additional reading. Use the assigned slide topics to
set scope, then develop additional in-scope insights, worked examples, exercises, and visualizations.
Keep enough explanation for the reading to stand on its own; avoid turning it into a slide transcript
or a running commentary on slide wording. Slide mappings are navigation aids, not the main structure.

Reuse useful figures from the course slides when they support an explanation or exercise. Preserve
source credits, identify the source slide, provide meaningful alt text, and explain what the reader
should notice. Pair borrowed figures with interpretation or a task rather than using them as decoration.

## Current handouts

| # | File | Session |
|---|------|---------|
| 1 | `01.foundations.html` | Foundations & prerequisites (math/CS review, what self-supervision is) |
| 2 | `02.language-modeling.html` | Language modeling: formal setup, scoring/generation, n-grams, sparsity, perplexity |
| 3 | `03.neural-nets.html` | From counting to learning (Tue Sept 8, slides 1–54): MLPs, expressivity, history, perceptrons |
| 4 | `04.training-neural-nets.html` | Losses, gradients, and optimization (Thu Sept 10, slides 55–89): losses, optimization, algebra, calculus, and readiness for backprop |
| 5 | `05.analytical-backprop.html` | Analytical backpropagation (Tue Sept 15, slides 90–109): layerwise derivatives, caching, and checked manual backprop |
| 6 | `06.backprop-in-practice.html` | Backprop in practice (Thu Sept 17, slides 111–156): computation graphs, reverse mode, autograd, PyTorch, reading `micrograd` |
| 7 | `07.training-practice.html` | Batching and training memory (Tue Sept 22, slides 165–182): execution modes, batching, broadcasting, memory/OOM, gradient accumulation; optional LM-loss application |
| 8 | `08.stable-training.html` | Training and tokenization (Thu Sept 24, feedforward slides 183–208 and tokenization slides 1–12): activations, gradient stability, residuals, initialization, normalization, validation, early stopping, dropout, tiny-data debugging, diagnostic capstone; word/character/subword tradeoffs, token IDs, whitespace, and number boundaries |
| 9 | `09.tokenization.html` | Tokenization and subwords (Tue Sept 29, slides 13–36): pipeline, information loss, BPE training/inference, byte coverage, modeling tradeoffs, core checks, and pinned minbpe lab. Blue bonus section covers slides 31, 35, and extensions 37–49. |
| 10 | `10.mlp-language-modeling.html` | Fixed-window neural LMs (Thu Oct 1, MLP slides 1–21): embedding lookup and repeated-ID gradients, ordered concatenation, next-token loss, joint training, parameter sharing and scaling, context limits, runnable PyTorch lab, and pinned PyTorch tutorial reading |
| 11 | `11.self-attention.html` | Self-attention and next-token prediction (Tue Oct 6, Transformer slides 1–51): numerical retrieval, scaling, batches/heads, fused QKV, blocks, positions, training/generation, leakage, attention lab, pinned nanoGPT reading |
| 12 | `12.transformer-masking-and-cost.html` | Masks, architecture, and cost (Thu Oct 8, Transformer slides 51–125): causal/padding/loss masks, encoder–decoder and cross-attention, FLOPs and IO, cache offsets/memory/cost, assigned at-home implementation reading, optional decoder experiments and pinned nanoGPT reading |
| 13 | `13.modern-llm-architectures.html` | Normalization and residual paths (Tue Oct 13, slides 1–13): topology, norm placement/statistics, precision, FLOPs versus runtime, scale/shift explorer, normalization lab, and code reading |
| 14 | `14.ffn-layout-and-tying.html` | LLM architecture choices and dimensions (Thu Oct 15, slides 14–50 of the latest 13-14-15 deck): RMSNorm evidence, FFNs, block layout, tying, FFN/head dimensions, depth/width budgets, checked labs and code reading; ends at the pre-training-data divider |
| 15 | `15.pre-training-data.html` | Pre-training data (Tue Oct 20, printed slides 51–93): collection, cleaning, duplicate removal, pipeline audit, perplexity limits, core exercises, and optional bonus/lab/code reading |
| 16 | `16.pre-training-optimization.html` | Pre-training optimization (Tue Oct 27, printed slides 94–131; PDF pages 92–129): compute allocation, optimizer memory, AdamW, schedules, token batches, packing, repetition, monitoring, core practice, and optional CPU lab/nanoGPT reading |
| 17 | `17.adaptation.html` | Adapting LMs (Thu Oct 29, slides 1–17; slide 18 bonus): response-only targets, freezing and gradient flow, rectangular LoRA counts and initialization, memory accounting, fair evaluation, adapted quiz practice, CPU checks, and pinned original-LoRA code reading |
| 18 | `18.post-training.html` | Instruction tuning and reward models (Tue Nov 3, slides 1–23 of 14-15.post-training.pdf): task holdouts and mixtures, demonstration limits, preference rubrics, expected reward, pairwise loss and gradients, stable implementation/pooling, CPU checks, and pinned TRL reading |
| 19 | `19.policy-gradient.html` | Policy gradients and RLHF (Thu Nov 5, slides 24–36 of 14-15.post-training.pdf): discrete sampling, REINFORCE estimates, sequence masks/sums, feedback loops, reward hacking, reference KL, worked practice, and optional CPU/upstream code reading |
| 20 | `20.reasoning-policy.html` | Post-training reasoning models (Tue Nov 10, printed slides 36–63 / PDF pages 36–62): verifiable outcomes, advantages, required rollout cost accounting, importance weighting, token/sequence ratios, PPO clipping, quiz practice, and optional CPU/CleanRL reading |

Handout #16 includes an offline token-budget calculator (`widgets/pretraining-optimization.js`
and `.css`) and `labs/16_optimization_checks.py`. Core reading is §1–§6; §7 is optional
implementation/code reading and §8 covers bonus slides 106, 115, and 117. The slide mapping
accounts for skipped printed labels: printed 94–131 correspond to PDF pages 92–129.
The cropped IsoFLOP plot and schedule slide retain their axes/legends and carry source-slide
links and attribution. An original packing diagram shows shifted targets, reset position IDs,
and independent-document causal attention. A synthetic equal-compute selection exercise uses
supplied budgets; training-compute formulas are deferred to the scaling-laws lecture.
The lab checks full-batch/accumulated gradient parity, unequal target-count weighting, a masked
next-token bigram LM, packing-mask isolation, AdamW zero-versus-missing gradients, and the cosine scheduler from pinned nanoGPT `3adf61e`.
Numerical examples and deliberate mutation/repair were checked with PyTorch 2.12.0 on CPU;
browser QA covered offline assets, KaTeX, calculator edits and invalid inputs, answers,
checklist persistence, 1280/390/320 px layouts, and print visibility/answer expansion.

Handout #17 adds `widgets/adaptation.js` and `.css`, plus the offline PyTorch lab
`labs/17_adaptation_checks.py`. The memory explorer separates parameter-related memory
from hypothetical activation budgets; it is an accounting model, not a peak-memory predictor.
The gradient-flow explorer compares head tuning, LoRA, and a detached frozen bridge, including
zero-versus-missing first-step gradients; it is a schematic matched to the CPU checks.
Core reading covers slides 1–17; multi-tenant serving from slide 18 is bonus. Exercises adapt
Spring 2025 Quiz 3 Q3.1.1–1.5 and HW6 §3.2(b), with source-page links and independently
checked answers. The LoRA implementation reading pins Microsoft LoRA revision
`c4593f060e6a368d7bb5af5273b8e42810cdef90`, with verbatim excerpts and a retained MIT license.
The handout explains frozen weight-gradient savings, unmerged training, and inference merging;
it does not repeat the slide’s inaccurate unchanged-training-FLOPs claim.

Validation: PyTorch 2.12.0 CPU checks cover response-target shifting/masking, zero-init gradients,
merge parity, frozen-bridge gradient flow, head-only gradients, and rectangular/model-wide counts.
The existing-layer factory is checked for exact initial output parity, preserved frozen weights/bias,
adapter-only updates, and float32 dtype preservation. A feature-collision counterexample and
generation-error diagnosis replace repeated exercises 6A and 1B. The interactive gradient trace
checks all three modes, resets, reachable stages, detached boundaries, and gradient-state tables.
Deliberate merge-sign and no-grad mutations fail as expected; the EOS-mask variation preserves
uniform-logit mean loss while changing supervision. Upstream initialization mutation and repeated
merge/unmerge behavior were run against the pinned file. Browser checks cover all 21 rank/placement
combinations, memory-ranking reversal, invalid inputs, math rendering, answer toggles, checkbox
persistence, offline operation without external requests, 1280/390/320 px layouts, and print expansion.
Static worked examples remain when the widget is hidden for print. The handout uses system fonts
to avoid the course font stylesheet’s remote font requests.

Handout #18 covers PDF/printed slides 1–23 of `14-15.post-training.pdf` and stops before
policy optimization (slide 24). It builds on #17 rather than repeating its response-mask lab.
The new companion adds `widgets/post-training.js` and `.css`, the CPU lab
`labs/18_reward_model_checks.py`, and credited crops of slide 7’s scaling plots, slide 18’s generation/comparison diagram,
and slide 19’s cake responses (Eric Mitchell). The first two panels of Ouyang et al. (2022), Figure 2, are reproduced from the online
InstructGPT paper with attribution, a direct source link, and local source notes. The third
(policy-optimization) panel is outside this handout. Slide 18’s related diagram is expandable.
The cake candidates are paired with a
novice-versus-expert rubric exercise, stack on phones, and link to full-resolution crops. Exercises
adapt Spring 2025 Quiz 3 Q1.3–4 and Q4.4 with source-page links; distribution-shift reasoning
is restricted to reward-model evaluation. The optional code reading pins TRL v0.11.4 revision
`1f8ba929fbb6962249386f7dce20155da8a26cda` and retains its Apache 2.0 license. Verbatim
excerpts separate the basic pairwise loss from optional margin/centering terms.

Validation: PyTorch 2.12.0 CPU checks cover logistic/softplus/BCE parity, gradient directions,
label reversal, offset invariance versus scaling, extreme logits, double sigmoid, shape checks,
annotation disagreement, last-valid pooling under either padding direction, and a shared scorer.
Deliberate sign/pooling mutations fail as expected; the eight-versus-two vote variation and the
upstream sum/centering exercises were checked independently. Browser checks cover 12 score/label
cases, offset invariance, score steps, reset and invalid/boundary inputs, all seven local vector images, including both online-paper panels,
desktop comparison columns, mobile image stacking, full-resolution image links,
math rendering, answers, checklist persistence, 1280/390/320 px layouts, zero external requests,
print expansion, and the course-page link. Static examples remain available without the widget.
The optional six-stage training-step trace follows scores → difference → probability → loss
→ gradients → one scalar-score update. It never autoplays; play/pause, single-stage advance,
rewind, input-change cancellation, and reduced-motion styling are supported. The animation
illustrates reward-model loss mechanics, not policy optimization.
Section 6 provides a 20-point, 20–25 minute quiz-preparation set with five scored questions,
worked solutions, explicit marking guides, and an optional preference-cycle challenge. The set
covers conceptual distinctions, a fresh loss/gradient/update calculation, code repair, expected
reward, and evaluation design. It replaces the former three-question core-practice section;
no policy-optimization algorithm is required. The new probability, signed gradients,
simultaneous score update, expected-reward arithmetic, and corrected/broken code were checked
independently. Browser QA verified all six revealable solutions, 61 rendered math expressions
without errors, narrow layouts, and print expansion; the code-repair page was inspected visually.
The optional PyTorch walkthrough now includes a complete shared-scorer training/validation
pass, TensorDataset/DataLoader usage, separate model/autograd mode controls, detached logging,
zeroing gradients, and pair-weighted metrics for unequal final batches. The expanded CPU
check verifies learning, validation state/gradient preservation, and combined-batch loss parity.
Final scope/quality audit: required explanations stop at reward-model training (slide 23);
policy-update algorithms remain deferred. Reading paths distinguish the lecture companion,
20-point practice set, and optional implementation extensions. Clarifications cover strict
preferences versus tied annotations and independent-score versus shared-parameter updates.
The training loop now rejects multi-output reward heads; code excerpts remain exact.
Independent arithmetic, CPU checks, browser/animation tests, local-source links, and print
layout were rechecked after the fixes.
A final editorial pass removed 830 words (about 14%) of repeated framing, captions, and prose.
Displayed equations, all code blocks, the complete 20-point practice/challenge, images, and
source links are unchanged; browser, offline, and print checks passed after tightening.

Handouts #7 and #8 split the practical-training material at slide 183. Figures and exercises move with their topic; the optional LM-loss application stays in Handout #7 §2.

### Written, session number not yet assigned

These older drafts have no number prefix and are **not linked from `index.html`**. Classes #11 and #12 now have scoped companions below; `transformers.html` remains an optional broader draft, not the assigned reading. When a session is
assigned, rename to `NN.slug.html`, update `data-handout` on `<body>` (it namespaces the
`localStorage` checkbox keys), fill in the masthead's "Session number TBD" and the footer note, and
add the schedule link.

| File | Fits session | Topic |
|------|--------------|-------|
| `rnn-language-models.html` | #8 (Recurrent Neural LMs) | Fixed-window limits, the recurrence, BPTT + truncation, vanishing/exploding, gates, sampling, reading `pytorch/examples` word LM |
| `transformers.html` | #10–#11 (Self-attention, decoder-only) | Attention from the bottleneck problem, causal mask, MHA/GQA, pre-norm block, RoPE, KV cache, reading HF `modeling_llama.py` |

Cross-links already in place: `rnn-language-models.html` links `transformers.html` (§5 and §7.4),
`transformers.html` links `rnn-language-models.html#s5`, and both link Handouts #2/#3/#6.
Renaming a file means fixing those hrefs — `grep -l 'rnn-language-models\|tokenization\.html\|transformers\.html' *.html`.

Widgets: `ssl-objective.js` (Handout #1), `next-token.js` (Handout #2 — order selector, per-model
perplexity table, sampler), `gradient-descent.js` (Handout #4 — learning-rate explorer).

Interactive backprop widgets (`widgets/backprop-intuition.js` and `.css`):
- Handout #5, after §1.5: forward/backward stepper using the same two-hidden-unit worked example;
  editable inputs, parameters, and activation; reveals cached values and summed path contributions.
- Handout #5 §2.2: softmax playground with three logits, target class, common shift, and a direct
  logit gradient step; displays probabilities, cross-entropy, and signed `p - y` gradients.
- Handout #6 §3.2: gradient-state inspector for leaf/non-leaf storage, `retain_grad`, `detach`,
  `no_grad`, and zero derivatives. This is a JavaScript simulation of the displayed PyTorch code.

All three run offline, use keyboard-accessible controls, and are omitted from print along with the
existing widgets. Their surrounding worked examples and code remain printable. Numerical derivatives,
softmax shift invariance, control handlers, and all 48 inspector configurations were checked; inspector
results agree with PyTorch 2.12.0.

Smoothing and backoff methods are not covered in Handout #2. The `next-token.js` widget offers
only fixed orders (unigram/bigram/trigram).

## Manual gradients and tensor-autograd labs

- Handout #5 §2.3 extends the regression lab to a batched multiclass classifier, checking every
  parameter gradient and the input gradient against autograd, plus a finite-difference directional check.
- Handout #6 §3.2 distinguishes leaf/non-leaf gradients, zero versus missing gradients, broken graph
  connections, and shared storage after `detach()`.
- Handout #6 §3.3 covers explicit backward seeds, Jacobian-transpose products, reductions, broadcasting,
  and the distinction between within-graph sums and accumulation across backward calls.
- Handout #6 §4.5 Q3 checks repeated backward in `micrograd`: stale intermediate gradients can produce
  more than twice the first result. Its example is run from a local `micrograd` checkout.

The new runnable labs and exercise variants were checked on PyTorch 2.12.0. The `micrograd` example
was checked against the verbatim `Value` methods included in the handout.

## Code-reading sections

From Handout #6 on, each handout ends with one **reading of real, widely used code** — a short
guided pass over a file from a popular repository, so students build confidence opening unfamiliar
code and recognizing machinery they already understand. Pattern (see Handout #6 §4):

- a `.callout.try` listing the file(s) with direct GitHub links and line counts;
- one `<h3>` per idea, each with a `<p class="hd-srcline">file · symbol</p>` line above a
  **verbatim** excerpt (mark it if abridged — students should be able to diff against the real file);
- explicit mapping from each excerpt back to a numbered section of the handouts;
- at least one "change one character / one word, watch it break" callout with the actual before/after
  numbers, run locally;
- exercises that require editing the file, not just reading it.

Planned assignments (instructor's list):

| Handout | Repo / file | Understanding to assess |
|---------|-------------|-------------------------|
| #6 backprop §4 | [`micrograd/engine.py`](https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py), [`nn.py`](https://github.com/karpathy/micrograd/blob/master/micrograd/nn.py) | Computation graphs, local derivatives, gradient accumulation, reverse topological traversal, how an MLP exposes parameters |
| #9 §7 | [`minbpe/basic.py`](https://github.com/karpathy/minbpe/blob/master/minbpe/basic.py), [`base.py`](https://github.com/karpathy/minbpe/blob/master/minbpe/base.py) | Pair counting, merge loop, vocabulary construction, encode/decode round-trip, and what the file deliberately omits |
| RNN §6–§7 | [`word_language_model/model.py`](https://github.com/pytorch/examples/blob/main/word_language_model/model.py), [`generate.py`](https://github.com/pytorch/examples/blob/main/word_language_model/generate.py), [`main.py`](https://github.com/pytorch/examples/blob/main/word_language_model/main.py) | Recurrent state threading, weight tying, hidden-state detach across batches, sampling loop |
| Transformer §6 | [HF `modeling_llama.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) | Attention block layout, RoPE, KV cache, RMSNorm/GQA/SwiGLU, triage in a 500-line production file |

Pin excerpts against the version you actually read. Commits as of Sept 2026:
`micrograd` `7bc720e`, `minbpe` `1acefe8`, `pytorch/examples` `acc295d`,
`transformers` `ac32445`. Re-check before each offering — `transformers` moves weekly, and the
Transformer handout says so in a callout so students are not confused by a mismatch.

Every excerpt is byte-verbatim against upstream unless its `hd-srcline` says otherwise (trimmed
docstrings, elided argument lists). Keep it that way: students are told they can diff against the
real file.

## `471-671-quiz-samples-public/`

Handouts #4–#8 include adapted exercises from this collection:
- #4: Quiz 1 sp2024 Q1.5–6 and Quiz 1 sp2025 Q1.9/Q2.1–2 (losses, gradient steps, Jacobians).
- #5: HW3 sp2025 §2.7/§3.5–6 (matrix gradients, softmax).
- #6: Quiz 1 sp2024 Q1.14–16/21 (autograd order, gradient accumulation, module registration). Existing graph/complexity questions are retained.
- #7–#8: HW3 sp2025 §1.2/7, HW4 sp2025 §3.2–4, Quiz 1 sp2024 Q1.18/23/24, and Quiz 3 sp2025 Q6.1/3 (vectorization, saturation, residuals, normalization, validation, memory). Matching existing initialization/dropout/debugging exercises now carry source labels.

Additional #7–#8 reasoning exercises adapt Quiz 1 sp2025 Q4.1(c)/Q4.2(b) (batch size,
parameter/gradient shapes, and repeated-example reductions), Quiz 1 sp2024 Q3.2–3 (a scaled
leaky ReLU and its gradient tradeoffs), and Quiz 2 sp2025 Q1.6 (norm clipping, including its
interaction with accumulation). Source PDF page links appear beside each exercise; numerical
extensions and worked answers were checked independently of the sample solutions.

Solution clarifications in these adaptations: the softmax outer product for a column probability vector is `p p^T`, not `p^T p` (HW3 §3.6); memory-component rankings depend on model/batch/optimizer (Quiz 3 Q6.1); clearing gradients at the end of a step is valid when the first step starts clear (Quiz 1 sp2024 Q1.16); normalization does not force equal feature influence (Quiz 1 sp2024 Q1.18). Residual derivatives use column gradients consistently.

Past quizzes and homeworks, with solutions. Source material for handout exercises — several are
ported into Handout #1 §1.3/§1.5/§6 and Handout #2 §1.2/§4.2/§4.4, marked with an `ex-src`
provenance label. Not linked from the schedule.

Three arithmetic errors found in these while porting (handouts use corrected values):
- sp2025 quiz1 Q3.3: `H(s2)` is 3.902 / ppl 14.95, not the printed 3.24 / 9.447 (and the second line
  is mislabelled `H(s1)`).
- sp2025 quiz1 Q3.4: solution says s1 wins "because it contains unobserved patterns" — should be
  *observed*.
- sp2024 quiz1 Q2.4: `H` uses `log2 2/6` while the same page's table and `P(s1)` both use `1/6`.
  With 1/6, H = 1.581 and ppl = 2.99, not 1.33 / 2.51.

PyTorch snippets live in Handout #1 §1.5, Handout #4 §1/§2, Handout #5 §2, Handout #6 §3, Handout #7 §1/§2, Handout #8 §3/§4/§5/§6/§7, the RNN handout
§3/§7 and the Transformer handout §2/§4/§6.
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

## Class 9 tokenizer companion

The unassigned `tokenization.html` draft was replaced by `09.tokenization.html`, linked from session 9.
It builds on #8 rather than repeating the introductory unit tradeoffs. The slide mapping separates core
13–36 material from the blue bonus topics (31 and 35), and includes the 37–49 extension as bonus;
43 and 47 continue surrounding optional topics despite lacking their own blue label. No core check
depends on a bonus topic. A crop of the pipeline from slide 13 is credited inline.

`widgets/tokenization.js` and `.css` provide offline character-BPE training and ranked-inference
explorers. Training lets students choose maximum-frequency ties, undo, and reset; inference includes
the slide trace, a longest-match counterexample, and a frequency-versus-rank counterexample.
Static traces and worked answers remain available without JavaScript and in print.
The code lab links minbpe revision `1acefe89412b20245db5a22d2a02001e547dc602`; examples exercise
training, rank application, byte decoding, overlapping pairs, and the empty-pair training edge case.

Validation: 14 core and 7 bonus checks; Python snippets and the minbpe mutation/repair were run
against the pinned source. Corpus counts, the longest-match counterexample’s training history,
and all three inference traces were checked. Browser QA covered merge selection, undo/reset,
exhaustion, answer toggles, KaTeX rendering, local links, 1280/390/320 px layouts, and print
visibility/answer expansion. No external scripts or styles are required.

## Class 10 MLP LM companion

`10.mlp-language-modeling.html` follows the revised 21-slide MLP deck and is linked from session 10.
The Transformer introduction is outside this handout’s scope. The slide-17 architecture is reused
with its Bengio et al. credit and an explicit note about the schematic output vocabulary.
The original lab uses a deterministic vocabulary, BOS/EOS, batched concatenation, raw-logit
cross-entropy, joint training, repeated-ID gradient checks, and a short sampling loop.
The guided reading pins PyTorch tutorials revision `e924720de192aee7e64c1f8628ec834d7e34a3ea`;
its two excerpts are verbatim. It contrasts recent-first context order, single-example reshaping,
and log-softmax/NLL with the handout’s oldest-first batched implementation.

Validation: original lab and all assertions ran on PyTorch 2.12.0 (training loss 2.3033 → 0.1001);
repeated-ID gradients, parameter counts, the broken reshape, batch-safe tutorial edit, and loss
mutation were checked. Excerpts match the pinned file. Browser QA at 1280/390/320 px found no
page overflow, KaTeX errors, or script errors; local links, figure loading, generated TOC, answer
opening, print answer expansion, and hidden print TOC were verified.


## Classes 11–12 Transformer companions

The current 125-page `10-11-12.transformers.pdf` is split at slide 51, deliberately
included in both: #11 ends with the leakage question, while #12 derives and tests the mask.
Both are linked from their schedule rows. #10 now points to #11; the older `transformers.html`
draft and its cross-links remain available. No RoPE/GQA/SwiGLU unit is added to the core scope.

The companions develop original worked examples and checks rather than transcribing slides.
`widgets/attention.js` and `.css` provide an offline, keyboard-accessible score explorer.
The static retrieval and masked-retrieval examples retain the explanation in print.
Slide 75 is reused with its Waterloo/Vaswani credits and an interpretation of its two input routes.

Original downloadable labs: `labs/11_attention.py` and `labs/12_causal_decoder.py`.
The latter is a two-layer, pre-norm decoder with right-padding support and per-layer KV caches;
it deliberately rejects all-masked queries and does not implement left-padded cached generation.
Code readings pin nanoGPT `3adf61e154c3fe3fca428ad6bc3818b27a3b8291`, `model.py` (330 lines).
Excerpts are verbatim; the MIT notice is retained in `labs/nanoGPT-LICENSE.txt`.

Validation: both labs ran on PyTorch 2.12.0. Prefix invariance, fused-projection
equivalence, row sums, shapes, cache agreement, padding equivalence, valid-target
counting, and backward were checked. Removing causality or restarting cached position
IDs fails the intended assertions. Pinned nanoGPT prefix invariance and the manual
mask/transpose mutation results were verified. All local links and heading IDs resolve.
Browser QA at 1280/390/320 px found no page overflow, KaTeX errors, or script errors;
score edits, prefix selection, reset, answer expansion, and print visibility were checked.
Desktop/mobile/print previews were visually inspected.


### Past-assessment additions for Classes 11–12

Screened all 13 PDFs in `471-671-quiz-samples-public` for the current Transformer scope.
New adaptations draw on Spring 2025 Quiz 2 Q3.1–2, Q3.5–7, Q4.3–4; Homework 6 §2.1–2.2;
and Homework 7 §1's einsum exercise. Sources have local PDF page links beside the questions.
Homework 6's John Hewitt credit is preserved for its mixture/copying question family.

#11 adds hard-versus-soft routing (including the remaining value-gradient path), optional
recovery from orthogonal value subspaces, and an optional named-axis einsum exercise.
#12 adds an additive-mask debugging question, unequal-width shape/cost derivation, and
expands the cache check to explain why past Q tensors are unnecessary for the next step.
Existing axis reasoning is now labeled against the corresponding quiz question.

Corrected/clarified source solutions: Quiz 2 Q3.2 incorrectly says to multiply scores by
the 0/-infinity additive mask; use addition or masked_fill. HW6 §2.1's blanket statement
about inability to backpropagate excludes the differentiable selected-value path; Q/K
routing has no useful ordinary gradient. HW6 §2.2's finite-score copying/two-value-average
constructions are approximate unless other keys are excluded or special values cancel.
HW7's scale-factor naming is made explicit as division by sqrt(d_k).

The nonzero-mean variance extension, arithmetic intensity, parallel-block variants, GQA,
model-specific parameter counts, sampling methods, fine-tuning, and alignment are not
added as core topics to these two slide-scoped readings. Basic blocks, positions, scaling,
training/generation, shapes, and cache concepts in the other samples mostly reinforce
material already present.

Validation: checked the added mask probabilities, NaN multiplication counterexample,
hard-routing value gradients, orthogonal recovery, and einsum/matmul equivalence in PyTorch.
Source page links, IDs, desktop/mobile layouts, KaTeX, and answer/print controls checked.


### Course implementation now has guided reading in #10 and #11

Removed the two loose Class-12 schedule entries (“The final section of the slides on
writing your own Transformer” and “Play with this implementation ...”). The schedule
retains the formatted Handout #12 link. Their content now has guided routes:
#10 §8.3 reads the JHU-CLSP tutorial's byte-token pipeline and shifted targets;
#11 §9.3 reads separate heads and tests fusion, and §9.4 uses slides 112–125 as an
assigned at-home assembly worksheet. The encoder–decoder and cache details still lead into #12.
#12's footer now points to the guided #11 workshop rather than a bare source link.

Pinned JHU-CLSP/jsalt-tutorial revision `92c7e31e08160e843ac33b8532b5e02688ba0503`;
file `lab1-shakespereLM/lab_1_pretraining_a_small_language_model.py` (254 lines).
The two short excerpts are verbatim; source bytes were checked against the Git blob.
`labs/10_11_jsalt_checks.py` is original companion code. It loads selected definitions
from a separately saved upstream file, bypassing its top-level package installation
and corpus download. It verifies UTF-8 IDs, shifted targets, the character/byte ragged
batch counterexample, separate/fused head agreement (288 parameters including biases),
independent layer parameters, untied embedding/head parameters, shapes, and causality.
Checks passed on PyTorch 2.12.0; no Muon install or training download was performed.

The tutorial exercises are optional extensions. Slides 112–125 are assigned reading to complete at home, guided by #11 §9.4 and #12 §7; the in-class #11 slide range remains 1–51.

Validation of these additions: local links and heading IDs resolve, source excerpts match,
and browser checks at 1280/390/320 px found no page overflow, math errors, or script errors.
The optional readings and assembly worksheet were visually inspected in mobile, desktop,
and print views; answers expand for printing.

## Worked traces and reading paths (October 6)

#11 §4.1 now shows two heads retrieving different positions for the same query,
then concatenating their separate feature vectors. The rank exercise uses
T=6, D=8, H=2, giving distinct bounds of 4 per head and 6 for the combined output.
#11 §7.4 follows A B C through token/position embeddings, Q/K/V projections,
prefix attention, residual additions, a ReLU FFN, vocabulary logits, and the
shifted B C A targets. Normalization, biases, and dropout are explicitly omitted
from this fixed-parameter teaching block; the complete block equations remain in §5.
`labs/11_attention.py` independently computes the two-head retrieval and complete
trace, checks the listed tensors, and verifies the capstone target-change result.
The trace's mean loss is 3.02041 nats; changing its final target A to B raises
that mean by 0.1 nats.

#12 §5 has an original accessible SVG cache-step diagram: cached K/V at 0–3,
new q/k/v at 4, K/V append, attention over 0–4, and prediction at position 5.
The caption gives batched head shapes; narrow screens can scroll the diagram.

Both handouts state a core reading path and use zero-based sequence positions.
#11's softmax derivative and permutation proof are optional-depth disclosure boxes;
#11 §8 and §9.1–§9.3, #12 §7's code experiments, and #12 §8 are optional extensions. #11 §9.4 and #12 §7 guide the assigned at-home reading of slides 112–125.
#11 §10 adds four cumulative questions with answers and a bulk answer control.
The optional-depth labels remain visible in print, and answers expand for printing.

Validation: the extended attention lab passes on PyTorch 2.12.0. Browser checks
at 1280/390/320 px report no math errors, script errors, broken images, or page
overflow. Local links and fragments resolve; heading IDs are unique. Optional
boxes, bulk practice answers, widgets, and print expansion work. The new worked
example, head retrieval, and cache diagram were visually inspected in desktop,
mobile, and print views. `git diff --check` passes.


## Classes 13–14: architecture companions

Current slide source: `../slides/13-14-15.modern-llm-architectures.pdf`.
Both handouts and the session-13/14 course rows link this version. Slide numbers
and PDF pages agree through page 50. The superseded 13-14 deck is not the source
for these readings or their current figure crops.

Session 13 stopped at slide 13, the first “Why RMSNorm?” slide.
`13.modern-llm-architectures.html` retains slides 1–13: topology, normalization
placement/statistics, residual gradients, axes, precision, and FLOPs versus runtime.
Its optional lab and pinned nanoGPT reading cover normalization only.

`14.ffn-layout-and-tying.html` now covers slides 14–50. Sections 1–4 cover RMSNorm
evidence/recap (14–16), FFNs (17–29), serial/parallel blocks (30–33), and embedding
tying (34–38). Section 5 covers dimensions and the architecture recap (39–49).
Slide 50 is the pre-training-data divider; substantive data-source and preparation
material starts at 51 and is outside this handout. The handout URL remains stable
although its title is now “LLM Architecture Choices and Dimensions.”

New dimension material develops parameter-matched FFN ratios and rounding,
attention width A=H*head_dim independent of residual D, 4DA projection counts,
and equal-block-parameter depth/width comparisons using 12LD². It distinguishes
embedding totals, ordinary MHA cache sizes, attention products, sequential layer
dependencies, and actual throughput. The aspect-ratio figure supports a broad
empirical basin, not a universal optimum. The old deck’s T5 exception is not an
assigned topic in the current version; the worked rounding example follows the
current gated-width discussion instead.

The current slide 31 already has the correct parallel equation. The old slide-error
note was removed; the handout now interprets its dependency graph. The sample quiz’s
separate serial-equation omission remains flagged. SiLU (the nonlinearity) and
SwiGLU (the gated FFN) are explicitly distinguished. Bias-removal stability is
identified as an empirical observation, separate from parameter savings.

Downloadable CPU-only original PyTorch labs:
- `labs/13_architecture_checks.py`: normalization invariances/axes, float16/reference
  agreement, registered gain updates, and residual Jacobians.
- `labs/14_architecture_checks.py`: gate trace, FFN shapes/gradients/counts, fused
  projections, serial/parallel outputs, shared gradients, and `FlexibleAttention`.
  The latter checks D=10,H=3,head_dim=4 (A=12): 480 projection parameters, output
  shape, causality, agreement with PyTorch SDPA, finite gradients, and failure of
  the residual-derived reshape. Rounding and depth/width counts are checked too.

Both handouts retain their original URLs; reading paths, section numbering,
objectives, reciprocal links, course scopes, and local assessment links are updated.
There are 12 structured exercises in #13 and 19 in #14, each with an answer, plus
figure-reading disclosures. The shared `widgets/architecture.js`/`.css` initialize
only each page’s present controls; progress keys are distinct. The normalization
explorer is in #13; the FFN-budget explorer is in #14.

### Slide visuals and code provenance

All crops in `images/architectures/` were exported from the current 13-14-15 PDF:
- `13-slide08-norm-paths.png`: slide 8, Xiong et al. (2020), Figure 1.
- `13-slide13-operator-cost.png`: slide 13, Ivanov et al. (2020 preprint), Table 1;
  slide credit Tatsu Hashimoto.
- `14-slide22-relu-gelu.png`: slide 22, with axes/legend and the slide’s annotation.
  The caption clarifies its “ELU” typo: the blue curve and lesson concern GELU.
- `14-slide35-embedding-tying.png`: slide 35, preserving the sharing annotation.
  The caption retains the slide’s Press & Wolf reference without asserting it is
  the original artwork source.
- `14-slide48-aspect-ratio.png`: slide 48, Kaplan et al. (2020), Figure 5’s aspect-ratio
  panel; slide credit Tatsu Hashimoto. The text explains the unlabeled vertical
  scale and approximately fixed non-embedding parameter comparison.

Captions identify the source slide/paper; alt text and interpretation prompts
preserve the teaching content. Figures remain visible in print. The operator table
scrolls in a keyboard-focusable region on narrow screens. The original serial/
parallel SVG remains in #14 §3. The PDFs themselves are not edited.

Code readings pin nanoGPT `3adf61e154c3fe3fca428ad6bc3818b27a3b8291` with its MIT
notice. #13 examines `LayerNorm.forward`; #14 examines block layout, sharing, and
the attention divisibility assertion. The constructor-clone caveat, +60 parameter
example, and shape/gradient checks remain. Past assessment adaptations retain
Spring 2025 HW6 §1.8 and Quiz 2 Q1.16, Q4.1–2, Q4.5, and Q4.10–11.


Validation against the latest source: pages/slides 13, 22, 31, 35, 48, 49, and 50
were checked directly, including the updated numbering and data divider. Both labs
pass on PyTorch 2.12.0. The new attention module agrees with SDPA and passes prefix
invariance/backward checks; the deliberate incorrect reshape fails. Rounding,
parameter/cache ratios, new excerpt bytes, answers, local assets/links/fragments,
and contiguous section numbering pass. Browser checks at 1280/390/320 px cover
expanded answers, math, widgets, print expansion/restoration, no-JS reading, and
course-row placement. New dimension sections and refreshed figures were visually
reviewed in desktop/mobile/print views. `git diff --check` passes.

### Optional deepening after the coverage review

Added three optional boxes without extending the session slide ranges or the core
reading paths. #13 §2.1 uses constant common-shift updates to show that LN branch
inputs can remain unchanged while residual RMS grows; its lab verifies all states
and values. #14 §2.2 derives both gated-product gradient paths and explains the
g=0,u=-3 example (gate gradient -1.5, content gradient 0 for incoming gradient 1).
Its lab checks the derivatives and a detach mutation that preserves forward values
but removes the isolated gate path. #14 §4.3 derives the tying fraction relative to
the untied baseline, using N_blocks≈12LD², distinguishes absolute from fractional
savings at fixed vocabulary, and compares the 38/162 versus 38/124 denominators.
The budget assumptions and forward link to §5.3 are explicit; the lab checks the
reported fractions. These are original explanations, not additional assigned topics.

All three optional boxes include answers and retain explicit optional labels in
print. Both labs pass; local links/IDs, math, fully expanded mobile layouts,
widgets, print state restoration, and no-JS reading pass. The new examples were
visually inspected in desktop/mobile/print views. With the warm-ups below, structured exercise counts are
12 in #13 and 19 in #14; these three boxes supplement them. A stale dimension-section
reference was corrected to the current slides 39–49. `git diff --check` passes.


### Warm-ups and quiz-source adaptations

Each handout now opens with three preparation questions intended for 5–7 minutes,
with worked answers and a separate answer toggle. They are ungraded and do not
change the assigned slide ranges or numbered sections.

- #13: a normalization/readout counterexample adapts Spring 2024 Quiz 1 Q1.18
  (pp. 4–5), correcting the source answer's claim that normalization guarantees
  equal feature influence. An original centering puzzle demonstrates loss of a
  common offset. A FLOPs-per-byte calculation extends Spring 2025 Quiz 2 Q1.17
  (p. 4), with explicit limits on inferring runtime from arithmetic intensity.
- #14: a tensor-shape question extends Spring 2025 Quiz 2 Q4.9 (p. 8); an original
  Parameter-aliasing puzzle extends its Q1.16 (p. 4); a scalar serial/parallel
  dataflow puzzle adapts Q4.5 (p. 7). The latter deliberately omits normalization
  to isolate the dependency difference. Source links and adaptation labels appear
  beside each question.

Both downloadable labs verify the warm-up examples. Local links, exercise answers,
math rendering, independent warm-up/practice answer toggles, 1280/390/320 px layouts,
and print expansion/restoration pass. Warm-up desktop/mobile/print views were
visually reviewed. The totals are 12 structured exercises in #13 and 19 in #14.


### Optional theory questions

Four original theory boxes extend the explanations without extending slide scope,
core reading time, or the 12/19 structured-exercise counts. Each contains prediction
or derivation prompts with worked answer disclosures and explicit assumptions.

- #13 `theory-norm-geometry`: sphere/zero-mean geometry, local degrees of freedom,
  the RMSNorm Jacobian, epsilon's radial sensitivity, and the zero-vector limit.
  It distinguishes exact common-shift invariance from approximate scale invariance.
- #13 `theory-residual-angle`: a fixed orthogonal branch update changes a larger
  residual's direction less; the exact 45°/5.71° example avoids claiming a general
  law of training dynamics or gradient size.
- #14 `theory-gate-interactions`: an identity-gated coordinate is a quadratic form;
  the local bias-free SwiGLU expansion explains feature interactions without claiming
  that ordinary nonlinear FFNs lack them or that gating guarantees better quality.
- #14 `theory-tying-conflict`: a scalar constrained-optimization puzzle demonstrates
  opposing route gradients, followed by the feasible-set argument for globally
  optimal unregularized training loss. Finite optimization and validation behavior
  are explicitly distinguished from that theoretical comparison.

The labs verify analytic normalization derivatives/null directions, residual angles,
quadratic identities, the local SwiGLU limit, and tied-gradient cancellation. Both
CPU labs and local link/ID/answer checks pass. Browser checks cover math, expanded
320/390/1280 px layouts, print expansion/restoration, and the existing page controls.


### Handout 14 bonus-slide reading status

The latest deck marks slide 23 and slides 44–48 as Bonus. The GELU CDF/error-function
formula now lives in an optional disclosure in §2.1; qualitative activation behavior
remains core. Entire subsections §5.2 (slides 44–45) and §5.3 (slides 46–48), including
their exercises and figures, are explicitly optional in their headings and callouts.
Practice E–F and the corresponding optional code reading carry matching labels.
The reading path, time estimate, goals, and slide map distinguish core from bonus
material. The GELU disclosure retains its optional heading in print. Slide ranges
and existing section anchors are unchanged.

## Class 15 pre-training data companion

`15.pre-training-data.html` covers printed slides 51–93 of the current
`13-14-15.modern-llm-architectures.pdf`. The final data-preparation summary
is PDF page 91, labeled 93; printed labels 91–92 are absent. The handout stops
before the actual-pre-training divider. Session 15 links the reading beside the
separate Skipjack GPU guide.

Core reading covers collection and token-count denominators, Common Crawl
coverage, cleaning-rule audits, evidence from C4, duplicate weighting and
retention, pipeline order, sharding, and the limits of reference-LM perplexity.
Blue Bonus slides 55, 67–80, 82, and 84–86 are summarized in three optional
collapsed notes in §5; their optional labels remain visible in print. Core
practice does not require the bonus algorithms. There are 19 worked exercises:
three warm-ups, one cleaning question, four optional implementation questions,
and eleven core synthesis questions, plus an inline figure-reading disclosure.

Two full-resolution slide figures preserve source credits and link to full-size
images. `widgets/pretraining-data.js` and `.css` provide an offline synthetic
corpus audit. Its whitespace-word counts are explicitly not tokenizer counts;
static traces remain available without JavaScript and in print.
`labs/15_data_checks.py` runs with the Python standard library and no GPU or
network. It checks retention counts, representative-selection order, perplexity
aggregation and mutations, plus the small optional suffix/Jaccard/mixture
examples. The CCNet reading pins revision
`bda555bd1cf1ee2e0b925363e62a61cd46c8b60d`; its two short excerpts are verbatim,
and the upstream `pp` function was extracted and executed without its optional
dependencies. The handout distinguishes perplexity bucketing from deletion and
corrects the suffix-index and near-duplicate-threshold ambiguities in the bonus
slides without assigning their implementation.

Validation: the lab and upstream `pp` checks pass. Local assets/fragments,
unique heading IDs, exercise answers, source-excerpt bytes, and `git diff --check`
pass. Browser checks cover all nine widget settings, KaTeX, image loading, TOC,
answer toggles, 1280/390/320 px layouts, print expansion/restoration and visible
optional labels, no-JavaScript reading, and both Session 15 course links.
Desktop, mobile, figure, widget, and print-style screenshots were visually reviewed.

Three additional core prediction questions (G–I) examine how deduplication changes
the loss-minimizing conditional distribution, how changed evaluation weights can
hide deterioration in every domain, and how identical paragraphs can still have
different conditioning contexts. Each includes a worked answer and explicit
assumptions; the evaluation question does not require bonus training-mixture
methods. With the further additions below, core reading time is estimated at 45–55 minutes.

Further additions: an optional runnable number-normalization collision exercise
(§6.1), a token-weighted contamination/loss/perplexity calculation (practice J),
and a one-pass versus fixed-token compute question (practice K). K is explicitly
credited as an extension of Spring 2025 Quiz 2 Q4.6, p. 7; its answer states the
budget assumptions needed for the source quiz's compute-saving claim. The
standard-library lab checks the new key collision, perplexities, and pass counts.

### Handout 18 figure quality

The handout now uses SVG assets in `images/post-training/`: the original paper’s
Figure 5(a,c) scaling plots, vector exports of InstructGPT Figure 2 panels 1–2,
and the course slide 18 diagram. Text/paths remain vector; original small icons remain embedded.
The slide 19 cake JPEG has no sharper source in the deck, so both responses are faithfully
typeset as SVG and explicitly credited as transcriptions of Eric Mitchell’s example.
`SOURCES.txt` records original URLs, page/panel locations, and transformations.
The low-resolution PNG derivatives were removed. Scaling plots now stack on phones.

## Class 19 policy-gradient companion

`19.policy-gradient.html` covers printed/PDF slides 24–36 of
`14-15.post-training.pdf`. Slide 27's derivation and the baseline/gradient
justifications are explicitly optional. Core reading develops the sampling
boundary, REINFORCE, response-only sequence sums, the feedback loop, reward
hacking, and current-policy-to-reference KL. It includes three warm-ups, worked
examples, a 20-point practice set with marking guides, and an offline two-response
explorer (`widgets/policy-gradient.js` and `.css`). The explorer compares exact
and selected-outcome gradients and updates a scalar logit; it is a finite model,
not an LM trainer. Its static worked examples remain available without JavaScript.

The optional CPU lab `labs/19_policy_gradient_checks.py` checks exact-versus-
expected sampled gradients, baseline means/variances, reward signs, masks and EOS,
sequence sums, reward detachment, KL gradients, reward-hacking arithmetic, and
practice answers. It executes the pinned PyTorch example's `select_action` in
a stubbed environment and verifies that detaching its saved log-probability
breaks backward. Code reading pins pytorch/examples
`acc295dc7b90714f1bf47f06004fc19a7fe235c4`; excerpts are byte-verbatim and retain
the BSD 3-Clause license. CartPole return normalization and dropout are explicitly
distinguished from the terminal-reward teaching model. Figure sources and transformations are recorded in
`images/policy-gradient/SOURCES.txt`; every figure has alt text, a source or model
caption, and a full-resolution link.

Validation: all lab checks pass on PyTorch 2.12.0 CPU. Deliberate sign and token-mean
mutations fail. Browser QA covers 12 independent numeric cases; exact/selected-
sample updates; invalid, boundary, reset, and preset controls; math rendering;
image loading; TOC; answer toggles; checklist persistence; expanded 1280/390/320 px
layouts; print expansion/restoration and optional labels; no-JavaScript reading;
and the Session 19 schedule link. The handout makes no external asset requests.
Desktop, mobile, widget, source-figure, and print-style screenshots were reviewed.
The local table positioning rule keeps KaTeX accessibility spans within their
scroll containers on phones; the shared handout assets are unchanged.

### Handout 19 visual additions

Replaced the full-slide image with a vector crop of slide 24's policy/scorer
diagram, preserving its trainable/fixed labels. Added InstructGPT Figure 2 panel 3
to complete Handout 18's two panels; the 900-dpi original-PDF crop is 1,612 pixels
wide and retains all text and arrows. A direct SVG export lost the PDF's
transparency groups, so the faithful high-resolution rendering is used instead.

An optional vector crop of Gao et al. (2022), Figure 1(b), adds measured
overoptimization curves with both legends and axes intact. The caption explains
the square-root KL axis, synthetic gold-model evaluator, and the source's PPO/zero
KL-coefficient setting. An original reproducible vector plot compares the same
two-response objective at beta 0, 1, and 2; marked optima were independently
verified by dense-grid search. Two new figure-reading exercises bring the total
to 15, with worked answers; the optional empirical illustration does not assign
the source paper's scaling laws or optimizer. Core reading is now 40–50 minutes.
Browser verification and visual inspection cover the four figures, phone layouts,
print visibility, and full-resolution links.

### Handout 19 extra quiz preparation

Section 6 now separates the existing 20-point timed set (§6.1) from a collapsed,
optional question bank (§6.2). The new bank contains eight core questions (32
points) and two optional challenges (8 points), all with worked solutions and
marking guides. Topics include selection bias, three-way softmax updates, unequal
response counts per prompt, signed masked sequence losses, token-level reference
log-ratios, reference resets, checkpoint tradeoffs, and SFT as a practical rather
than mathematical prerequisite. The challenges cover a response-dependent
one-sample baseline and reward offset/scaling effects. A topic guide supports
choosing a few questions without extending the core reading time.

Questions 6K and 6M adapt Spring 2025 Quiz 3 Q1.5 (PDF p. 2) and Q4.1 (p. 7),
with source links and explicit revised assumptions. The solutions distinguish a
KL penalty from a truthfulness guarantee and avoid treating SFT as necessary for
the score-function identity. There are now 25 structured exercises in the handout.

Validation: fresh arithmetic was checked independently, including autograd for
binary and three-response gradients. IDs, local links, source-page locations,
point totals, and all answer disclosures pass. Browser checks cover independent
bank/timed-set answer toggles, KaTeX, expanded 1280/390/320 px layouts, print
expansion/restoration and visible optional labels, and no-JavaScript access.
Desktop, phone, and print-style screenshots of the additions were reviewed.

### Handout 19: KL definition and penalty derivation

Section 5.2 now defines Kullback–Leibler divergence as a distribution-weighted sum
and expectation. It explains the log-ratio, nonnegativity/equality, asymmetry,
support conditions, nats, and the entropy/cross-entropy identity. An optional proof
uses -log(u) >= 1-u; a two-response table checks both forward and reverse KL.

New §5.3 expands the expected adjusted reward line by line to identify the
subtracted beta * KL term, then restores the outer prompt expectation. It
emphasizes that the equality is exact and depends on current-policy sampling.
A direct adjusted-reward average agrees with reward-minus-KL numerically; the
sequence/token factorization connects the penalty to response masks and EOS.
The policy-gradient discussion is now §5.4. New exercise 5C shows that reference
sampling produces the negative reverse KL for the same log-ratio. There are now
26 structured exercises; the optional question bank and its point totals are
unchanged. Core reading time is 45–55 minutes.

Validation: independently checked entropy/cross-entropy, forward/reverse KL,
weighted adjusted rewards, and the reverse-sampling sign. All local links, IDs,
and answer disclosures pass. Browser checks cover math rendering, expanded
1280/390/320 px layouts, the new exercise disclosure, and print equations and
optional-proof labels. Desktop, phone, and print-style screenshots were reviewed.
Long display derivations use the existing horizontal equation scrolling on phones.

### Handout 19 correctness and tightening review

Reviewed the entire handout against slides 24–36, figure provenance, code excerpts,
and worked answers. The main policy-gradient/KL formulas and numerical solutions
check out. Clarified that Challenge 1 differentiates the Bernoulli logit, and that
exercise 5C changes sampling for the entire adjusted reward. Replaced exercise 5A's
repeated arithmetic with a question about sample bonuses versus expected penalties.
Shortened repeated explanations and the InstructGPT caption; made token-gradient
summing explicit. The central KL derivation now uses declared fixed-prompt shorthand
and shorter lines that fit phone screens. All 26 exercises and optional material remain.

The explorer now validates every input on each edit, preventing stale aria-invalid
states when several fields are invalid. All numeric behavior is unchanged. Local
print rules reduce padding/spacing, hide answer-toggle buttons, remove trailing
body space, and keep the KL proof/derivation intact. The complete Letter-size
print audit with all answers and optional sections expanded fell from 38 to 27
pages; no equation, table, or code scroll overflow was detected at print width.

Validation: the CPU lab passes on PyTorch 2.12.0; 60 independent finite-distribution
cases agree between exact autograd and expected detached-coefficient gradients.
Verbatim upstream excerpts and the displayed loss/lab function match. Browser QA
passes numeric cases, selected/exact updates, all invalid-field states, independent
answer controls, KaTeX, figures, TOC, checklist persistence, offline/no-JS access,
expanded 1280/390/320 layouts, and print expansion/restoration. Local links, IDs,
source fragments, exercise counts, and git diff whitespace checks pass. The full
print contact sheet and final KL pages were visually reviewed.
A final visual check also found inherited uppercase styling on table-header math;
the local stylesheet now preserves the case of p(y), q(y), and other symbols.

### Handout 19: generation versus supplied-response scoring

New §3.3 distinguishes one next-token prediction, autoregressive response
production, and scoring an already supplied response. It explains prompt prefill
plus T-1 cached decode passes versus one causally masked teacher-forced scoring
pass with P+T-1 input positions. Gathering target IDs is cheap only after logits
and vocabulary normalization have been computed. The required cost accounting
includes Transformer, vocabulary-head, and normalization costs, separates the
same asymptotic forward arithmetic from different latency, and notes backward
work for policy training. It links to Handout 12's existing cache/cost discussion.
Exercise 3B checks token counts, input shifting, and model-call counts; there are
now 27 exercises. Core reading time is 50–60 minutes.

The CPU lab adds check_parallel_response_scoring: a deterministic tiny causal
Transformer agrees between one bulk pass and separate prefix conditionals,
including EOS; changing a future token cannot affect earlier outputs, and bulk
scoring retains gradients. This is a probability-parity check, not a KV-cache
implementation or a runtime benchmark. All lab checks pass on PyTorch 2.12 CPU;
local assets/fragments, math rendering, expanded 1280/390/320 px layouts, and
print equation widths pass.

### Handout 19 cost accounting is required

The full cost accounting in §3.3 is core reading. Its former optional disclosure
was replaced by an always-visible section. Scope, reading path, learning objectives,
and the section checklist explicitly include generation/scoring arithmetic and
sequential latency. Exercise 3B now also asks for the dense forward cost, including
the response-only vocabulary head and normalization.

## Class 20 reasoning-model post-training companion

`20.reasoning-policy.html` covers printed slides 36–63 of the current
`14-15.post-training.pdf`, corresponding to PDF pages 36–62 because printed 39 is
absent. It stops at the first clipping pulse check; group-relative advantages
and the full GRPO objective begin later. Core reading connects the KL recap to
reasoning completions, outcome/process/format rewards, detached prompt baselines,
required rollout-versus-scoring cost accounting, old-policy importance ratios,
and the sign-dependent PPO clipped surrogate. It explicitly separates the fixed
KL reference from the policy that generated a rollout batch.

The handout has 19 structured exercises: three warm-ups, six in-section checks,
a five-question 20-point practice set with marking guides, four optional extra
questions, and an implementation-reading exercise. The offline widget
`widgets/reasoning-policy.js`/`.css` shows both surrogate branches, their minimum,
the minimization sign, local slopes and kinks, and a scalar-ratio plot. It does
not update a policy or treat clipping thresholds as hard probability bounds.
Static worked cases and the original-paper diagram remain without JavaScript
and in print. Phone plots scroll horizontally to retain readable axes.

The native-resolution JPEG from slide 38 retains the full-vocabulary token-KL
diagram. Its caption distinguishes the pictured token mean from a sequence sum.
The PPO Figure 1 crop comes from arXiv:1707.06347v2 and remains vector. Source
credits, PDF-page links, alt text, and transformations are recorded in
`images/reasoning-policy/SOURCES.txt`. The rule-based reward example uses the
January 2025 R1 report, with R1-Zero distinguished from multi-stage R1.

The optional CPU lab `labs/20_reasoning_policy_checks.py` checks verifier coverage,
advantage signs and baseline cancellation, the moving-zero-baseline pitfall,
importance identities and gradients, support failures, sequence/token ratios,
clipping cases and finite differences, boundary finiteness, data detachment,
overflow rejection, deliberate mutations, and practice calculations. The pinned
CleanRL reading uses `cleanrl/ppo.py` at
`fe8d8a03c41a7ef5b523e2e354bd01c363e786bb` (312 lines). Excerpts are byte-verbatim,
the complete upstream license text is retained (trailing whitespace normalized), and the extracted policy-loss fragment
is executed in a minimal namespace. Minibatch normalization, GAE/critic details,
entropy terms, and optional stopping are identified as separate implementation
choices outside the assigned derivations.

Validation: all CPU checks pass on PyTorch 2.12.0. The browser widget agrees with
16 independently computed CPU fixtures (including zero advantage) and four
active-kink cases. Invalid/boundary inputs, presets/reset, numeric slopes, math rendering,
figure loading, TOC, independent answer toggles, checklist persistence, offline
and no-JS operation, and the Class 20 course link pass. Expanded 1280/390/320 px
layouts and print expansion/restoration/optional labels pass with no remote
asset requests or browser errors. Desktop/mobile/figure/widget screenshots and
the complete paginated print rendering were visually reviewed; equations and
tables fit the print width. Local assets/fragments, heading IDs, exercise answers,
source excerpt bytes, displayed/lab function parity, and whitespace checks pass.
