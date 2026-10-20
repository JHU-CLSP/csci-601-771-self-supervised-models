#!/usr/bin/env python3
"""Handout 15: deterministic data-pipeline exercises. Python 3, stdlib only.

Run: python 15_data_checks.py
Synthetic texts and whitespace counts are teaching examples, not LM tokens.
No crawl downloads, GPU, language classifier, or KenLM installation is needed.
"""
import ast
import math
import re
from pathlib import Path

DOCUMENTS = [
    ('a', 'Water freezes at zero degrees Celsius.'),
    ('b', 'Water freezes at zero degrees Celsius.'),
    ('c', 'water   freezes at zero degrees celsius.'),
    ('d', 'Menu Home Login Subscribe'),
    ('e', 'def square(x): return x*x'),
    ('f', 'Plants use sunlight to make sugars.'),
    ('g', 'Water boils at one hundred degrees Celsius.'),
    ('h', 'PLANTS use sunlight to make sugars.'),
]


def key(text, mode):
    if mode == 'normalized':
        return ' '.join(text.lower().split())
    if mode == 'exact':
        return text
    raise ValueError(mode)


def prepare(documents, minimum_words=5, mode='normalized'):
    """Filter, then deduplicate; preserve original text and first accepted copy."""
    seen, kept, log = set(), [], []
    for doc_id, text in documents:
        if len(text.split()) < minimum_words:
            log.append((doc_id, 'short'))
            continue
        match_key = key(text, mode) if mode != 'none' else None
        if mode != 'none' and match_key in seen:
            log.append((doc_id, 'duplicate'))
            continue
        seen.add(match_key)
        kept.append((doc_id, text))
        log.append((doc_id, 'keep'))
    return kept, log


def score_perplexity(log10_scores, lengths):
    """Length-weighted aggregate, matching the pp convention in CCNet."""
    return 10 ** (-sum(log10_scores) / sum(lengths))


def optional_checks():
    # This literal suffix construction is a tiny illustration, not scalable code.
    text = 'abaaba$'
    suffix_array = sorted(range(len(text)), key=lambda i: text[i:])
    assert suffix_array == [6, 5, 2, 3, 0, 4, 1]
    assert sorted(i for i in suffix_array if text[i:].startswith('aba')) == [0, 3]
    a, b = set('a b c d'.split()), set('a b c e'.split())
    assert len(a & b) / len(a | b) == 3 / 5
    # Synthetic 90M/10M corpus, 100M processed tokens, 50/50 sampling.
    assert 100 * .5 / 90 == 5 / 9
    assert 100 * .5 / 10 == 5


def verify_upstream(path):
    """Optional: execute only CCNet's dependency-free pp function from a checkout."""
    tree = ast.parse(Path(path).read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'pp')
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
    assert namespace['pp'](-6, 3) == 100
    assert math.isclose(namespace['pp'](-11, 12), 8.254041852680183)
    print('Pinned-source pp checks passed.')


def main():
    assert sum(len(text.split()) for _, text in DOCUMENTS) == 45
    expected = {'none': (6, 37), 'exact': (5, 31), 'normalized': (3, 19)}
    for mode, counts in expected.items():
        kept, log = prepare(DOCUMENTS, mode=mode)
        actual = len(kept), sum(len(text.split()) for _, text in kept)
        assert actual == counts, (mode, actual)
        assert len(log) == len(DOCUMENTS)
        print(f'{mode}: {actual[0]} documents, {actual[1]} whitespace tokens')
    kept, log = prepare(DOCUMENTS)
    assert [doc_id for doc_id, _ in kept] == ['a', 'f', 'g']
    assert dict(log)['e'] == 'short'  # Valid code falsely rejected by a prose heuristic.
    assert prepare([]) == ([], [])
    assert key('A  B', 'normalized') == key('a b', 'normalized')
    assert key('Version 2024', 'normalized') != key('Version 2025', 'normalized')

    # Why selecting the first duplicate before a source-policy filter can lose data.
    copies = [('unreviewed', 'same text'), ('accepted', 'same text')]
    def unique(rows):
        seen, out = set(), []
        for source, text in rows:
            if text not in seen:
                seen.add(text)
                out.append((source, text))
        return out
    allow = lambda rows: [row for row in rows if row[0] == 'accepted']
    assert allow(unique(copies)) == []
    assert unique(allow(copies)) == [('accepted', 'same text')]

    assert math.isclose(score_perplexity([-2, -9], [2, 10]), 8.254041852680183)
    assert not math.isclose(score_perplexity([-2, -9], [2, 10]), (10 + 10**.9) / 2)
    correct, wrong_base = 10 ** 2, math.exp(2)
    assert correct == 100 and math.isclose(wrong_base, 7.38905609893065)
    # Deliberately dropping the minus sign in pp gives .01, not 100.
    assert 10 ** (-6 / 3) == .01
    optional_checks()
    print('All core and optional numerical checks passed.')


if __name__ == '__main__':
    import sys
    main()
    if len(sys.argv) > 1:
        verify_upstream(sys.argv[1])
