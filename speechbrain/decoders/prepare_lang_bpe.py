#!/usr/bin/env python3

# Copyright (c)  2021  Xiaomi Corporation (authors: Fangjun Kuang)

"""
This script takes as input a lexicon file "data/lang/lexicon.txt"
consisting of words and phones and generates the following files
in the directory data/lang/bpe:

    - lexicon.txt
    - lexicon_disambig.txt
    - L.pt
    - L_disambig.pt
    - phones.txt
"""


from collections import defaultdict
from typing import Any, Dict, List, Tuple

import sentencepiece as spm

import k2

Lexicon = List[Tuple[str, List[str]]]


def write_lexicon(filename: str, lexicon: Lexicon) -> None:
    """Write a lexicon to a file.

    Args:
      filename:
        Path to the lexicon file to be generated.
      lexicon:
        It can be the return value of :func:`read_lexicon`.
    """
    with open(filename, "w", encoding="utf-8") as f:
        for word, prons in lexicon:
            f.write(f"{word} {' '.join(prons)}\n")


def write_mapping(filename: str, sym2id: Dict[str, int]) -> None:
    """Write a symbol to ID mapping to a file.

    Note:
      No need to implement `read_mapping` as it can be done
      through :func:`k2.SymbolTable.from_file`.

    Args:
      filename:
        Filename to save the mapping.
      sym2id:
        A dict mapping symbols to IDs.
    Returns:
      Return None.
    """
    with open(filename, "w", encoding="utf-8") as f:
        for sym, i in sym2id.items():
            f.write(f"{sym} {i}\n")


def add_disambig_symbols(lexicon: Lexicon) -> Tuple[Lexicon, int]:
    """It adds pseudo-phone disambiguation symbols #1, #2 and so on
    at the ends of phones to ensure that all pronunciations are different,
    and that none is a prefix of another.

    See also add_lex_disambig.pl from kaldi.

    Args:
      lexicon:
        It is returned by :func:`read_lexicon`.
    Returns:
      Return a tuple with two elements:

        - The output lexicon with disambiguation symbols
        - The ID of the max disambiguation symbols that appears
          in the lexicon
    """

    # (1) Work out the count of each phone-sequence in the
    # lexicon.
    count = defaultdict(int)
    for _, prons in lexicon:
        count[" ".join(prons)] += 1

    # (2) For each left sub-sequence of each phone-sequence, note down
    # that it exists (for identifying prefixes of longer strings).
    issubseq = defaultdict(int)
    for _, prons in lexicon:
        prons = prons.copy()
        prons.pop()
        while prons:
            issubseq[" ".join(prons)] = 1
            prons.pop()

    # (3) For each entry in the lexicon:
    # if the phone sequence is unique and is not a
    # prefix of another word, no disambig symbol.
    # Else output #1, or #2, #3, ... if the same phone-seq
    # has already been assigned a disambig symbol.
    ans = []

    # We start with #1 since #0 has its own purpose
    first_allowed_disambig = 1
    max_disambig = first_allowed_disambig - 1
    last_used_disambig_symbol_of = defaultdict(int)

    for word, prons in lexicon:
        phnseq = " ".join(prons)
        assert phnseq != ""
        if issubseq[phnseq] == 0 and count[phnseq] == 1:
            ans.append((word, prons))
            continue

        cur_disambig = last_used_disambig_symbol_of[phnseq]
        if cur_disambig == 0:
            cur_disambig = first_allowed_disambig
        else:
            cur_disambig += 1

        if cur_disambig > max_disambig:
            max_disambig = cur_disambig
        last_used_disambig_symbol_of[phnseq] = cur_disambig
        phnseq += f" #{cur_disambig}"
        ans.append((word, phnseq.split()))
    return ans, max_disambig


def generate_id_map(symbols: List[str]) -> Dict[str, int]:
    """Generate ID maps, i.e., map a symbol to a unique ID.

    Args:
      symbols:
        A list of unique symbols.
    Returns:
      A dict containing the mapping between symbols and IDs.
    """
    return {sym: i for i, sym in enumerate(symbols)}


def add_self_loops(
    arcs: List[List[Any]], disambig_phone: int, disambig_word: int
) -> List[List[Any]]:
    """Adds self-loops to states of an FST to propagate disambiguation symbols
    through it. They are added on each state with non-epsilon output symbols
    on at least one arc out of the state.

    See also fstaddselfloops.pl from Kaldi. One difference is that
    Kaldi uses OpenFst style FSTs and it has multiple final states.
    This function uses k2 style FSTs and it does not need to add self-loops
    to the final state.

    Args:
      arcs:
        A list-of-list. The sublist contains
        `[src_state, dest_state, label, aux_label, score]`
      disambig_phone:
        It is the phone ID of the symbol `#0`.
      disambig_word:
        It is the word ID of the symbol `#0`.

    Return:
      Return new `arcs` containing self-loops.
    """
    states_needs_self_loops = set()
    for arc in arcs:
        src, dst, ilabel, olabel, score = arc
        if olabel != 0:
            states_needs_self_loops.add(src)

    ans = []
    for s in states_needs_self_loops:
        ans.append([s, s, disambig_phone, disambig_word, 0])

    return arcs + ans


def lexicon_to_fst_no_sil(
    lexicon: Lexicon,
    token2id: Dict[str, int],
    word2id: Dict[str, int],
    need_self_loops: bool = False,
) -> k2.Fsa:
    """Convert a lexicon to an FST (in k2 format).

    Args:
      lexicon:
        The input lexicon. See also :func:`read_lexicon`
      token2id:
        A dict mapping tokens to IDs.
      word2id:
        A dict mapping words to IDs.
      need_self_loops:
        If True, add self-loop to states with non-epsilon output symbols
        on at least one arc out of the state.
    Returns:
      Return an instance of `k2.Fsa` representing the given lexicon.
    """
    loop_state = 0  # words enter and leave from here
    next_state = 1  # the next un-allocated state, will be incremented as we go.

    arcs = []

    assert token2id["<unk>"] == 0
    assert word2id["<eps>"] == 0

    eps = 0

    for word, prons in lexicon:
        assert len(prons) > 0, f"{word} has no pronunciations"
        cur_state = loop_state

        word = word2id[word]
        prons = [token2id[i] for i in prons]

        for i in range(len(prons) - 1):
            if i == 0:
                arcs.append([cur_state, next_state, prons[i], word, 0])
            else:
                arcs.append([cur_state, next_state, prons[i], eps, 0])

            cur_state = next_state
            next_state += 1

        # now for the last phone of this word
        i = len(prons) - 1
        w = word if i == 0 else eps
        arcs.append([cur_state, loop_state, prons[i], w, 0])

    if need_self_loops:
        disambig_phone = token2id["#0"]
        disambig_word = word2id["#0"]
        arcs = add_self_loops(
            arcs, disambig_phone=disambig_phone, disambig_word=disambig_word,
        )

    final_state = next_state
    arcs.append([loop_state, final_state, -1, -1, 0])
    arcs.append([final_state])

    arcs = sorted(arcs, key=lambda arc: arc[0])
    arcs = [[str(i) for i in arc] for arc in arcs]
    arcs = [" ".join(arc) for arc in arcs]
    arcs = "\n".join(arcs)

    fsa = k2.Fsa.from_str(arcs, acceptor=False)
    return fsa


def generate_lexicon(model_file: str, words: List[str]) -> Lexicon:
    """Generate a lexicon from a BPE model.

    Args:
      model_file:
        Path to a sentencepiece model.
      words:
        A list of strings representing words.
    Returns:
      Return a dict whose keys are words and values are the corresponding
      word pieces.
    """
    sp = spm.SentencePieceProcessor()
    sp.load(str(model_file))

    words_pieces: List[List[str]] = sp.encode(words, out_type=str)

    lexicon = []
    for word, pieces in zip(words, words_pieces):
        lexicon.append((word, pieces))

    #lexicon.append(("<UNK>", ["<UNK>"]))
    lexicon.append(("<UNK>", ["<unk>"]))
    
    return lexicon
