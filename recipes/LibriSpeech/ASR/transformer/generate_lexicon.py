#!/usr/bin/env python3

# Copyright (c)  2021  Xiaomi Corporation (authors: Fangjun Kuang)

"""
This script takes as inputs the following files:
    - data/lang/bpe/bpe.model,
    - data/lang/bpe/tokens.txt (will remove it),
    - data/lang/bpe/words.txt

and generates the following files in the directory data/lang/bpe:

    - lexicon.txt
    - lexicon_disambig.txt
    - L.pt
    - L_disambig.pt
    - phones.txt
"""

from pathlib import Path

import k2

import torch

from speechbrain.decoders.prepare_lang_bpe import (
    write_lexicon,
    add_disambig_symbols,
    lexicon_to_fst_no_sil,
    generate_lexicon,
)


def main():
    lang_dir = Path("data/lang/bpe")
    model_file = lang_dir / "bpe.model"

    word_sym_table = k2.SymbolTable.from_file(lang_dir / "words.txt")

    words = word_sym_table.symbols

    excluded = ["<eps>", "!SIL", "<SPOKEN_NOISE>", "<UNK>", "#0", "<s>", "</s>"]
    for w in excluded:
        if w in words:
            words.remove(w)

    lexicon = generate_lexicon(model_file, words)

    # TODO(fangjun): Remove tokens.txt and generate it from the model directly.
    #
    # We are using it since the IDs we are using in tokens.txt is
    # different from the one contained in the model
    token_sym_table = k2.SymbolTable.from_file(lang_dir / "tokens.txt")

    lexicon_disambig, max_disambig = add_disambig_symbols(lexicon)

    for i in range(max_disambig + 1):
        disambig = f"#{i}"
        assert disambig not in token_sym_table
        token_sym_table.add(f"#{i}")

    word_sym_table.add("#0")
    word_sym_table.add("<s>")
    word_sym_table.add("</s>")

    token_sym_table.to_file(lang_dir / "phones.txt")

    write_lexicon(lang_dir / "lexicon.txt", lexicon)
    write_lexicon(lang_dir / "lexicon_disambig.txt", lexicon_disambig)

    L = lexicon_to_fst_no_sil(
        lexicon,
        token2id=token_sym_table,
        word2id=word_sym_table,
    )

    L_disambig = lexicon_to_fst_no_sil(
        lexicon_disambig,
        token2id=token_sym_table,
        word2id=word_sym_table,
        need_self_loops=True,
    )
    torch.save(L.as_dict(), lang_dir / "L.pt")
    torch.save(L_disambig.as_dict(), lang_dir / "L_disambig.pt")

    if False:
        # Just for debugging, will remove it
        L.labels_sym = k2.SymbolTable.from_file(lang_dir / "phones.txt")
        L.aux_labels_sym = k2.SymbolTable.from_file(lang_dir / "words.txt")
        L_disambig.labels_sym = L.labels_sym
        L_disambig.aux_labels_sym = L.aux_labels_sym
        L.draw(lang_dir / "L.svg", title="L")
        L_disambig.draw(lang_dir / "L_disambig.svg", title="L_disambig")


if __name__ == "__main__":
    main()
