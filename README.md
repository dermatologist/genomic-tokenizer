# Genomic Tokenizer

## About

This is a tokenizer for genomic data. It is designed to Tokenize DNA sequences in the FASTA format.

## Installation

```bash
pip install git+https://github.com/dermatologist/genomic-tokenizer.git
```

### Tokenization algorithm
Identify the first occurence of the start codon `ATG`.
Split the sequence into codons of length 3 starting from the start codon.
Convert synonymous codons to the same token.
Convert stop codons to [SEP] token.

## Inspired by

* https://github.com/HazyResearch/hyena-dna/blob/main/src/dataloaders/datasets/hg38_char_tokenizer.py
* https://github.com/dariush-bahrami/character-tokenizer/blob/master/charactertokenizer/core.py
* And the CanineTokenizer in transformers package.

## Cite as

```
@misc{genomic-tokenizer,
  author = {Bell Raj Eapen},
  title = {Genomic Tokenizer},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{
    https://github.com/dermatologist/genomic-tokenizer
    }},
}
```

## Give us a star ⭐️
If you find this project useful, give us a star. It helps others discover the project.

## Contributors

* [Bell Eapen](https://nuchange.ca) | [![Twitter Follow](https://img.shields.io/twitter/follow/beapen?style=social)](https://twitter.com/beapen)