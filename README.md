# Genomic Tokenizer

## About

This is a tokenizer for genomic data. It is designed to Tokenize DNA sequences in the FASTA format.

## Installation

```bash
pip install git+https://github.com/dermatologist/genomic-tokenizer.git
```

## Example usage

```python
from genomic_tokenizer import GenomicTokenizer
# Remove the fasta header if present
fasta = """
AGGCGTACCCGCCCCTGGGGACGTCATTGGTGGCGGAGGCAATGGCCGGCAACCAGCTGTAAGCGAGGTA
GGCTCACTCGGGCGCGGAGGGTGCGGGTGAGAAAGGGAACGATTTGCTAGGAGTGTATGCGCCCGTGCTA
GGGCTCCCAGGAATTCTTACAAGCGTAGACAGCCTAGCAATCAGCCCTGTGAAGTGGGGTTCAGGGCAGA
"""
model_max_length = 2048
tokenizer = GenomicTokenizer(model_max_length)
tokens = tokenizer(fasta)
print(tokens)
```

### Output
```
{'input_ids': [2, 7, 12, 17, 19, 16, 1, 7, 20, 6, 12, 21, 16, 12, 20, 12, 12, 8, 12, 1, 10, 20, 10, 20, 11, 7, 20, 21, 23, 8, 7, 20, 7, 6, 12, 21, 19, 10, 11, 16, 19, 7, 1, 22, 7, 1, 19, 21, 7, 16, 1, 21, 12, 23, 19, 12, 20, 6, 1], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
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