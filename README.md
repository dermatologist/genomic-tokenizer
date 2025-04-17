# :chains: Genomic Tokenizer

[![PyPI download total](https://img.shields.io/pypi/dm/genomic-tokenizer.svg)](https://pypi.python.org/pypi/genomic-tokenizer/)
![GitHub tag (latest by date)](https://img.shields.io/github/v/tag/dermatologist/genomic-tokenizer)

## About
This is a tokenizer for DNA :chains: that aligns with the central dogma of molecular biology. The Genomic Tokenizer (GT) incorporates the biological process flow into a [**standard tokenizer interface** within the HuggingFace transformer package](https://huggingface.co/docs/transformers/en/main_classes/tokenizer). GT can be used to pre-train foundational transformer models on DNA sequences. [[Read the preprint](https://www.biorxiv.org/content/10.1101/2025.04.02.646836v1)]

Please [cite](#books-cite) / [contact me](https://nuchange.ca/contact) if you use it in your research.

## 🚀 Installation

```bash
pip install genomic-tokenizer
```

### If you want to install the latest version from the repository, use the following command:
```bash
pip install git+https://github.com/dermatologist/genomic-tokenizer.git
```

## 🔧 Example usage

```python
from genomic_tokenizer import GenomicTokenizer
# Fasta header if present is ignored.
fasta = """
AGGCGAGGCGCGGGCGGAGGCGGTGCGCGGGCGGAGGCGGGGCGCGGAGATGTGGCGGAGGTGGAGGCGG
AGGCGTAGCCGCCCCTGGGGACGTCATTGGTGGCGGAAGCAATCGCCGGCAACCAGCTGTAAGCGAGGTA
GGCTCACTCGGGCACGGAGGGTGCGGGTGAGAAAGGGAACGATTTGCTAGGAGTGTATGCGCCCGTGCTA
"""
model_max_length = 2048
tokenizer = GenomicTokenizer(model_max_length)  # Use this in your model training pipeline
tokens = tokenizer(fasta)
print(tokens)
```

### ✨ Output
```
{'input_ids': [2, 7, 12, 17, 19, 16, 1, 7, 20, 6, 12, 21, 16, 12, 20, 12, 12, 8, 12, 1, 10, 20, 10, 20, 11, 7, 20, 21, 23, 8, 7, 20, 7, 6, 12, 21, 19, 10, 11, 16, 19, 7, 1, 22, 7, 1, 19, 21, 7, 16, 1, 21, 12, 23, 19, 12, 20, 6, 1],
'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
```

## 🔧 Tokenization algorithm
* Identify the first occurrence of the start codon `ATG`.
* Split the sequence into codons of length 3 starting from the start codon.
* Convert synonymous codons to the same token.
* Convert stop codons to `[SEP]` token.


## :books: Cite

```

@misc{GT-Eapen2025,
  title = {Genomic {{Tokenizer}}: {{Toward}} a Biology-Driven Tokenization in Transformer Models for {{DNA}} Sequences},
  shorttitle = {Genomic {{Tokenizer}}},
  author = {Eapen, Bell Raj},
  year = {2025},
  month = apr,
  pages = {2025.04.02.646836},
  publisher = {bioRxiv},
  doi = {10.1101/2025.04.02.646836},
  urldate = {2025-04-11},
}

```

## Give us a star ⭐️
If you find this project useful, give us a star. It helps others discover the project.

## Contributors

* [Bell Eapen](https://nuchange.ca) | [![Twitter Follow](https://img.shields.io/twitter/follow/beapen?style=social)](https://twitter.com/beapen)
