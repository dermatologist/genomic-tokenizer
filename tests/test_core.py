import pytest
from genomic_tokenizer.core import GenomicTokenizer

def test_vocab_size():
	tokenizer = GenomicTokenizer(model_max_length=512)
	# expected_vocab_size = len(tokenizer.codons) + len(tokenizer._vocab_str_to_int) - len(tokenizer.codons) + 1  # Adjusting for the unique keys in codons and special tokens
	# assert tokenizer.vocab_size == expected_vocab_size
	assert tokenizer.vocab_size == 28

def test_tokenize_gene_fna():
	with open("tests/gene.fna", "r") as file:
		gene_data = file.read().replace("\n", "")
	tokenizer = GenomicTokenizer(model_max_length=512)
	tokenized_data = tokenizer._tokenize(gene_data)
	print(tokenized_data)
	assert len(tokenized_data) != 0
	#Further assertions can be added based on expected tokenized data characteristics