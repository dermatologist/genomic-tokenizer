import pytest
from genomic_tokenizer.core import GenomicTokenizer

def test_vocab_size():
	tokenizer = GenomicTokenizer(model_max_length=512)
	# expected_vocab_size = len(tokenizer.codons) + len(tokenizer._vocab_str_to_int) - len(tokenizer.codons) + 1  # Adjusting for the unique keys in codons and special tokens
	# assert tokenizer.vocab_size == expected_vocab_size
	assert tokenizer.vocab_size == 71

def test_input_ids():
    tokenizer = GenomicTokenizer(model_max_length=512)
    # with open("tests/gene.fna", "r") as file:
    # 	gene_data = file.read().replace("\n", "")
    gene_data = """
CAGTCTGAGCCTGGCCGTCGCCTCCAGCAAAGCTTGAGCTGCAGGAATGTCCCCGGCCTTGGCTCCCAGTG
CCCTCCTTGGGGTCAAGGCCACCTCATCCTTGCCCCCAGGGGTGATACCTCGGGGGTTCTCCAGGCTGAGG
CACCTGCAGGGCATAGGAAGGATGCAGGGCTTATGGTCTAGAGGAGGCAGAGGGAACTCTGGGCCCTGATG
GTCTCCCCCTCCCTGCACACCCAGGGAGCAGAGGGAAGGTTCCTTGCAGGTGGGCAATGAGGCCCCTGTGA
CCGGCTCCTCCCCGCTGGGCGCCACGCAGCTGGACACTGATGGAGCCCTGTGGCTTGGTGAGTGTTTTGGG
GAGACTAGAGAGGGATGCCCAAGGGTCTCATGATATCCGAGGGACAGACTCCACCCCCCAGCGCCCACCCT
TGAGTCAGGGTGCATGTGAGCCGGCGGGCTGGGCTCTCTTCTCCCGCTGTAGCCCCTGCAGTTCCCAGTGC
TGTGGGGCCGGGAGG
    """
    output = tokenizer(gene_data, padding="max_length", max_length=512, truncation=True)
    print(output)
    tokenizer = GenomicTokenizer(model_max_length=512, introns=False)
    output = tokenizer(gene_data, padding="max_length", max_length=512, truncation=True)
    print(output)

def test_tokenize_gene_fna():
	# with open("tests/gene.fna", "r") as file:
	# 	gene_data = file.read().replace("\n", "")
	gene_data = """
AGGCGAGGCGCGGGCGGAGGCGGTGCGCGGGCGGAGGCGGGGCGCGGAGATGTGGCGGAGGTGGAGGCGG
AGGCGTAGCCGCCCCTGGGGACGTCATTGGTGGCGGAAGCAATCGCCGGCAACCAGCTGTAAGCGAGGTA
GGCTCACTCGGGCACGGAGGGTGCGGGTGAGAAAGGGAACGATTTGCTAGGAGTGTATGCGCCCGTGCTA
    """
	tokenizer = GenomicTokenizer(model_max_length=512)
	tokenized_data = tokenizer._tokenize(gene_data)
	print(tokenized_data)
	assert len(tokenized_data) != 0
	#Further assertions can be added based on expected tokenized data characteristics

def test_convert_tokens_to_ids():
    tokenizer = GenomicTokenizer(model_max_length=512)
    tokens = ["GGA", "TGA", "GGG", "CAG"]
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    print(token_ids)
    assert len(token_ids) == len(tokens)
    assert all(isinstance(token_id, int) for token_id in token_ids)

def test_convert_ids_to_tokens():
    tokenizer = GenomicTokenizer(model_max_length=512)
    token_ids = [1, 2, 3, 4]
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    print(tokens)
    assert len(tokens) == len(token_ids)
    assert all(isinstance(token, str) for token in tokens)

def test_save_pretrained():
    tokenizer = GenomicTokenizer(model_max_length=512)
    tokenizer.save_pretrained("tests/")