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
    # Further assertions can be added based on expected tokenized data characteristics


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


# ============================================================================
# Tests for new HF compatibility methods/properties
# ============================================================================

def test_added_tokens_encoder():
    """Test _added_tokens_encoder contains all 7 special tokens."""
    tokenizer = GenomicTokenizer(model_max_length=512)
    
    # Should have exactly 7 special tokens
    assert len(tokenizer._added_tokens_encoder) == 7
    
    # Verify all special tokens are present with correct IDs
    expected_special_tokens = {
        "[CLS]": 0,
        "[SEP]": 1,
        "[BOS]": 2,
        "[MASK]": 3,
        "[PAD]": 4,
        "[RESERVED]": 5,
        "[UNK]": 6,
    }
    
    for token, expected_id in expected_special_tokens.items():
        assert token in tokenizer._added_tokens_encoder
        assert tokenizer._added_tokens_encoder[token] == expected_id


def test_added_tokens_decoder():
    """Test _added_tokens_decoder contains all 7 special tokens as AddedToken objects."""
    from transformers.tokenization_utils import AddedToken
    
    tokenizer = GenomicTokenizer(model_max_length=512)
    
    # Should have exactly 7 special tokens
    assert len(tokenizer._added_tokens_decoder) == 7
    
    # Verify all special tokens are present as AddedToken objects
    expected_special_tokens = {
        0: "[CLS]",
        1: "[SEP]",
        2: "[BOS]",
        3: "[MASK]",
        4: "[PAD]",
        5: "[RESERVED]",
        6: "[UNK]",
    }
    
    for token_id, expected_token in expected_special_tokens.items():
        assert token_id in tokenizer._added_tokens_decoder
        assert isinstance(tokenizer._added_tokens_decoder[token_id], AddedToken)
        assert tokenizer._added_tokens_decoder[token_id].content == expected_token


def test_vocab_property():
    """Test vocab property exists and is a safe copy."""
    tokenizer = GenomicTokenizer(model_max_length=512)
    
    # vocab property should exist
    assert hasattr(tokenizer, 'vocab')
    
    # Should have same content as _vocab_str_to_int
    assert tokenizer.vocab == tokenizer._vocab_str_to_int
    assert len(tokenizer.vocab) == 71
    
    # Modifying vocab should not affect internal state (it's a copy)
    tokenizer.vocab['TEST_TOKEN'] = 999
    assert 'TEST_TOKEN' not in tokenizer._vocab_str_to_int
    assert tokenizer.vocab_size == 71  # Unchanged


def test_save_vocabulary():
    """Test save_vocabulary method saves vocab.json correctly."""
    import tempfile
    import json
    import os
    
    tokenizer = GenomicTokenizer(model_max_length=512)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save vocabulary
        files = tokenizer.save_vocabulary(tmpdir)
        
        # Should return a tuple with one file path
        assert isinstance(files, tuple)
        assert len(files) == 1
        
        # File should exist
        vocab_file = files[0]
        assert os.path.exists(vocab_file)
        assert vocab_file.endswith('vocab.json')
        
        # Load and verify content
        with open(vocab_file, 'r') as f:
            saved_vocab = json.load(f)
        
        assert len(saved_vocab) == 71
        assert saved_vocab == tokenizer.get_vocab()
        
        # Verify internal state unchanged
        assert tokenizer.vocab_size == 71


def test_decode_method():
    """Test _decode method converts token IDs back to text."""
    tokenizer = GenomicTokenizer(model_max_length=512)
    
    # Test basic decoding
    # Note: ID 11 maps to TTC (synonymous with TTT for Phenylalanine)
    token_ids = [2, 12, 11]  # ATG, GGG, TTC
    decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
    assert isinstance(decoded, str)
    assert 'GGG' in decoded
    assert 'TTC' in decoded  # ID 11 decodes to TTC (synonymous with TTT)
    
    # Test decoding with special tokens included
    decoded_with_special = tokenizer.decode(token_ids, skip_special_tokens=False)
    assert isinstance(decoded_with_special, str)
    
    # Test decoding single token ID
    single_decoded = tokenizer.decode(2, skip_special_tokens=True)
    assert isinstance(single_decoded, str)
    
    # Test decoding with padding tokens
    padded_ids = [4, 4, 2, 12, 11]  # PAD, PAD, ATG, GGG, TTC
    decoded_padded = tokenizer.decode(padded_ids, skip_special_tokens=True)
    assert 'GGG' in decoded_padded
    assert 'TTC' in decoded_padded
    
    # Test that decoding doesn't modify tokenizer state
    original_vocab_size = tokenizer.vocab_size
    tokenizer.decode([2, 12, 11])
    assert tokenizer.vocab_size == original_vocab_size


def test_is_fast_property():
    """Test is_fast property returns False for Python tokenizer."""
    tokenizer = GenomicTokenizer(model_max_length=512)
    
    # Should have is_fast property
    assert hasattr(tokenizer, 'is_fast')
    
    # Should return False (this is a Python tokenizer, not Rust-based)
    assert tokenizer.is_fast == False


def test_pickling_support():
    """Test __getstate__ and __setstate__ enable proper pickling."""
    import pickle
    
    # Create tokenizer with specific configuration
    original = GenomicTokenizer(model_max_length=1024, introns=True, padding_side='right')
    
    # Pickle and unpickle
    pickled = pickle.dumps(original)
    restored = pickle.loads(pickled)
    
    # Verify all state is preserved
    assert restored.vocab_size == original.vocab_size
    assert restored.model_max_length == original.model_max_length
    assert restored.introns == original.introns
    assert restored.padding_side == original.padding_side
    assert restored._vocab_str_to_int == original._vocab_str_to_int
    assert restored._vocab_int_to_str == original._vocab_int_to_str
    
    # Verify functionality is preserved
    test_seq = "ATGGGGTTT"
    original_tokens = original._tokenize(test_seq)
    restored_tokens = restored._tokenize(test_seq)
    assert original_tokens == restored_tokens


def test_hf_compatibility_complete():
    """Comprehensive test of all HF compatibility features."""
    import tempfile
    import pickle
    
    tokenizer = GenomicTokenizer(model_max_length=512)
    
    # 1. Check all required properties exist
    assert hasattr(tokenizer, 'vocab_size')
    assert hasattr(tokenizer, 'vocab')
    assert hasattr(tokenizer, 'is_fast')
    
    # 2. Check all required methods exist
    assert hasattr(tokenizer, 'get_vocab')
    assert hasattr(tokenizer, 'save_vocabulary')
    assert hasattr(tokenizer, '_decode')
    
    # 3. Check added tokens handling
    assert len(tokenizer._added_tokens_encoder) == 7
    assert len(tokenizer._added_tokens_decoder) == 7
    
    # 4. Test round-trip: encode -> decode
    original_text = "ATGGGGTTT"
    encoded = tokenizer(original_text)
    decoded = tokenizer.decode(encoded['input_ids'], skip_special_tokens=True)
    assert isinstance(decoded, str)
    
    # 5. Test save/load cycle
    with tempfile.TemporaryDirectory() as tmpdir:
        tokenizer.save_pretrained(tmpdir)
        loaded = GenomicTokenizer.from_pretrained(tmpdir)
        assert loaded.vocab_size == tokenizer.vocab_size
    
    # 6. Test pickling
    pickled = pickle.dumps(tokenizer)
    unpickled = pickle.loads(pickled)
    assert unpickled.vocab_size == tokenizer.vocab_size
    
    print("✓ All HF compatibility features working correctly")
