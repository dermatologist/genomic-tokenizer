"""
Inspired by: https://github.com/HazyResearch/hyena-dna/blob/main/src/dataloaders/datasets/hg38_char_tokenizer.py
and
CharacterTokenzier: https://github.com/dariush-bahrami/character-tokenizer
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

from transformers.tokenization_utils import AddedToken, PreTrainedTokenizer


class GenomicTokenizer(PreTrainedTokenizer):
    # Define start codons and stop codons
    start_codon = ["ATG"]
    stop_codons = ["TAA", "TAG", "TGA"]
    # Define codons for each amino acid
    codons = {
        7: ["GCT", "GCC", "GCA", "GCG"],  # Alanine
        8: ["TGT", "TGC"],  # Cysteine
        9: ["GAT", "GAC"],  # Aspartic acid
        10: ["GAA", "GAG"],  # Glutamic acid
        11: ["TTT", "TTC"],  # Phenylalanine
        12: ["GGT", "GGC", "GGA", "GGG"],  # Glycine
        13: ["CAT", "CAC"],  # Histidine
        14: ["ATT", "ATC", "ATA"],  # Isoleucine
        15: ["AAA", "AAG"],  # Lysine
        16: ["TTA", "TTG", "CTT", "CTC", "CTA", "CTG"],  # Leucine
        2: ["ATG"],  # Methionine (Start)
        17: ["AAT", "AAC"],  # Asparagine
        18: ["CCT", "CCC", "CCA", "CCG"],  # Proline
        19: ["CAA", "CAG"],  # Glutamine
        20: ["CGT", "CGC", "CGA", "CGG", "AGA", "AGG"],  # Arginine
        21: ["TCT", "TCC", "TCA", "TCG", "AGT", "AGC"],  # Serine
        22: ["ACT", "ACC", "ACA", "ACG"],  # Threonine
        23: ["GTT", "GTC", "GTA", "GTG"],  # Valine
        24: ["TGG"],  # Tryptophan
        25: ["TAT", "TAC"],  # Tyrosine
        1: ["TAA", "TAG", "TGA"],  # Stop
    }
    _vocab_str_to_int = {
        "[CLS]": 0,
        "[SEP]": 1,
        "[BOS]": 2,
        "[MASK]": 3,
        "[PAD]": 4,
        "[RESERVED]": 5,
        "[UNK]": 6,
        "GCT": 7,
        "GCC": 7,
        "GCA": 7,
        "GCG": 7,
        "TGT": 8,
        "TGC": 8,
        "GAT": 9,
        "GAC": 9,
        "GAA": 10,
        "GAG": 10,
        "TTT": 11,
        "TTC": 11,
        "GGT": 12,
        "GGC": 12,
        "GGA": 12,
        "GGG": 12,
        "CAT": 13,
        "CAC": 13,
        "ATT": 14,
        "ATC": 14,
        "ATA": 14,
        "AAA": 15,
        "AAG": 15,
        "TTA": 16,
        "TTG": 16,
        "CTT": 16,
        "CTC": 16,
        "CTA": 16,
        "CTG": 16,
        "ATG": 2,
        "AAT": 17,
        "AAC": 17,
        "CCT": 18,
        "CCC": 18,
        "CCA": 18,
        "CCG": 18,
        "CAA": 19,
        "CAG": 19,
        "CGT": 20,
        "CGC": 20,
        "CGA": 20,
        "CGG": 20,
        "AGA": 20,
        "AGG": 20,
        "TCT": 21,
        "TCC": 21,
        "TCA": 21,
        "TCG": 21,
        "AGT": 21,
        "AGC": 21,
        "ACT": 22,
        "ACC": 22,
        "ACA": 22,
        "ACG": 22,
        "GTT": 23,
        "GTC": 23,
        "GTA": 23,
        "GTG": 23,
        "TGG": 24,
        "TAT": 25,
        "TAC": 25,
        "TAA": 1,
        "TAG": 1,
        "TGA": 1,
    }

    def __init__(
        self,
        model_max_length: int,
        padding_side: str = "left",
        introns: bool = True,  # Whether to include introns in the tokenized output
        **kwargs
    ):
        """Character tokenizer for Hugging Face transformers.
        [UNK] token is used for anything that are not in the codons.
        Args:
                    "[CLS]": 0
                    "[SEP]": 1
                    "[BOS]": 2
                    "[MASK]": 3
                    "[PAD]": 4
                    "[RESERVED]": 5
                    "[UNK]": 6
                an id (starting at 7) will be assigned to each codon.
            model_max_length (int): Model maximum sequence length.
        """
        self.model_max_length = model_max_length
        self.introns = introns
        bos_token = AddedToken("[BOS]", lstrip=False, rstrip=False)
        eos_token = AddedToken("[SEP]", lstrip=False, rstrip=False)
        sep_token = AddedToken("[SEP]", lstrip=False, rstrip=False)
        cls_token = AddedToken("[CLS]", lstrip=False, rstrip=False)
        pad_token = AddedToken("[PAD]", lstrip=False, rstrip=False)
        unk_token = AddedToken("[UNK]", lstrip=False, rstrip=False)

        mask_token = AddedToken("[MASK]", lstrip=True, rstrip=False)

        super().__init__(
            bos_token=bos_token,
            eos_token=sep_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            unk_token=unk_token,
            add_prefix_space=False,
            model_max_length=model_max_length,
            padding_side=padding_side,
            **kwargs,
        )

        self.characters = {}
        for i in self.codons.keys():
            for codon in self.codons[i]:
                self._vocab_str_to_int[codon] = i
                self.characters[codon] = i

        self._vocab_int_to_str = {v: k for k, v in self._vocab_str_to_int.items()}

    @property
    def vocab_size(self) -> int:
        return len(self._vocab_str_to_int)

    def _tokenize(self, text: str) -> List[str]:
        """Tokenizes a gene sequence in FASTA format.

        Args:
            text (str): The gene sequence in FASTA format.

        Returns:
            List[str]: A list of codons (tokens) starting from the first occurrence of a start codon in the text.
        """
        # replace fasta header (line starting with >) if it exists
        if text.startswith(">"):
            text = text.split("\n", 1)[1]
        # Convert the text to uppercase and remove newlines
        text = text.upper().replace("\n", "")

        start_index = self.find_any_substring(text, self.start_codon)
        if start_index == -1:
            # No start codon found, encode the entire sequence
            pass
        else:
            # Start codon found, encode the sequence starting from the first start codon
            text = text[start_index:]

        # Convert the text to a list of codons
        codons = [text[i : i + 3] for i in range(0, len(text), 3)]
        encoded = []
        encode = True

        #  Alrighty, So in short,
        #  special tokens - attend & don’t compute loss
        #  padding - don’t attend & don’t compute loss
        for codon in codons:
            if encode:
                # If the codon is 3 characters long after removing spaces, add it
                if len(codon.strip()) == 3:
                    encoded.append(codon)
            else:
                # Attend & don’t compute loss for introns
                if self.introns:
                    encoded.append(self.unk_token)
            # If a stop codon is found, stop encoding
            if codon in self.stop_codons:
                encode = False
            # If a start codon is found, start encoding
            if codon in self.start_codon:
                encode = True
                encoded.append(codon)
        # Now strip any trailing unknown tokens,
        # They will be padded
        while encoded and encoded[-1] == self.unk_token:
            encoded.pop()
        return encoded

    def _convert_token_to_id(self, token: str) -> int:
        return self._vocab_str_to_int.get(token, self._vocab_str_to_int["[UNK]"])

    def _convert_id_to_token(self, index: int) -> str:
        return self._vocab_int_to_str[index]

    def convert_tokens_to_string(self, tokens):
        return "".join(tokens)

    def find_any_substring(self, string, substring_list):
        """Finds any substring from the list in the given string.

        Args:
            string: The string to search in.
            substring_list: A list of substrings to search for.

        Returns:
            The first substring found, or None if no substring is found.
        """

        for substring in substring_list:
            if string.find(substring) != -1:
                return string.find(substring)
        return -1

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        sep = [self.sep_token_id]
        # cls = [self.cls_token_id]
        result = token_ids_0 + sep
        if token_ids_1 is not None:
            result += token_ids_1 + sep
        return result

    def get_special_tokens_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: bool = False,
    ) -> List[int]:
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0,
                token_ids_1=token_ids_1,
                already_has_special_tokens=True,
            )

        result = ([0] * len(token_ids_0)) + [1]
        if token_ids_1 is not None:
            result += ([0] * len(token_ids_1)) + [1]
        return result

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        sep = [self.sep_token_id]
        # cls = [self.cls_token_id]

        #! result = len(cls + token_ids_0 + sep) * [0]
        result = len(token_ids_0 + sep) * [0]
        if token_ids_1 is not None:
            result += len(token_ids_1 + sep) * [1]
        return result

    def get_config(self) -> Dict:
        _config = {
            "tokenizer_class": self.__class__.__name__,
            "unk_token": self.unk_token,
            "pad_token": self.pad_token,
            "cls_token": self.cls_token,
            "sep_token": self.sep_token,
            "mask_token": self.mask_token,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "model_max_length": self.model_max_length,
        }
        _config["codons"] = self.characters
        return _config

    def get_vocab(self) -> Dict[str, int]:
        """
        Returns the vocabulary as a dictionary of token to index.

        `tokenizer.get_vocab()[token]` is equivalent to `tokenizer.convert_tokens_to_ids(token)` when `token` is in the
        vocab.

        Returns:
            `Dict[str, int]`: The vocabulary.
        """
        return self._vocab_str_to_int

    @classmethod
    def from_config(cls, config: Dict) -> "GenomicTokenizer":
        cfg = {}
        cfg["characters"] = [chr(i) for i in config["char_ords"]]
        cfg["model_max_length"] = config["model_max_length"]
        return cls(**cfg)

    def save_pretrained(self, save_directory: Union[str, os.PathLike], **kwargs):
        cfg_file = Path(save_directory) / "tokenizer_config.json"
        cfg = self.get_config()
        with open(cfg_file, "w") as f:
            json.dump(cfg, f, indent=4)

    @classmethod
    def from_pretrained(cls, save_directory: Union[str, os.PathLike], **kwargs):
        cfg_file = Path(save_directory) / "tokenizer_config.json"
        with open(cfg_file) as f:
            cfg = json.load(f)
        return cls.from_config(cfg)

    def set_start_codon(self, start_codons: List[str]):
        self.start_codon = start_codons
        self.codons[2] = start_codons
        for codon in start_codons:
            self._vocab_str_to_int[codon] = 2

    def set_stop_codons(self, stop_codons: List[str]):
        self.stop_codons = stop_codons
        self.codons[1] = stop_codons
        for codon in stop_codons:
            self._vocab_str_to_int[codon] = 1
