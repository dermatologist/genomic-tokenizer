"""
From: https://github.com/HazyResearch/hyena-dna/blob/main/src/dataloaders/datasets/hg38_char_tokenizer.py

CharacterTokenzier for HuggingFace Transformers.
This is heavily inspired from CanineTokenizer in transformers package.
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

from transformers.tokenization_utils import AddedToken, PreTrainedTokenizer


class GenomicTokenizer(PreTrainedTokenizer):
    # Define start codons and stop codons
    start_codon = ["ATG"]
    stop_codons = ["TAA", "TAG", "TGA"]
    # Define codons for each amino acid
    codons = {
        7: ["GCT", "GCC", "GCA", "GCG"], # Alanine
        8: ["TGT", "TGC"], # Cysteine
        9: ["GAT", "GAC"], # Aspartic acid
        10: ["GAA", "GAG"], # Glutamic acid
        11: ["TTT", "TTC"], # Phenylalanine
        12: ["GGT", "GGC", "GGA", "GGG"], # Glycine
        13: ["CAT", "CAC"], # Histidine
        14: ["ATT", "ATC", "ATA"], # Isoleucine
        15: ["AAA", "AAG"], # Lysine
        16: ["TTA", "TTG", "CTT", "CTC", "CTA", "CTG"], # Leucine
        2: ["ATG"], # Methionine (Start)
        17: ["AAT", "AAC"], # Asparagine
        18: ["CCT", "CCC", "CCA", "CCG"], # Proline
        19: ["CAA", "CAG"], # Glutamine
        20: ["CGT", "CGC", "CGA", "CGG", "AGA", "AGG"], # Arginine
        21: ["TCT", "TCC", "TCA", "TCG", "AGT", "AGC"], # Serine
        22: ["ACT", "ACC", "ACA", "ACG"], # Threonine
        23: ["GTT", "GTC", "GTA", "GTG"], # Valine
        24: ["TGG"], # Tryptophan
        25: ["TAT", "TAC"], # Tyrosine
        1: ["TAA", "TAG", "TGA"], # Stop
    }

    def __init__(self, model_max_length: int, padding_side: str='left', **kwargs):
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

        self._vocab_str_to_int = {
            "[CLS]": 0,
            "[SEP]": 1,
            "[BOS]": 2,
            "[MASK]": 3,
            "[PAD]": 4,
            "[RESERVED]": 5,
            "[UNK]": 6,
        }
        for i in self.codons.keys():
            for codon in self.codons[i]:
                self._vocab_str_to_int[codon] = i

        self._vocab_int_to_str = {v: k for k, v in self._vocab_str_to_int.items()}

    @property
    def vocab_size(self) -> int:
        return len(self._vocab_str_to_int)

    def _tokenize(self, text: str) -> List[str]:
        # First convert the text to uppercase
        # starting from the first occurrence of a self.start_codon in the text
        # split the text into a list of 3 character long strings
        text = text.upper()
        start_codon = self.start_codon[0]
        start_index = text.find(start_codon)
        if start_index == -1:
            return []
        text = text[start_index:]
        return [text[i : i + 3] for i in range(0, len(text), 3)]

    def _convert_token_to_id(self, token: str) -> int:
        return self._vocab_str_to_int.get(token, self._vocab_str_to_int["[UNK]"])

    def _convert_id_to_token(self, index: int) -> str:
        return self._vocab_int_to_str[index]

    def convert_tokens_to_string(self, tokens):
        return "".join(tokens)

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
        cls = [self.cls_token_id]

        result = len(cls + token_ids_0 + sep) * [0]
        if token_ids_1 is not None:
            result += len(token_ids_1 + sep) * [1]
        return result

    def get_config(self) -> Dict:
        return {
            "char_ords": [ord(ch) for ch in self.characters],
            "model_max_length": self.model_max_length,
        }

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