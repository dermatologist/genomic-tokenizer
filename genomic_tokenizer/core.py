"""
Copyright 2025 Bell Eapen

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""Genomic tokenizer core utilities.

This module defines the :class:`GenomicTokenizer`, a lightweight
Hugging Face compatible tokenizer that operates on DNA codons (triplets)
and collapses synonymous codons to shared token ids. It optionally
includes intronic regions (segments between stop and subsequent start
codons) as ``[UNK]`` tokens so models can attend to gene structure while
masking loss as needed downstream.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from transformers.tokenization_utils import PreTrainedTokenizer

try:  # transformers >=4.40
    from transformers.tokenization_utils import AddedToken  # type: ignore
except ImportError:  # fallback
    from transformers import AddedToken  # type: ignore


class GenomicTokenizer(PreTrainedTokenizer):
    """Tokenizer for genomic (DNA) sequences at codon granularity.

    The vocabulary contains fixed special tokens plus groups of synonymous
    codons mapped to the same integer id (reflecting the amino acid or
    functional category: start / stop).

    Args:
        model_max_length (int): Maximum allowable sequence length (used
            by downstream padding/truncation logic in HF pipelines).
        padding_side (str, optional): Padding side; ``"left"`` or
            ``"right"``. Defaults to ``"left"``.
        introns (bool, optional): If True (default) codons appearing
            after a stop codon and before the next start codon are
            emitted as ``[UNK]``. If False they are skipped entirely.
        **kwargs: Additional keyword arguments passed to
            :class:`transformers.PreTrainedTokenizer`.
    """

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
        **kwargs,
    ):
        """Initialize tokenizer and build codon vocabulary.

        Populates the internal mapping from codon strings to integer ids
        (sharing ids among synonymous codons) and registers special
        tokens with Hugging Face infrastructure.

        Args:
            model_max_length (int): Maximum sequence length expected.
            padding_side (str, optional): Side to apply padding on.
            introns (bool, optional): Retain intronic regions as ``[UNK]``.
            **kwargs: Extra keyword args forwarded to parent class.
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
        """Return size of vocabulary (special + codon tokens).

        Returns:
            int: Number of distinct string tokens recognized.
        """
        return len(self._vocab_str_to_int)

    def _tokenize(self, text: str) -> List[str]:  # type: ignore[override]
        """Convert raw DNA text (optionally FASTA) into codon tokens.

        Processing steps:
            1. Drop FASTA header line (starting with ">") if present.
            2. Uppercase and strip newlines.
            3. Find first start codon; begin at that index if found.
            4. Segment sequence into non-overlapping triplets.
            5. Append codons while in an active (post-start, pre-stop) region.
            6. After a stop codon, either append ``[UNK]`` (if ``introns``)
               or skip codons until next start codon.
            7. Trim trailing ``[UNK]`` tokens (these will be padded later).

        Args:
            text (str): Raw DNA sequence or FASTA formatted string.

        Returns:
            list[str]: Ordered list of codon and special token strings.
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
        """Map a token string to its integer id.

        Args:
            token (str): Codon or special token string.

        Returns:
            int: Token id (``[UNK]`` id if not found).
        """
        return self._vocab_str_to_int.get(token, self._vocab_str_to_int["[UNK]"])

    def _convert_id_to_token(self, index: int) -> str:
        """Map an integer id back to its token string.

        Args:
            index (int): Token id.

        Returns:
            str: Token string.
        """
        return self._vocab_int_to_str[index]

    def convert_tokens_to_string(self, tokens):
        """Concatenate token list into a single string.

        Note: This is a naive join and does not reverse codon merging or
        restore original FASTA formatting.

        Args:
            tokens (list[str]): Token sequence to concatenate.

        Returns:
            str: Concatenated string.
        """
        return "".join(tokens)

    def find_any_substring(self, string, substring_list):
        """Return index of first occurrence of any candidate substring.

        Args:
            string (str): Text to search.
            substring_list (list[str]): Substrings to test.

        Returns:
            int: Start index of first match; -1 if none found.
        """

        for substring in substring_list:
            if string.find(substring) != -1:
                return string.find(substring)
        return -1

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):  # type: ignore[override]
        """Combine one or two sequences and append ``[SEP]`` token(s).

        Args:
            token_ids_0 (List[int]): First sequence ids.
            token_ids_1 (List[int] | None): Second sequence ids.

        Returns:
            List[int]: Concatenated ids with trailing/intermediate ``[SEP]``.
        """
        sep_id = self.sep_token_id
        if token_ids_1 is None:
            return list(token_ids_0) + [sep_id]
        return list(token_ids_0) + [sep_id] + list(token_ids_1) + [sep_id]

    def get_special_tokens_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: bool = False,
    ) -> List[int]:
        """Create mask marking special tokens with 1.

        Args:
            token_ids_0 (list[int]): First sequence token ids.
            token_ids_1 (list[int], optional): Second sequence token ids.
            already_has_special_tokens (bool): If True, defer to parent
                implementation assuming special tokens already present.

        Returns:
            list[int]: Parallel mask of 0/1 values.
        """
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
    ) -> List[int]:  # type: ignore[override]
        """Return token type (segment) ids aligned to input sequences.

        The first (or only) segment receives id 0; the second (if any)
        receives id 1.

        Args:
            token_ids_0 (list[int]): First sequence token ids.
            token_ids_1 (list[int], optional): Second sequence token ids.

        Returns:
            list[int]: Segment id list.
        """
        sep = [self.sep_token_id]
        # cls = [self.cls_token_id]

        #! result = len(cls + token_ids_0 + sep) * [0]
        result = len(token_ids_0 + sep) * [0]
        if token_ids_1 is not None:
            result += len(token_ids_1 + sep) * [1]
        return result

    def get_config(self) -> Dict:
        """Return a serializable configuration dictionary.

        Returns:
            dict: Minimal config for saving/loading tokenizer.
        """
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
        """Return the string token to integer id mapping.

        Returns:
            dict[str, int]: Vocabulary dictionary.
        """
        return self._vocab_str_to_int

    @classmethod
    def from_config(cls, config: Dict) -> "GenomicTokenizer":
        """Instantiate tokenizer from config dictionary.

        Note: Expects keys produced by :meth:`get_config`. Currently only
        `model_max_length` is restored; other dynamic codon changes must be
        applied via setters after loading if desired.

        Args:
            config (dict): Configuration produced by :meth:`get_config`.

        Returns:
            GenomicTokenizer: New tokenizer instance.
        """
        model_max_length = config.get("model_max_length", 512)
        return cls(model_max_length=model_max_length)

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        legacy_format: Optional[bool] = None,
        filename_prefix: Optional[str] = None,
        push_to_hub: bool = False,
        **kwargs,
    ) -> Tuple[str]:  # type: ignore[override]
        """Persist tokenizer configuration to a directory.

        Args:
            save_directory (str | os.PathLike): Target directory path.
            **kwargs: Unused extra save arguments for interface parity.
        """
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        cfg_file = save_dir / "tokenizer_config.json"
        cfg = self.get_config()
        with open(cfg_file, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=4)
        # Return tuple of saved file paths per HF convention
        return (str(cfg_file),)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        *init_inputs,
        **kwargs,
    ) -> "GenomicTokenizer":  # type: ignore[override]
        """Load tokenizer from directory or identifier.

        Args:
            pretrained_model_name_or_path (str | os.PathLike): Local path or
                identifier containing ``tokenizer_config.json``.
            *init_inputs: Unused, for HF API compatibility.
            **kwargs: Ignored extra keyword args.

        Returns:
            GenomicTokenizer: Loaded tokenizer.
        """
        cfg_file = Path(pretrained_model_name_or_path) / "tokenizer_config.json"
        with open(cfg_file, encoding="utf-8") as f:
            cfg = json.load(f)
        return cls.from_config(cfg)

    def set_start_codon(self, start_codons: List[str]):
        """Set (override) list of start codon triplets.

        Args:
            start_codons (list[str]): New start codon sequences.
        """
        self.start_codon = start_codons
        self.codons[2] = start_codons
        for codon in start_codons:
            self._vocab_str_to_int[codon] = 2

    def set_stop_codons(self, stop_codons: List[str]):
        """Set (override) list of stop codon triplets.

        Args:
            stop_codons (list[str]): New stop codon sequences.
        """
        self.stop_codons = stop_codons
        self.codons[1] = stop_codons
        for codon in stop_codons:
            self._vocab_str_to_int[codon] = 1
