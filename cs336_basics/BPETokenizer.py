import json
import os
import regex as re
from typing import Iterable, Iterator

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
TOKEN_RE = re.compile(PAT)

def pre_tokenization(
        text: str,
        special_tokens: list[str]
) -> list[tuple[bytes, ...]]:
    # Sort special tokens by length (desc) to prefer longest matches when overlapping
    if special_tokens:
        special_tokens_sorted = sorted(special_tokens, key=len, reverse=True)
        st_set = set(special_tokens_sorted)
        pattern = "(" + "|".join(map(re.escape, special_tokens_sorted)) + ")"
        chunks = re.split(pattern, text)
    else:
        st_set = set()
        chunks = [text]

    # Use a sentinel to prevent merges across piece boundaries (regex pieces and specials)
    SENTINEL = None  # type: ignore[assignment]

    tokens: list[object] = []  # elements are bytes or SENTINEL
    for chunk in chunks:
        # If this chunk is exactly a special token, keep it as a single bytes token
        if st_set and chunk in st_set:
            tokens.append(chunk.encode("utf-8"))
            tokens.append(SENTINEL)
            continue
        # Otherwise, run the GPT-2 style regex and break each match into single-byte tokens
        for m in TOKEN_RE.finditer(chunk):
            w = m.group()
            b = w.encode("utf-8")
            for bt in b:
                tokens.append(bytes([bt]))
            # boundary between regex matches to avoid cross-piece merges
            tokens.append(SENTINEL)

    # Drop trailing sentinel if present
    if tokens and tokens[-1] is SENTINEL:
        tokens.pop()

    # Filter out sentinels before returning (merges will see boundaries because equality can't cross None)
    # We keep them during merging by leaving them in the list; the caller expects a list of bytes tokens.
    # Therefore, return the list with sentinels included; the merge routine compares only bytes.
    return tokens  # type: ignore[return-value]

class BPETokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        """
        method that constructs and return a Tokenizer from a serialized vocabulary and list of merges
        (in the same format that your BPE training code output) and (optionally) a list of special
        tokens. This method should accept the following additional parameters:
        vocab_filepath: str
        merges_filepath: str
        special_tokens: list[str] | None = None
        """
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            vocab_hex = json.load(f)
        vocab = {int(i): bytes.fromhex(h) for i, h in vocab_hex.items()}

        with open(merges_filepath, "r", encoding="utf-8") as f:
            merges_hex = json.load(f)
        merges = [(bytes.fromhex(a), bytes.fromhex(b)) for a, b in merges_hex]

        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        """
        Encode an input text into a sequence of token IDs.
        """
        tokens = pre_tokenization(text, self.special_tokens)
        for first, second in self.merges:
            i = 0
            merged_tokens = []
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == first and tokens[i+1] == second:
                    merged_tokens.append(first + second)
                    i += 2
                else:
                    merged_tokens.append(tokens[i])
                    i += 1
            tokens = merged_tokens
        ids = []
        for tok in tokens:
            for k, v in self.vocab.items():
                if v == tok:
                    ids.append(k)
                    break
        return ids


    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs. This is
        required for memory-eﬀicient tokenization of large files that we cannot directly load into
        memory.
        """
        for chunk in iterable:
            tokens = pre_tokenization(chunk, self.special_tokens)

            for first, second in self.merges:
                i = 0
                merged_tokens = []
                while i < len(tokens):
                    if i < len(tokens) - 1 and tokens[i] == first and tokens[i+1] == second:
                        merged_tokens.append(first + second)
                        i += 2
                    else:
                        merged_tokens.append(tokens[i])
                        i += 1
                tokens = merged_tokens
            for tok in tokens:
                for k, v in self.vocab.items():
                    if v == tok:
                        yield k
                        break
        
    def decode(self, ids: list[int]) -> str:
        """
        Decode a sequence of token IDs into text.
        """
        byte_seq = b"".join(self.vocab[i] for i in ids if i in self.vocab)

        return byte_seq.decode("utf-8", errors="replace")