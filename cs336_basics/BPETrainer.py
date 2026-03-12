import json
import logging
import os
import time
from collections import Counter, defaultdict
from typing import BinaryIO

import regex as re

logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(message)s', force=False)

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
TOKEN_RE = re.compile(PAT)


def _split_text(text: str, special_tokens: list[str]) -> list[str]:
    if not special_tokens:
        return [text]
    # Sort by descending length to preserve the longest overlapping special token.
    pattern = "|".join(re.escape(tok) for tok in sorted(special_tokens, key=len, reverse=True))
    return re.split(pattern, text)


def pre_tokenization_impl(text: str, special_tokens: list[str]) -> Counter[tuple[int, ...]]:
    words: Counter[str] = Counter()
    for chunk in _split_text(text, special_tokens):
        for match in TOKEN_RE.finditer(chunk):
            words[match.group()] += 1

    tokenized_words: Counter[tuple[int, ...]] = Counter()
    for word, count in words.items():
        tokenized_words[tuple(word.encode("utf-8"))] += count
    return tokenized_words


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    if desired_num_chunks <= 1 or file_size == 0:
        return [0, file_size]

    chunk_size = max(file_size // desired_num_chunks, 1)
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    if not split_special_token:
        return sorted(set(chunk_boundaries))

    mini_chunk_size = 4096
    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    return sorted(set(chunk_boundaries))


def _pair_counter(word: tuple[int, ...]) -> Counter[tuple[int, int]]:
    return Counter(zip(word, word[1:]))


def _merge_pair_in_word(word: tuple[int, ...], pair: tuple[int, int], new_token_id: int) -> tuple[int, ...]:
    first, second = pair
    merged: list[int] = []
    i = 0
    changed = False
    word_len = len(word)
    while i < word_len:
        if i + 1 < word_len and word[i] == first and word[i + 1] == second:
            merged.append(new_token_id)
            i += 2
            changed = True
        else:
            merged.append(word[i])
            i += 1
    return tuple(merged) if changed else word


class BPETrainer:
    def __init__(self, input_path, vocab_size, special_tokens):
        self.input_path = input_path
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens or []
        self.vocab: dict[int, bytes] = {}
        self.merges: list[tuple[bytes, bytes]] = []
        self.next_id = 0

        logging.debug(
            f"Init BPE Tokenizer with "
            f"Input Path={self.input_path}\n"
            f"Vocab Size={self.vocab_size}\n"
            f"special Tokens={special_tokens}\n"
        )

        for b in range(256):
            self.vocab[self.next_id] = bytes([b])
            self.next_id += 1

        for tok in self.special_tokens:
            self.vocab[self.next_id] = tok.encode("utf-8")
            self.next_id += 1

    def _count_words(self) -> Counter[tuple[int, ...]]:
        file_size = os.path.getsize(self.input_path)
        num_chunks = min(4, max(1, file_size // (8 * 1024 * 1024) + 1))
        split_token = self.special_tokens[0].encode("utf-8") if self.special_tokens else b""

        with open(self.input_path, "rb") as f:
            boundaries = find_chunk_boundaries(f, num_chunks, split_token)
            global_counts: Counter[tuple[int, ...]] = Counter()
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                if start == end:
                    continue
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8", errors="ignore")
                global_counts.update(pre_tokenization_impl(chunk, self.special_tokens))
        return global_counts

    def train(self) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        tokenized_words = self._count_words()
        return self.merge(tokenized_words)

    def merge(self, word_counts: Counter[tuple[int, ...]]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        start_next_id = self.next_id
        t0 = time.time()
        t_last = t0

        words = list(word_counts.keys())
        counts = [word_counts[word] for word in words]
        pair_counts: dict[tuple[int, int], int] = defaultdict(int)
        pair_to_word_ids: dict[tuple[int, int], set[int]] = defaultdict(set)

        for word_id, word in enumerate(words):
            if len(word) < 2:
                continue
            for pair, occurrences in _pair_counter(word).items():
                pair_counts[pair] += occurrences * counts[word_id]
                pair_to_word_ids[pair].add(word_id)

        while self.next_id < self.vocab_size:
            if not pair_counts:
                break
            best_pair = max(
                pair_counts.items(),
                key=lambda item: (item[1], (self.vocab[item[0][0]], self.vocab[item[0][1]])),
            )[0]

            first, second = best_pair
            merged_bytes = self.vocab[first] + self.vocab[second]
            new_token_id = self.next_id
            self.vocab[new_token_id] = merged_bytes
            self.merges.append((self.vocab[first], self.vocab[second]))
            self.next_id += 1

            affected_word_ids = list(pair_to_word_ids.pop(best_pair, ()))
            pair_counts.pop(best_pair, None)

            for word_id in affected_word_ids:
                old_word = words[word_id]
                new_word = _merge_pair_in_word(old_word, best_pair, new_token_id)
                if new_word == old_word:
                    continue

                word_count = counts[word_id]
                old_pairs = _pair_counter(old_word)
                new_pairs = _pair_counter(new_word)

                for old_pair, occurrences in old_pairs.items():
                    new_total = pair_counts.get(old_pair, 0) - occurrences * word_count
                    if new_total > 0:
                        pair_counts[old_pair] = new_total
                    else:
                        pair_counts.pop(old_pair, None)
                    word_ids = pair_to_word_ids.get(old_pair)
                    if word_ids is not None:
                        word_ids.discard(word_id)
                        if not word_ids:
                            pair_to_word_ids.pop(old_pair, None)

                for new_pair, occurrences in new_pairs.items():
                    pair_counts[new_pair] = pair_counts.get(new_pair, 0) + occurrences * word_count
                    pair_to_word_ids[new_pair].add(word_id)

                words[word_id] = new_word

            added = self.next_id - start_next_id
            if added % 100 == 0:
                now = time.time()
                logging.info(
                    f"added {added} new tokens in {now - t_last:.2f}s (total {now - t0:.2f}s); next_id={self.next_id}"
                )
                t_last = now

        now_total = time.time()
        logging.info(
            f"merge complete: added {self.next_id - start_next_id} new tokens in {now_total - t0:.2f}s"
        )
        return self.vocab, self.merges

    def save_artifacts(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        vocab_serialized = {str(i): self.vocab[i].hex() for i in sorted(self.vocab.keys())}
        with open(os.path.join(output_dir, "vocab.hex.json"), "w", encoding="utf-8") as f:
            json.dump(vocab_serialized, f, ensure_ascii=False, indent=2)

        merges_serialized = [[a.hex(), b.hex()] for (a, b) in self.merges]
        with open(os.path.join(output_dir, "merges.hex.json"), "w", encoding="utf-8") as f:
            json.dump(merges_serialized, f, ensure_ascii=False, indent=2)

        with open(os.path.join(output_dir, "vocab.tsv"), "w", encoding="utf-8") as f:
            f.write("id\tutf8\thex\n")
            for i in sorted(self.vocab.keys()):
                token_bytes = self.vocab[i]
                try:
                    text = token_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    text = token_bytes.decode("utf-8", errors="replace")
                f.write(f"{i}\t{text}\t{token_bytes.hex()}\n")

        with open(os.path.join(output_dir, "merges.txt"), "w", encoding="utf-8") as f:
            for first, second in self.merges:
                try:
                    first_text = first.decode("utf-8")
                except UnicodeDecodeError:
                    first_text = first.decode("utf-8", errors="replace")

                try:
                    second_text = second.decode("utf-8")
                except UnicodeDecodeError:
                    second_text = second.decode("utf-8", errors="replace")

                f.write(f"{first.hex()} ({first_text}) + {second.hex()} ({second_text})\n")


if __name__ == "__main__":
    bpe = BPETrainer(
        input_path="../../data/owt_train.txt",
        vocab_size=32000,
        special_tokens=["<|endoftext|>"],
    )
    vocab, merges = bpe.train()
    base_name = os.path.splitext(os.path.basename(bpe.input_path))[0]
    out_dir = os.path.join(os.path.dirname(bpe.input_path), f"bpe_artifacts_{base_name}")
    bpe.save_artifacts(out_dir)
    print(f"Artifacts saved to {out_dir}")
