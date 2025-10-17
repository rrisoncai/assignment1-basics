import logging
import os
from typing import BinaryIO
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
import regex as re

logging.basicConfig(
    level=logging.DEBUG,
    format='[%(levelname)s] %(message)s',
    force=False)

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
TOKEN_RE = re.compile(PAT)

def pre_tokenization_impl(
        text: str,
        special_tokens: list[str]
) -> dict[tuple[bytes, ...], int]:
    if special_tokens:
        pattern = "|".join(map(re.escape, special_tokens))
        chunks = re.split(pattern, text)
    else:
        chunks = [text]

    word_count: dict[str, int] = {}
    for chunk in chunks:
        for m in TOKEN_RE.finditer(chunk):
            w = m.group()
            word_count[w] = word_count.get(w, 0) + 1

    byte_tuple_count: dict[tuple[bytes, ...], int] = {}
    for word, count in word_count.items():
        b = word.encode("utf-8")
        tup = tuple(bytes([bt]) for bt in b)
        byte_tuple_count[tup] = byte_tuple_count.get(tup, 0) + count
    return byte_tuple_count

def _count_chunk_worker(
    args: tuple[str, int, int, list[str]]
) -> dict[tuple[bytes, ...], int]:
    input_path, start, end, special_tokens = args
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        return pre_tokenization_impl(chunk, special_tokens)

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

class BPETokenizer:
    def __init__(self, input_path, vocab_size, special_tokens):
        self.input_path = input_path
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.vocab = {}
        self.merges = []
        self.next_id = 0
        logging.debug(f"Init BPE Tokenizer with "
                      f"Input Path={self.input_path}\n"
                      f"Vocab Size={self.vocab_size}\n"
                      f"special Tokens={special_tokens}\n")
        for b in range(256):
            self.vocab[self.next_id] = bytes([b])
            self.next_id += 1

        for tok in self.special_tokens:
            self.vocab[self.next_id] = tok.encode("utf-8")
            self.next_id += 1

    def train(
            self,
    ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        with open(self.input_path, "rb") as f:
            num_processes = 4
            boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

            tasks = [
                (self.input_path, start, end, self.special_tokens)
                for start, end, in zip(boundaries[:-1], boundaries[1:])
            ]
            global_counts: dict[tuple[bytes, ...], int] = {}
            with ProcessPoolExecutor(max_workers=num_processes) as ex:
                for local_counts in ex.map(_count_chunk_worker, tasks):
                    for k, v in local_counts.items():
                        global_counts[k] = global_counts.get(k, 0) + v

            return self.merge(global_counts)
    
    def pre_tokenization(
            self,
            text: str
    ) -> dict[tuple[bytes, ...], int]:
        """
        Perform pre-tokenization on a text chunk:
        - Remove special tokens
        - Count occurrence of each byte tuple

        Args:
            text (str): Text segment

        Returns:
            dict[tuple[bytes, ...], int]: A freq map of byte level tokens
        """
        return pre_tokenization_impl(text, self.special_tokens)
    
    def merge(
            self,
            byte_count: dict[tuple[bytes, ...], int]
    ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        # Merge
        while self.next_id < self.vocab_size:
            byte_pair_count = {}
            for tup, count in byte_count.items():
                pairs = [(tup[i], tup[i+1]) for i in range(len(tup)-1)]

                for p in pairs:
                    byte_pair_count[p] = byte_pair_count.get(p, 0) + count

            largest_pair_count = max(byte_pair_count.items(), key=lambda x:(x[1],x[0]))

            first, second = largest_pair_count[0]
            merged_byte = b"".join((first, second))
            self.merges.append((first, second))
            self.vocab[self.next_id] = merged_byte
            self.next_id += 1

            merged_tuple_count = {}
            for tup, count in byte_count.items():
                new_tuple = []
                i = 0
                while i < len(tup):
                    if i < len(tup) - 1 and tup[i] == first and tup[i + 1] == second:
                        new_tuple.append(merged_byte)
                        i += 2
                    else:
                        new_tuple.append(tup[i])
                        i += 1
                new_tuple = tuple(new_tuple)
                merged_tuple_count[new_tuple] = merged_tuple_count.get(new_tuple, 0) + count
            byte_count = merged_tuple_count
        
        return self.vocab, self.merges

# TEST CODE
if __name__ == "__main__":
    bpe = BPETokenizer(input_path="../../data/TinyStoriesV2-GPT4-valid.txt", vocab_size=1000, special_tokens=["<|endoftext|>"])
    bpe.train()