import logging
import os
from typing import BinaryIO

logging.basicConfig(
    level=logging.DEBUG,
    format='[%(levelname)s] %(message)s',
    force=True)

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

            # The following is a serial implementation, but you can parallelize this
            # by sending each start/end pair to a set of processes.
            global_counts: dict[tuple[bytes, ...], int] = {}
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8", errors="ignore")
                # Run pre-tokenization on your chunk and store the counts for each pre-token
                byte_tuple_count = self.pre_tokenization(chunk)

                for k, v in byte_tuple_count.items():
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
        # Init vocab
        import regex as re

        pattern = "|".join(map(re.escape, self.special_tokens))

        # Split corpus into chunks separated by special tokens
        chunks = re.split(pattern, text)

        # Filter out any empty strings and whitespace-only chunks
        chunks = [chunk.strip() for chunk in chunks if chunk.strip()]

        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        word_count = {}
        for chunk in chunks:
            matches = re.finditer(PAT, chunk)
            for m in matches:
                w = m.group()
                word_count[w] = word_count.get(w, 0) + 1

        byte_tuple_count = {
            tuple(ch.encode("utf-8") for ch in word): count for word, count in word_count.items()
        }
        return byte_tuple_count
    
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
            # print("pair to merge:", largest_pair_count[0])

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

text = "low low low low low <|endoftext|> lower lower widest widest widest newest newest newest newest newest newest"

bpe = BPETokenizer(input_path="", vocab_size=260, special_tokens=["<|endoftext|>"])
btc = bpe.pre_tokenization(text)
print(len(bpe.vocab))
vocab, merges = bpe.merge(btc)
print(vocab)
print(merges)