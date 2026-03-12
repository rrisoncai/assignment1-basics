import json
import os
import time
import regex as re
from typing import Iterable, Iterator

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
TOKEN_RE = re.compile(PAT)

def pre_tokenization(
        text: str,
        special_tokens: list[str],
        inv_vocab: dict[bytes, int]
) -> list[int | None]:
    # Sort special tokens by length (desc) to prefer longest matches when overlapping
    if special_tokens:
        special_tokens_sorted = sorted(special_tokens, key=len, reverse=True)
        st_set = set(special_tokens_sorted)
        pattern = "(" + "|".join(map(re.escape, special_tokens_sorted)) + ")"
        chunks = re.split(pattern, text)
    else:
        st_set = set()
        chunks = [text]

    SENTINEL = None  # boundary marker to prevent merges across regex/special chunks

    tokens: list[int | None] = []
    for chunk in chunks:
        # keep special tokens intact as a single bytes token
        if st_set and chunk in st_set:
            tokens.append(inv_vocab[chunk.encode("utf-8")])
            tokens.append(SENTINEL)
            continue

        # GPT-2 regex over the non-special chunk; then break each match into single bytes
        for m in TOKEN_RE.finditer(chunk):
            b = m.group().encode("utf-8")
            # push each byte as a separate token so merges can build multi-byte tokens
            tokens.extend(inv_vocab[bytes([bt])] for bt in b)
            # boundary between regex matches
            tokens.append(SENTINEL)

    # drop trailing sentinel
    if tokens and tokens[-1] is SENTINEL:
        tokens.pop()

    return tokens

class BPETokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self._inv_vocab = {v: k for k, v in vocab.items()}
        self.merges = merges
        self.special_tokens = special_tokens

        # Build pair_rank: (id_a, id_b) -> (rank, merged_id)
        self.pair_rank: dict[tuple[int, int], tuple[int, int]] = {}
        for r, (ba, bb) in enumerate(self.merges):
            ia = self._inv_vocab[ba]
            ib = self._inv_vocab[bb]
            ic = self._inv_vocab[ba + bb]
            self.pair_rank[(ia, ib)] = (r, ic)

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
        # 1) integer pre-tokenization with sentinels
        tokens = pre_tokenization(text, self.special_tokens, self._inv_vocab)

        # 2) split by sentinel (None) into segments of int ids
        ids: list[int] = []
        seg: list[int] = []
        for t in tokens + [None]:  # flush last segment by appending sentinel
            if t is None:
                if seg:
                    ids.extend(self._encode_segment_ids(seg))
                    seg = []
            else:
                seg.append(t)
        return ids

    def _encode_segment_ids(self, base_ids: list[int]) -> list[int]:
        import heapq
        n = len(base_ids)
        if n <= 1:
            return base_ids[:]  # nothing to merge

        ids = base_ids[:]  # mutable copy
        prev = [-1] + list(range(0, n - 1))
        next = list(range(1, n)) + [-1]
        alive = [True] * n
        version = [0] * n

        heap: list[tuple[int, int, int]] = []  # (rank, i, ver)

        def push_pair(i: int) -> None:
            j = next[i]
            if j == -1 or not alive[i] or not alive[j]:
                return
            key = (ids[i], ids[j])
            pr = self.pair_rank.get(key)
            if pr is None:
                return
            rank, _ = pr
            heapq.heappush(heap, (rank, i, version[i]))

        for i in range(0, n - 1):
            push_pair(i)

        while heap:
            rank, i, ver = heapq.heappop(heap)
            j = next[i]
            if j == -1 or not alive[i] or not alive[j]:
                continue
            if version[i] != ver:
                continue  # stale
            pr = self.pair_rank.get((ids[i], ids[j]))
            if pr is None:
                continue
            _, merged_id = pr

            # merge j into i
            ids[i] = merged_id
            alive[j] = False
            nj = next[j]
            next[i] = nj
            if nj != -1:
                prev[nj] = i
            version[i] += 1

            li = prev[i]
            if li != -1 and alive[li]:
                version[li] += 1
                push_pair(li)
            if nj != -1 and alive[nj]:
                version[i] += 1
                push_pair(i)

        # collect alive ids in order
        out: list[int] = []
        k = 0
        while k != -1:
            if alive[k]:
                out.append(ids[k])
            k = next[k]
        return out

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs. This is
        required for memory-eﬀicient tokenization of large files that we cannot directly load into
        memory.
        """
        for chunk in iterable:
            tokens = pre_tokenization(chunk, self.special_tokens, self._inv_vocab)
            seg: list[int] = []
            for t in tokens + [None]:
                if t is None:
                    if seg:
                        for _id in self._encode_segment_ids(seg):
                            yield _id
                        seg = []
                else:
                    seg.append(t)
        
    def decode(self, ids: list[int]) -> str:
        """
        Decode a sequence of token IDs into text.
        """
        byte_seq = b"".join(self.vocab[i] for i in ids if i in self.vocab)

        return byte_seq.decode("utf-8", errors="replace")
    
import os
from typing import BinaryIO


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

if __name__ == "__main__":
    # input_file = "../../data/TinyStoriesV2-GPT4-train.txt"
    input_file = "../../data/owt_valid.txt"
    lines = []


    bpe = BPETokenizer.from_files(
        # vocab_filepath="../../data/bpe_artifacts_TinyStoriesV2-GPT4-train/vocab.hex.json",
        # merges_filepath="../../data/bpe_artifacts_TinyStoriesV2-GPT4-train/merges.hex.json",
        vocab_filepath="../../data/bpe_artifacts_owt_train/vocab.hex.json",
        merges_filepath="../../data/bpe_artifacts_owt_train/merges.hex.json",
        special_tokens=["<|endoftext|>"]
        )

    token_list = []
    with open(input_file, "rb") as f:
        num_processes = 16
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        num_chunks = len(boundaries) - 1

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            # Run pre-tokenization on your chunk and store the counts for each pre-token
            text = chunk
            tik = time.time()
            token_chunk = bpe.encode(text)
            tok = time.time()
            token_list.extend(token_chunk)
            elapsed = tok - tik
            bytes_of_text = len(text.encode("utf-8"))
            print(f"Thoughput is {bytes_of_text / elapsed / 1024 / 1024:.2f} MB/sec")
            print("texts length:", bytes_of_text)
            print("token length:", len(token_chunk))
            print(f"compression ratio: {bytes_of_text / len(token_chunk):.2f}")
    import numpy as np
    os.makedirs("../../data/token_ids", exist_ok=True)
    out_path = os.path.join("../../data/token_ids", os.path.basename(input_file) + ".npy")
    np.save(out_path, np.array(token_list, dtype=np.uint16))
    print(f"Token ids saved to {out_path}")
