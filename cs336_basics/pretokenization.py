import collections
import regex as re
import os
import time

from concurrent.futures import ProcessPoolExecutor
from typing import BinaryIO

def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

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

def pretokenize_document(document: str, preprocess_pattern: str, encoding: str) -> list[list[bytes]]:
    pretokens = re.findall(preprocess_pattern, document)
    token_streams = []
    for pretoken in pretokens:
        token_bytes = pretoken.encode(encoding)
        token_stream = [bytes([b]) for b in token_bytes]
        token_streams.append(token_stream)
    return token_streams

def get_pairs(stream):
    pairs = collections.Counter()
    for tokens in stream:
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i+1])
            pairs[pair] += 1
    return pairs

def merge_vocab_stream(token_streams, initial_vocab_size, vocab_size):
    all_merges, new_vocab_set = [], set()

    # Step 1: Initialize frequencies
    pair_freq = collections.Counter()
    for tokens in token_streams:
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i+1])
            pair_freq[pair] += 1

    while initial_vocab_size + len(new_vocab_set) < vocab_size:
        if not pair_freq:
            break

        # Find the most frequent pairs
        max_freq = max(pair_freq.values())
        candidates = [p for p, f in pair_freq.items() if f == max_freq]
        # max() will pick lexicographically automatically.
        best_pair = max(candidates)  

        merged = best_pair[0] + best_pair[1]
        all_merges.append(best_pair)
        new_vocab_set.add(merged)

        # Rebuild streams and update frequencies
        # we have to use a new stream
        new_token_streams = []
        new_pair_freq = collections.Counter()

        for tokens in token_streams:
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == best_pair:
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            new_token_streams.append(new_tokens)

            # IMPORTANT: we need to re-calculate pair frequency
            for j in range(len(new_tokens) - 1):
                p = (new_tokens[j], new_tokens[j+1])
                new_pair_freq[p] += 1

        token_streams = new_token_streams
        pair_freq = new_pair_freq

    return list(new_vocab_set), all_merges

# This is not working
def merge_vocab_stream_v2(token_streams, initial_vocab_size, vocab_size):
    all_merges = []
    new_vocab_set = set()

    # Initialize pair frequencies and pair positions
    pair_freq = collections.Counter()
    pair_positions = collections.defaultdict(list)

    for stream_id, tokens in enumerate(token_streams):
        for i in range(len(tokens) - 1):
            p = (tokens[i], tokens[i+1])
            pair_freq[p] += 1
            pair_positions[p].append((stream_id, i))

    while initial_vocab_size + len(new_vocab_set) < vocab_size:
        if not pair_freq:
            break

        # Find the best pair
        max_freq = max(pair_freq.values())
        candidates = [p for p, f in pair_freq.items() if f == max_freq]
        best_pair = max(candidates)

        merged_token = best_pair[0] + best_pair[1]
        all_merges.append(best_pair)
        new_vocab_set.add(merged_token)

        locations = list(pair_positions.pop(best_pair, []))

        for (stream_id, idx) in sorted(locations, key=lambda x: x[1]):
            tokens = token_streams[stream_id]
            if idx >= len(tokens) - 1:
                continue
            if tokens[idx] != best_pair[0] or tokens[idx+1] != best_pair[1]:
                continue

            # Remove left neighbor
            if idx > 0:
                left = (tokens[idx-1], tokens[idx])
                pair_freq[left] -= 1
                if (stream_id, idx-1) in pair_positions[left]:
                    pair_positions[left].remove((stream_id, idx-1))

            # Remove right neighbor
            if idx < len(tokens) - 2:
                right = (tokens[idx+1], tokens[idx+2])
                pair_freq[right] -= 1
                if (stream_id, idx+1) in pair_positions[right]:
                    pair_positions[right].remove((stream_id, idx+1))

            # Merge
            tokens[idx] = merged_token
            del tokens[idx+1]

            # Add new neighbors
            if idx > 0:
                new_left = (tokens[idx-1], tokens[idx])
                pair_freq[new_left] += 1
                pair_positions[new_left].append((stream_id, idx-1))

            if idx < len(tokens) - 1:
                new_right = (tokens[idx], tokens[idx+1])
                pair_freq[new_right] += 1
                pair_positions[new_right].append((stream_id, idx))

            # Shift all positions to the right of idx by -1
            for p in list(pair_positions.keys()):
                new_list = []
                for (sid, pos) in pair_positions[p]:
                    if sid == stream_id and pos > idx:
                        new_list.append((sid, pos-1))
                    else:
                        new_list.append((sid, pos))
                pair_positions[p] = new_list

        if best_pair in pair_freq:
            del pair_freq[best_pair]

    return list(new_vocab_set), all_merges

# def run_train_bpe_serial(input_path: str | os.PathLike,
#                   vocab_size: int,
#                   special_tokens: list[str],
#                   encoding: str = 'utf-8',
#                   num_processes: int = 16) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
#     vocab = {i: bytes([i]) for i in range(256)}
#     cur_vocab_size = 256

#     for special_token in special_tokens:
#         vocab[cur_vocab_size] = special_token.encode(encoding)
#         cur_vocab_size += 1

#     special_tokens_pattern = "|".join(re.escape(token) for token in special_tokens)
#     preprocess_pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

#     token_streams = []

#     with open(input_path, "rb") as f:
#         boundaries = find_chunk_boundaries(
#             f, num_processes, "<|endoftext|>".encode("utf-8"))
            
#         # The following is a serial implementation, but you can parallelize this 
#         # by sending each start/end pair to a set of processes.
#         for start, end in zip(boundaries[:-1], boundaries[1:]):
#             f.seek(start)
#             chunk = f.read(end - start).decode("utf-8", errors="ignore")
#             # Run pre-tokenization on your chunk and store the counts for each pre-token
#             documents = re.split(special_tokens_pattern, chunk)

#             for document in documents:
#                 document = document.strip()
#                 if not document:
#                     continue
#                 token_streams.extend(pretokenize_document(document, preprocess_pattern, encoding))

#     new_vocab_set, merges = merge_vocab_stream(token_streams, cur_vocab_size, vocab_size)

#     for new_vocab in new_vocab_set:
#         vocab[cur_vocab_size] = new_vocab
#         cur_vocab_size += 1

#     return vocab, merges

def _process_chunk(
    chunk,
    special_tokens_pattern,
    preprocess_pattern,
    encoding
):
    token_streams = []
    documents = re.split(special_tokens_pattern, chunk)

    for document in documents:
        # this will remove \n
        # document = document.strip()
        if not document:
            continue
        token_streams.extend(pretokenize_document(document, preprocess_pattern, encoding))
    
    return token_streams

def run_train_bpe(input_path: str | os.PathLike,
                  vocab_size: int,
                  special_tokens: list[str],
                  encoding: str = 'utf-8',
                  num_processes: int = 32) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab = {i: bytes([i]) for i in range(256)}
    cur_vocab_size = 256

    for special_token in special_tokens:
        vocab[cur_vocab_size] = special_token.encode(encoding)
        cur_vocab_size += 1

    special_tokens_pattern = "|".join(re.escape(token) for token in special_tokens)
    preprocess_pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    token_streams = []
    chunks = []
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f, num_processes, "<|endoftext|>".encode("utf-8"))
            
        # Parallelize this by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            chunks.append(chunk)

        start_time = time.time()
        # Run pre-tokenization on your chunk and store the counts for each pre-token
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = []
            for chunk in chunks:
                futures.append(
                    executor.submit(
                        _process_chunk,
                        chunk,
                        special_tokens_pattern,
                        preprocess_pattern,
                        encoding
                    )
                )

            for future in futures:
                token_streams.extend(future.result())
        end_time = time.time()
        print("Parallel tasks time: ", end_time - start_time)

    start_time = time.time()
    new_vocab_set, merges = merge_vocab_stream(token_streams, cur_vocab_size, vocab_size)

    for new_vocab in new_vocab_set:
        vocab[cur_vocab_size] = new_vocab
        cur_vocab_size += 1
    end_time = time.time()
    print("Merge time: ", end_time - start_time)

    return vocab, merges

