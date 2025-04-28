import collections
import regex as re
import os

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

def process_document(
    document: str,
    preprocess_pattern: str,
    encoding: str,
    initial_vocab_size: int,
    vocab_size: int
) -> tuple[list[bytes], list[tuple[bytes, bytes]]]:
    # Example ['some', ' text', ' that', ' i', "'ll", ' pre', '-', 'tokenize']
    pretokens = re.findall(preprocess_pattern, document)

    # Frequency count
    # This is not enough because we need to do tuple bytes
    # token_map = collections.Counter(pretokens)
    token_map = collections.Counter()
    for pretoken in pretokens:
        # this is the last step of pretokenization
        # to char array tuple('s','o','m','e') where 's' will be some utf-8 encoded number
        pretoken_bytes = [t.encode(encoding) for t in pretoken]
        token_map[tuple(pretoken_bytes)] += 1

    # Merging
    # count frequency of successive bytes
    # e.g. {lo: 7, ow: 7, we: 8, er: 2, wi: 3, id: 3, de: 3, es: 9, st: 9, ne: 6, ew: 6}
    # keep running the merge until we hit the vocab size limit
    all_merges = []
    new_vocab_set = set() # already remembers insertion order
    while initial_vocab_size + len(new_vocab_set) <= vocab_size:
        byte_pairs = collections.Counter()
        byte_pairs_to_pre_merge_token_keys = collections.defaultdict(list)

        # Count byte pairs
        for pre_merge_token, count in token_map.items():
            for i in range(len(pre_merge_token) - 1):
                byte_pair = (pre_merge_token[i], pre_merge_token[i+1])
                # byte_pairs[byte_pair] = count
                # some pre_merge_token may have multiple matches
                byte_pairs[byte_pair] += count
                byte_pairs_to_pre_merge_token_keys[byte_pair].append(pre_merge_token)

        # lexicographically greater pair should be done automatically
        # print(byte_pair)
        byte_pair_to_merge = byte_pairs.most_common(1)[0][0]
        all_merges.append(byte_pair_to_merge)

        # Don't remove newly merged token for now
        # double check if one of the pair is already part of new vocab and delete it
        # for byte_item in byte_pair_to_merge:
        #     if byte_item in new_vocab_set:
        #         new_vocab_set.remove(byte_item)
        new_vocab_set.add(byte_pair_to_merge[0] + byte_pair_to_merge[1])

        for token_key in byte_pairs_to_pre_merge_token_keys[byte_pair_to_merge]:
            # Re-work
            i = 0
            new_token_key = [] # hard to do it in place because of multiple matches
            n = len(token_key)
            while i < n:
                if i < n - 1 and (token_key[i], token_key[i+1]) == byte_pair_to_merge:
                    new_token_key.append(byte_pair_to_merge[0] + byte_pair_to_merge[1])
                    i += 2
                else:
                    # keep the old byte, note that this will handle i == n - 1
                    # if it does not need to be merged.
                    new_token_key.append(token_key[i])
                    i += 1
            
            token_map[tuple(new_token_key)] = token_map[token_key]
            del token_map[token_key]
            
            # Original implementation
            # token_count = token_map[token_key]
            # # remove old key
            # del token_map[token_key]
            # # find merge index
            # token_key_list = list(token_key)
            # for i in range(len(token_key) - 1):
            #     byte_pair = (token_key[i], token_key[i+1])
            #     if byte_pair_to_merge == byte_pair:
            #         # Help me with merging code here
            #         pass
            # merge_index = token_key.find(byte_pair_to_merge)
            # new_key = tuple(
            #     token_key_list[:merge_index] + [byte_pair_to_merge[0], byte_pair_to_merge[1]] + token_key_list[merge_index+1:]
            # )
            # token_map[new_key] = token_count
        
    return list(new_vocab_set), all_merges

def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    num_processes = 8,
    encoding = 'utf-8',
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # Create initial vocab map with 256 utf characters
    vocab = {i: bytes([i]) for i in range(256)}
    cur_vocab_size = len(vocab)
    for special_token in special_tokens:
        cur_vocab_size += 1
        vocab[cur_vocab_size] = special_token

    # Build a regex pattern that matches any special token
    special_tokens_pattern = "|".join(re.escape(token) for token in special_tokens)
    preprocess_pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    merges = []
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f, num_processes, "<|endoftext|>".encode(encoding))
            
        # The following is a serial implementation, but you can parallelize this 
        # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")

            # Run pre-tokenization on your chunk and store the counts for each pre-token
            documents = re.split(special_tokens_pattern, chunk)

            for document in documents:
                # TODO: double check if strip would mess up the spaces later
                document = document.strip()
                if not document:
                    continue

                new_vocabs, new_merges = process_document(
                    document,
                    preprocess_pattern,
                    encoding,
                    cur_vocab_size,
                    vocab_size,
                )

                # Update results
                for new_vocab in new_vocabs:
                    cur_vocab_size += 1
                    vocab[cur_vocab_size] = new_vocab
                merges += new_merges
    return vocab, new_merges


if __name__ == "__main__":
    preprocess_pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    cur_vocab_size = 255
    vocab_size = 255 + 3

    document = "low low low low low lower lower widest widest widest newest newest newest newest newest newest"
    results = process_document(
        document,
        preprocess_pattern,
        "utf-8",
        cur_vocab_size,
        vocab_size
    )
    print(results)

















                    
                

