
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
from torch.utils.data import Dataset


def tokenize_and_pack(
    tokenizer,
    bio_stream_iter,
    n_bios_total,            # for progress bar + size estimation
    out_path,                # e.g. Path("data/cache/bios_tokens.bin")
    seq_len=512,
    avg_tokens_per_bio=200,  # conservative upper bound for sizing
    batch_size=1024,
):
    """Tokenize every bio in the stream, prefix each with <|endoftext|>,
    concatenate into one long uint16 array, save to disk.

    Returns: (n_tokens, n_sequences) — the dataset shape.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Pre-allocate a generous memmap; we'll truncate at the end.
    cap = n_bios_total * (avg_tokens_per_bio + 1) + seq_len
    arr = np.memmap(out_path, dtype=np.uint16, mode="w+", shape=(cap,))

    write_pos = 0
    batch_texts = []

    def flush():
        nonlocal write_pos
        if not batch_texts:
            return
        # Prepend <|endoftext|> to each bio so they're independent
        prefixed = ["<|endoftext|>" + t for t in batch_texts]
        enc = tokenizer(prefixed, add_special_tokens=False)["input_ids"]
        for ids in enc:
            n = len(ids)
            arr[write_pos:write_pos + n] = ids
            write_pos += n
        batch_texts.clear()

    for _, _, text in tqdm(bio_stream_iter, total=n_bios_total):
        batch_texts.append(text)
        if len(batch_texts) >= batch_size:
            flush()
    flush()

    # Truncate to actual size: clean .bin file with N tokens
    arr.flush()
    del arr
    actual = np.memmap(out_path, dtype=np.uint16, mode="r+", shape=(write_pos,))
    actual.flush()
    # Resize the underlying file
    with open(out_path, "r+b") as f:
        f.truncate(write_pos * 2)   # 2 bytes per uint16

    n_seq = write_pos // seq_len    # drop the trailing partial sequence
    return write_pos, n_seq

#Since we have a condensed library of tokens, we remap them
def build_vocab_remap(token_file_path):
    tokens = np.memmap(token_file_path, dtype=np.uint16, mode="r")

    unique = np.unique(tokens)
    unique = unique.astype(np.int64)

    old_to_new = {old: i for i, old in enumerate(unique)}
    new_to_old = {i: old for i, old in enumerate(unique)}

    return old_to_new, new_to_old, len(unique)

def remap_token_file(in_path, out_path, old_to_new):
    in_path = Path(in_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    src = np.memmap(in_path, dtype=np.uint16, mode="r")

    # build lookup table (fast)
    max_id = int(max(old_to_new.keys()))
    lut = np.zeros(max_id + 1, dtype=np.uint16)

    for k, v in old_to_new.items():
        lut[k] = v

    remapped = lut[src]

    dst = np.memmap(out_path, dtype=np.uint16, mode="w+", shape=src.shape)
    dst[:] = remapped
    dst.flush()

    return out_path

def decode_from_remapped(ids, new_to_old, tokenizer):
    gpt2_ids = [new_to_old[int(i)] for i in ids]
    return tokenizer.decode(gpt2_ids)

class PackedTokenDataset(Dataset):
    """Reads tokens from a uint16 memmap, returns (input_ids, labels)
    chunks of length seq_len. Labels = inputs (causal LM shift handled
    by the model's loss function)."""

    def __init__(self, path, seq_len=512):
        self.tokens = np.memmap(path, dtype=np.uint16, mode="r")
        self.seq_len = seq_len
        self.n_seq = len(self.tokens) // seq_len

    def __len__(self):
        return self.n_seq

    def __getitem__(self, idx):
        chunk = self.tokens[idx * self.seq_len:(idx + 1) * self.seq_len]
        # Cast uint16 -> int64 (PyTorch needs int64 for embeddings)
        ids = torch.from_numpy(chunk.astype(np.int64))
        return {"input_ids": ids, "labels": ids.clone()}
