"""
Streaming Dataset from AllenAI Dolma v1.5 sample.
Reference: Section 4.1 (Soldaini et al., 2024)
Direct line-by-line streaming via HTTP and Gzip for maximum speed and zero disk usage.
"""

import gzip
import json
import urllib.request
from typing import List, Iterator
import torch
from torch.utils.data import IterableDataset


def get_dolma_urls(max_shards: int = 30) -> List[str]:
    """Fetches shard URLs for dolma-v1.5-sample, prioritizing highly available c4/wiki/cc shards."""
    manifest_url = "https://huggingface.co/datasets/allenai/dolma/raw/main/urls/v1_5-sample.txt"
    req = urllib.request.Request(manifest_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        all_urls = [u.strip() for u in resp.read().decode("utf-8").splitlines() if u.strip()]
    
    preferred = [u for u in all_urls if any(k in u for k in ['c4', 'wiki', 'cc_en_head', 'reddit', 'pes2o', 'stack'])]
    other = [u for u in all_urls if u not in preferred]
    ordered = preferred + other
    return ordered[:max_shards]


class ShardedStreamingDataset(IterableDataset):
    """
    Direct, ultra-fast, zero-disk streaming dataset over remote JSONL.GZ shards.
    Partitions shards evenly across distributed GPU ranks.
    Loops indefinitely over assigned shards.
    """
    def __init__(self, urls: List[str], tokenizer, seq_len: int = 512, rank: int = 0, world_size: int = 1):
        self.urls = [u for i, u in enumerate(urls) if i % world_size == rank]
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __iter__(self) -> Iterator[dict]:
        buffer = []
        while True:
            for url in self.urls:
                try:
                    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                    with urllib.request.urlopen(req, timeout=60) as resp:
                        with gzip.GzipFile(fileobj=resp) as gz_file:
                            for line in gz_file:
                                if not line:
                                    continue
                                try:
                                    record = json.loads(line.decode("utf-8"))
                                    text = record.get("text", "")
                                    if not text or len(text.strip()) < 10:
                                        continue
                                    token_ids = self.tokenizer.encode(text, add_special_tokens=False)
                                    buffer.extend(token_ids)
                                    while len(buffer) >= self.seq_len:
                                        chunk = buffer[:self.seq_len]
                                        buffer = buffer[self.seq_len:]
                                        yield {
                                            "input_ids": torch.tensor(chunk, dtype=torch.long),
                                            "labels": torch.tensor(chunk, dtype=torch.long)
                                        }
                                except Exception:
                                    continue
                except Exception as e:
                    print(f"[Warning] Issue streaming shard {url}: {e}, advancing to next shard...")
                    continue
