#!/usr/bin/env python3
"""Batch reclassify torrent titles and files directly using classifier functions via thread pool."""

import argparse
import csv
import time
from concurrent.futures import ThreadPoolExecutor

from bitmagnet_classify.models import TorrentFile, TorrentPayload
from bitmagnet_classify.server import classify_torrent


class TorrentItem:
    __slots__ = ("extensions", "name", "sample_paths")

    def __init__(self, name: str, sample_paths: str = "", extensions: str = ""):
        self.name = name
        self.sample_paths = sample_paths
        self.extensions = extensions

    def to_payload(self) -> TorrentPayload:
        files = []
        if self.sample_paths:
            for p in self.sample_paths.split("|"):
                p = p.strip()
                if not p:
                    continue
                ext = p.rsplit(".", 1)[-1].lower() if "." in p else None
                files.append(TorrentFile(path=p, size=0, extension=ext))

        first_ext = None
        if self.extensions:
            first_ext = self.extensions.split(",")[0].strip() or None

        return TorrentPayload(
            infoHash="0000000000000000000000000000000000000000",
            name=self.name,
            size=0,
            extension=first_ext,
            files=files,
        )


def classify_item(
    item: TorrentItem, early_lang: set[str] | None = None
) -> tuple[TorrentItem, str, str]:
    try:
        verdict = classify_torrent(item.to_payload(), early_lang=early_lang)
        action = str(verdict.action)
        tag_str = ";".join(verdict.tags) if verdict.tags else ""
    except Exception as e:  # noqa: BLE001
        action = "error"
        tag_str = f"error:{type(e).__name__}"
    return item, action, tag_str


def main():
    parser = argparse.ArgumentParser(
        description="Reclassify titles from a CSV file directly via ThreadPoolExecutor"
    )
    parser.add_argument(
        "input_csv",
        nargs="?",
        default="scratch/titles_with_files.csv",
        help="Input CSV file path (default: scratch/titles_with_files.csv)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="scratch/classified.csv",
        help="Output CSV file path (default: scratch/classified.csv)",
    )
    parser.add_argument(
        "-t",
        "--threads",
        type=int,
        default=8,
        help="Number of worker threads (default: 8)",
    )
    parser.add_argument(
        "-l",
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of rows to process",
    )
    parser.add_argument(
        "--early-lang",
        nargs="*",
        default=["ja", "ko", "zh", "ru", "uk", "be", "bg", "sr", "mk"],
        help="Language codes (e.g. ja ko zh) that trigger immediate early-return skipping further checks",
    )
    args = parser.parse_args()

    early_lang = {
        code.strip().lower()
        for item in args.early_lang
        for part in item.replace(",", " ").split()
        if (code := part.strip())
    }

    # Read items
    items: list[TorrentItem] = []
    with open(args.input_csv, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            name = row[0].strip()
            sample_paths = row[1].strip() if len(row) > 1 else ""
            extensions = row[2].strip() if len(row) > 2 else ""
            items.append(TorrentItem(name, sample_paths, extensions))
            if args.limit and len(items) >= args.limit:
                break

    total = len(items)
    print(f"Loaded {total:,} items from {args.input_csv}")
    if early_lang:
        print(f"Early-return languages ({len(early_lang)}): {sorted(early_lang)}")
    print(f"Running in-process with ThreadPoolExecutor ({args.threads} threads)...")

    chunk_size = 500
    processed = 0
    t0 = time.time()

    # Column order: verdict, torrent, [files...]
    with open(args.output, "w", encoding="utf-8", newline="") as out_f:
        writer = csv.writer(out_f)

        classify_fn = (
            (lambda it: classify_item(it, early_lang=early_lang))
            if early_lang
            else classify_item
        )
        with ThreadPoolExecutor(max_workers=args.threads) as executor:
            for i in range(0, total, chunk_size):
                chunk = items[i : i + chunk_size]
                results = list(executor.map(classify_fn, chunk))
                for item, action, tags in results:
                    verdict = f"{action}:{tags}" if tags else action
                    writer.writerow([verdict, item.name, item.sample_paths])
                processed += len(chunk)
                elapsed = time.time() - t0
                rate = processed / elapsed if elapsed > 0 else 0
                if processed % 1000 == 0 or processed == total:
                    pct = processed * 100 // total if total else 100
                    print(
                        f"Progress: {processed:,}/{total:,} ({pct}%) - {rate:,.0f} items/s"
                    )

    elapsed = time.time() - t0
    print(
        f"\nFinished writing {processed:,} classified rows to {args.output} in {elapsed:.1f}s ({processed / elapsed:,.0f} items/s)"
    )


if __name__ == "__main__":
    main()
