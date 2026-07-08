"""Multi-threaded GEO supplementary file downloader with resume support.

Usage:
    python download_geo_multithread.py GSE233812
    python download_geo_multithread.py GSE233811 GSE233812 GSE233813 GSE233814
"""
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

BASE_URL = "https://ftp.ncbi.nlm.nih.gov/geo/series/{prefix}nnn/{gse}/suppl/{filename}"
CHUNK_SIZE = 8 * 1024 * 1024  # 8 MiB per worker chunk
MAX_WORKERS_PER_FILE = 8

# Extra non-RAW supplementary files of interest per GSE.
EXTRA_FILES = {
    "GSE233811": ["GSE233811_DESeqNormalizedMatrix.txt.gz"],
}


def build_url(gse: str, filename: str) -> str:
    prefix = gse[:6]
    return BASE_URL.format(prefix=prefix, gse=gse, filename=filename)


def get_total_size(url: str) -> int:
    resp = requests.head(url, timeout=30)
    resp.raise_for_status()
    return int(resp.headers["Content-Length"])


def merge_parts(target_path: str, part_paths: list[str]) -> None:
    with open(target_path, "wb") as out:
        for part_path in part_paths:
            with open(part_path, "rb") as part:
                while True:
                    block = part.read(1024 * 1024)
                    if not block:
                        break
                    out.write(block)


def download_file(url: str, target: str) -> None:
    out_dir = os.path.dirname(target)
    os.makedirs(out_dir, exist_ok=True)

    total_size = get_total_size(url)
    print(f"\nURL: {url}")
    print(f"Target: {target}")
    print(f"Total size: {total_size / 1024 / 1024:.2f} MiB")

    if os.path.exists(target) and os.path.getsize(target) == total_size:
        print("File already fully downloaded.")
        return

    if os.path.exists(target):
        os.remove(target)

    num_workers = min(MAX_WORKERS_PER_FILE, max(1, total_size // CHUNK_SIZE))
    chunk_len = total_size // num_workers
    ranges = []
    part_paths = []
    for i in range(num_workers):
        start = i * chunk_len
        end = start + chunk_len - 1 if i < num_workers - 1 else total_size - 1
        part_path = f"{target}.part{i:03d}"
        if os.path.exists(part_path):
            existing = os.path.getsize(part_path)
            expected = end - start + 1
            if existing == expected:
                print(f"Part {i} already complete ({existing / 1024 / 1024:.2f} MiB).")
                ranges.append(None)
                part_paths.append(part_path)
                continue
            elif existing < expected:
                print(f"Part {i} resuming from {existing / 1024 / 1024:.2f} MiB.")
                start += existing
            else:
                os.remove(part_path)
        ranges.append((start, end, part_path, i))
        part_paths.append(part_path)

    start_time = time.time()
    downloaded_total = sum(
        os.path.getsize(p) if os.path.exists(p) else 0 for p in part_paths
    )

    def run_task(task):
        if task is None:
            return 0
        start, end, part_path, idx = task
        print(f"Downloading part {idx}: bytes {start}-{end}")
        mode = "ab" if os.path.exists(part_path) else "wb"
        headers = {"Range": f"bytes={start}-{end}"}
        part_downloaded = 0
        with requests.get(url, headers=headers, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            with open(part_path, mode) as f:
                for block in resp.iter_content(chunk_size=64 * 1024):
                    if block:
                        f.write(block)
                        part_downloaded += len(block)
        return part_downloaded

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(run_task, r): i for i, r in enumerate(ranges)}
        for future in as_completed(futures):
            part_downloaded = future.result()
            downloaded_total += part_downloaded
            elapsed = time.time() - start_time
            speed = downloaded_total / elapsed / 1024 / 1024
            pct = downloaded_total / total_size * 100
            print(
                f"Progress: {pct:.1f}% | "
                f"{downloaded_total / 1024 / 1024:.1f}/{total_size / 1024 / 1024:.1f} MiB | "
                f"{speed:.2f} MiB/s"
            )

    print("Merging parts...")
    merge_parts(target, part_paths)

    print("Cleaning up part files...")
    for part_path in part_paths:
        if os.path.exists(part_path):
            os.remove(part_path)

    final_size = os.path.getsize(target)
    print(f"Done. Final size: {final_size / 1024 / 1024:.2f} MiB")
    if final_size != total_size:
        raise RuntimeError(f"Size mismatch: expected {total_size}, got {final_size}")


def collect_files(gse_list: list[str]) -> list[tuple[str, str, str]]:
    """Return list of (gse, filename, target_path)."""
    tasks = []
    for gse in gse_list:
        gse = gse.strip().upper()
        out_dir = os.path.join("data", "external", gse)
        raw_name = f"{gse}_RAW.tar"
        tasks.append((gse, raw_name, os.path.join(out_dir, raw_name)))
        for extra in EXTRA_FILES.get(gse, []):
            tasks.append((gse, extra, os.path.join(out_dir, extra)))
    return tasks


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <GSE1> [GSE2] ...")
        sys.exit(1)

    gse_list = [a.strip().upper() for a in sys.argv[1:]]
    tasks = collect_files(gse_list)

    print(f"Will download {len(tasks)} file(s) for: {', '.join(gse_list)}")
    for gse, filename, target in tasks:
        url = build_url(gse, filename)
        try:
            download_file(url, target)
        except Exception as e:
            print(f"FAILED to download {filename}: {e}")
            raise

    print("\nAll downloads complete.")


if __name__ == "__main__":
    main()
