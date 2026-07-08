"""Download the complete GSE233815 SuperSeries from GEO + processed Seurat objects from Mendeley Data.

Usage:
    python download_gse233815_full.py [--skip-geo] [--skip-mendeley]

Output structure under data/external/GSE233815/:
    geo/
        GSE233811/  (bulk RNA-seq)
        GSE233812/  (scRNA-seq)
        GSE233813/  (snRNA-seq)
        GSE233814/  (spatial transcriptomics)
    mendeley/
        seurat_1stSpatial.rds
        seurat_2ndSpatial.rds
        Seurat_OLs_integrated_annotated.Rds
        Seurat_sn_MCAO_Zucha2023_QCFiltered.Rds
        spatial_seurat_1DP_13_nygen.rds
"""
import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

CHUNK_SIZE = 8 * 1024 * 1024  # 8 MiB per worker chunk
MAX_WORKERS_PER_FILE = 8
MENDELEY_API = "https://data.mendeley.com/api/datasets/gnb2dsjms2/files"

GEO_SUBSERIES = {
    "GSE233811": {
        "type": "bulk RNA-Seq",
        "files": ["GSE233811_RAW.tar", "GSE233811_DESeqNormalizedMatrix.txt.gz"],
    },
    "GSE233812": {
        "type": "scRNA-Seq",
        "files": ["GSE233812_RAW.tar"],
    },
    "GSE233813": {
        "type": "snRNA-Seq",
        "files": ["GSE233813_RAW.tar"],
    },
    "GSE233814": {
        "type": "spatial transcriptomics",
        "files": ["GSE233814_RAW.tar"],
    },
}


def geo_url(gse: str, filename: str) -> str:
    prefix = gse[:6]
    return f"https://ftp.ncbi.nlm.nih.gov/geo/series/{prefix}nnn/{gse}/suppl/{filename}"


def get_total_size(url: str) -> int:
    resp = requests.head(url, timeout=30)
    resp.raise_for_status()
    return int(resp.headers["Content-Length"])


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def merge_parts(target_path: str, part_paths: list[str]) -> None:
    with open(target_path, "wb") as out:
        for part_path in part_paths:
            with open(part_path, "rb") as part:
                while True:
                    block = part.read(1024 * 1024)
                    if not block:
                        break
                    out.write(block)


def download_with_resume(url: str, target: str, expected_size: int | None = None) -> None:
    """Download a single file with multi-threading and resume support."""
    out_dir = os.path.dirname(target)
    os.makedirs(out_dir, exist_ok=True)

    if expected_size is None:
        total_size = get_total_size(url)
    else:
        total_size = expected_size

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
    downloaded_total = sum(os.path.getsize(p) if os.path.exists(p) else 0 for p in part_paths)

    def run_task(task):
        if task is None:
            return 0
        start, end, part_path, idx = task
        print(f"Downloading part {idx}: bytes {start}-{end}")
        mode = "ab" if os.path.exists(part_path) else "wb"
        headers = {"Range": f"bytes={start}-{end}"}
        part_downloaded = 0
        with requests.get(url, headers=headers, stream=True, timeout=180) as resp:
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


def download_geo(base_dir: str) -> None:
    geo_dir = os.path.join(base_dir, "geo")
    for gse, info in GEO_SUBSERIES.items():
        print(f"\n=== GEO {gse} ({info['type']}) ===")
        out_dir = os.path.join(geo_dir, gse)
        for filename in info["files"]:
            url = geo_url(gse, filename)
            target = os.path.join(out_dir, filename)
            try:
                download_with_resume(url, target)
            except Exception as e:
                print(f"FAILED {filename}: {e}")
                raise


def download_mendeley(base_dir: str) -> None:
    print("\n=== Fetching Mendeley Data file list ===")
    resp = requests.get(MENDELEY_API, timeout=30)
    resp.raise_for_status()
    files = resp.json()

    mendeley_dir = os.path.join(base_dir, "mendeley")
    os.makedirs(mendeley_dir, exist_ok=True)

    for f in files:
        filename = f["filename"]
        expected_size = f["size"]
        expected_sha = f["content_details"]["sha256_hash"]
        url = f["content_details"]["download_url"]
        target = os.path.join(mendeley_dir, filename)

        print(f"\n=== Mendeley {filename} ({expected_size / 1024 / 1024:.1f} MiB) ===")
        if os.path.exists(target) and os.path.getsize(target) == expected_size:
            print("File exists, verifying SHA256...")
            actual_sha = sha256_file(target)
            if actual_sha == expected_sha:
                print("SHA256 verified.")
                continue
            else:
                print("SHA256 mismatch, re-downloading...")
                os.remove(target)

        download_with_resume(url, target, expected_size=expected_size)
        actual_sha = sha256_file(target)
        if actual_sha != expected_sha:
            raise RuntimeError(f"SHA256 mismatch for {filename}: expected {expected_sha}, got {actual_sha}")
        print("SHA256 verified.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download complete GSE233815 dataset.")
    parser.add_argument("--skip-geo", action="store_true", help="Skip GEO raw data download")
    parser.add_argument("--skip-mendeley", action="store_true", help="Skip Mendeley processed objects download")
    args = parser.parse_args()

    base_dir = os.path.join("data", "external", "GSE233815")
    os.makedirs(base_dir, exist_ok=True)

    print(f"Download root: {os.path.abspath(base_dir)}")

    if not args.skip_geo:
        download_geo(base_dir)
    if not args.skip_mendeley:
        download_mendeley(base_dir)

    print("\nAll requested downloads complete.")


if __name__ == "__main__":
    main()
