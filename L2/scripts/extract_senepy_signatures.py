"""
[DEPRECATED] Extract SenePy gene signatures from GitHub-hosted pickle files.

SUPERSEDED BY: save_senepy_universal_signatures.py
- This script attempts to download SenePy pickle files from GitHub and parse them
  with pandas, but fails due to (a) network timeouts on raw.githubusercontent.com,
  (b) numpy/pandas DLL compatibility issues in the current conda environment.
- save_senepy_universal_signatures.py uses only Python built-in modules (csv) and
  embeds the pre-computed SenePy universal signatures directly, providing a
  reliable alternative.

RETAINED FOR REFERENCE: This script may be useful if the full cell-type-specific
SenePy signatures (not just universal) are needed in the future, and the
environment issues are resolved (e.g., Docker or fresh conda environment).

SenePy: Sanborn MA et al., Nature Communications (2025), DOI: 10.1038/s41467-025-57047-7
"""

import os
import sys
import pickle
import traceback
import logging
import urllib.request
import socket
from datetime import datetime

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "extract_senepy_signatures.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

BASE_URL = "https://raw.githubusercontent.com/jaleesr/SenePy/main/senepy/senepy/data"
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "senepy")
os.makedirs(OUT_DIR, exist_ok=True)

socket.setdefaulttimeout(30)


def download_file(url, local_path):
    logger.info("Downloading %s ...", os.path.basename(url))
    try:
        urllib.request.urlretrieve(url, local_path)
        size = os.path.getsize(local_path)
        logger.info("  Saved %d bytes", size)
        return True
    except Exception as e:
        logger.error("  Download failed: %s", e)
        return False


def main():
    logger.info("=" * 60)
    logger.info("Extract SenePy signatures from GitHub")
    logger.info("Start time: %s", datetime.now().isoformat())

    files_to_download = {
        "mouse": {
            "dict": "5_TMS_HUBS_DICTIONARY_FILTERED.pickle",
            "meta": "5_TMS_HUBS_METADATA_FILTERED.pickle",
        },
        "human": {
            "dict": "6_TMS_HUBS_DICTIONARY_HUMAN.pickle",
            "meta": "6_TMS_HUBS_METADATA_HUMAN.pickle",
        },
    }

    all_results = {}

    for species, file_info in files_to_download.items():
        logger.info("\n--- Processing %s signatures ---", species)

        dict_file = file_info["dict"]
        dict_url = f"{BASE_URL}/{dict_file}"
        dict_local = os.path.join(OUT_DIR, dict_file)

        if not os.path.exists(dict_local):
            ok = download_file(dict_url, dict_local)
            if not ok:
                logger.error("Cannot download %s dict, skipping", species)
                continue

        meta_file = file_info["meta"]
        meta_url = f"{BASE_URL}/{meta_file}"
        meta_local = os.path.join(OUT_DIR, meta_file)

        if not os.path.exists(meta_local):
            download_file(meta_url, meta_local)

        try:
            with open(dict_local, "rb") as f:
                hubs_dict = pickle.load(f)
            with open(meta_local, "rb") as f:
                hubs_meta = pickle.load(f)
        except Exception as e:
            logger.error("Failed to load pickle: %s", e)
            traceback.print_exc()
            continue

        logger.info("Loaded %d hubs, %d metadata rows", len(hubs_dict), len(hubs_meta))

        rows = []
        for hub_key, hub_data in hubs_dict.items():
            if isinstance(hub_key, tuple) and len(hub_key) >= 3:
                tissue, cell_type, hub_num = hub_key[0], hub_key[1], hub_key[2]
            else:
                tissue, cell_type, hub_num = str(hub_key), "", 0

            genes = hub_data.get("genes", [])
            importances = hub_data.get("importances", [])

            if not genes:
                continue

            if importances and len(importances) == len(genes):
                for g, imp in zip(genes, importances):
                    rows.append({
                        "tissue": tissue,
                        "cell_type": cell_type,
                        "hub_num": hub_num,
                        "gene": g,
                        "importance": imp,
                        "species": species,
                    })
            else:
                for g in genes:
                    rows.append({
                        "tissue": tissue,
                        "cell_type": cell_type,
                        "hub_num": hub_num,
                        "gene": g,
                        "importance": 1.0,
                        "species": species,
                    })

        import pandas as pd
        df = pd.DataFrame(rows)
        csv_path = os.path.join(OUT_DIR, f"senepy_{species}_signatures.csv")
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        logger.info("Exported %d gene entries to %s", len(df), csv_path)

        summary = df.groupby(["tissue", "cell_type"]).agg(
            n_genes=("gene", "nunique"),
            n_hubs=("hub_num", "nunique"),
        ).reset_index()
        summary_path = os.path.join(OUT_DIR, f"senepy_{species}_summary.csv")
        summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
        logger.info("Summary: %d tissue-cell_type combinations", len(summary))

        tissue_counts = df["tissue"].value_counts().to_dict()
        logger.info("Tissues: %s", tissue_counts)

        all_results[species] = {
            "n_hubs": len(hubs_dict),
            "n_genes": df["gene"].nunique(),
            "n_tissue_celltype": len(summary),
            "tissues": list(tissue_counts.keys()),
        }

    logger.info("\n" + "=" * 60)
    logger.info("Summary:")
    for species, info in all_results.items():
        logger.info("  %s: %d hubs, %d unique genes, %d tissue-cell_type pairs",
                    species, info["n_hubs"], info["n_genes"], info["n_tissue_celltype"])
        logger.info("    Tissues: %s", ", ".join(info["tissues"]))

    logger.info("\nOutput directory: %s", OUT_DIR)
    logger.info("End time: %s", datetime.now().isoformat())
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        logger.error(traceback.format_exc())
        sys.exit(1)