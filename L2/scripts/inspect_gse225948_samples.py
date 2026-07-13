"""检查GSE225948样本结构与细胞类型分布"""
import tarfile
import gzip

TAR_PATH = r"C:\Users\Jy-Mentor-7\Desktop\申请书\原始数据\GSE225948_RAW (1).tar"

def main():
    tar = tarfile.open(TAR_PATH, "r")
    members = sorted(
        [m for m in tar.getmembers() if "metadata" in m.name and "Blood" not in m.name],
        key=lambda m: m.name,
    )
    total_cells = 0
    all_subtypes = {}
    all_parents = {}

    print("=" * 90)
    print(f"{'GSM':12s} {'Cond':7s} {'Age':5s} {'Sex':4s} {'Cells':>7s}  Description")
    print("-" * 90)

    for member in members:
        f = tar.extractfile(member)
        subtypes = {}
        parents = {}

        with gzip.open(f, "rt") as gz:
            gz.readline()
            n_rows = 0
            for line in gz:
                line = line.strip()
                if not line:
                    continue
                n_rows += 1
                fields = line.split(",")
                if len(fields) > 13:
                    sub = fields[13].strip('"')
                    parent = fields[14].strip('"')
                    subtypes[sub] = subtypes.get(sub, 0) + 1
                    parents[parent] = parents.get(parent, 0) + 1
                    all_subtypes[sub] = all_subtypes.get(sub, 0) + 1
                    all_parents[parent] = all_parents.get(parent, 0) + 1

        # Parse first row for treatment/age/desc
        f = tar.extractfile(member)
        with gzip.open(f, "rt") as gz:
            gz.readline()
            first_line = gz.readline()
            fields = first_line.strip().split(",")
            treatment = clean(fields, 10)
            age_str = clean(fields, 9)
            sex = clean(fields, 8)
            desc = clean(fields, 12)

        name = member.name.replace(".csv.gz", "")
        parts = name.split("_")
        gsm = parts[0] if parts else ""

        top_sub = sorted(subtypes.items(), key=lambda x: -x[1])[:3]
        top_sub_str = ", ".join(f"{k}:{v}" for k, v in top_sub)

        print(f"{gsm:12s} {treatment:7s} {age_str:5s} {sex:4s} {n_rows:>7d}  [{top_sub_str}]")
        total_cells += n_rows

    print("-" * 90)
    print(f"Total brain cells (CD45hi sorted): {total_cells:,}")
    print()
    print("=== Global Cell Type Distribution (parent) ===")
    for k, v in sorted(all_parents.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v} ({v/total_cells*100:.1f}%)")
    print()
    print("=== Young Brain D14 Stroke (n=4) summary ===")
    d14_cells = []
    for member in members:
        name = member.name
        if "aged" not in name:
            f = tar.extractfile(member)
            with gzip.open(f, "rt") as gz:
                gz.readline()
                for line in gz:
                    line = line.strip()
                    if not line:
                        continue
                    fields = line.split(",")
                    treatment = clean(fields, 10)
                    if treatment == "D14":
                        d14_cells.append(1)

    tar.close()

def clean(fields, idx):
    return fields[idx].strip('"') if len(fields) > idx else "NA"

if __name__ == "__main__":
    main()
