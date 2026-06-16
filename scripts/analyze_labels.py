from pathlib import Path
from collections import Counter

audio_dir = Path("data/audio")
label_counts = Counter()
cycle_counts = []
mixed_files = 0

for txt_path in sorted(audio_dir.glob("*.txt")):
    has_crackles = False
    has_wheezes = False
    cycles = 0
    per_cycle_labels = []
    for line in txt_path.read_text().strip().splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        c, w = int(parts[2]), int(parts[3])
        cycles += 1
        if c and w:
            per_cycle_labels.append("both")
        elif c:
            per_cycle_labels.append("crackles")
        elif w:
            per_cycle_labels.append("wheezes")
        else:
            per_cycle_labels.append("normal")
        if c:
            has_crackles = True
        if w:
            has_wheezes = True

    cycle_counts.append(cycles)
    if has_crackles and has_wheezes:
        label_counts["both"] += 1
    elif has_crackles:
        label_counts["crackles"] += 1
    elif has_wheezes:
        label_counts["wheezes"] += 1
    else:
        label_counts["normal"] += 1

    if len(set(per_cycle_labels)) > 1:
        mixed_files += 1

print(f"Total files: {len(cycle_counts)}")
print(f"Per-file labels: {dict(label_counts)}")
print(f"Cycles per file: min={min(cycle_counts)}, max={max(cycle_counts)}, avg={sum(cycle_counts)/len(cycle_counts):.1f}")
print(f"Total cycles: {sum(cycle_counts)}")
print(f"Files with MIXED per-cycle labels: {mixed_files} ({100*mixed_files/len(cycle_counts):.0f}%)")
print()

# Per-cycle label distribution
all_cycle_labels = []
for txt_path in sorted(audio_dir.glob("*.txt")):
    for line in txt_path.read_text().strip().splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        c, w = int(parts[2]), int(parts[3])
        if c and w:
            all_cycle_labels.append("both")
        elif c:
            all_cycle_labels.append("crackles")
        elif w:
            all_cycle_labels.append("wheezes")
        else:
            all_cycle_labels.append("normal")

cycle_dist = Counter(all_cycle_labels)
total = len(all_cycle_labels)
print("Per-cycle label distribution:")
for label in ["normal", "crackles", "wheezes", "both"]:
    count = cycle_dist[label]
    print(f"  {label:10s}: {count:5d} ({100*count/total:.0f}%)")
