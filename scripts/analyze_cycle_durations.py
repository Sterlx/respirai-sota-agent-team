"""Analyze cycle durations to find optimal fixed length for padding."""
from pathlib import Path
import numpy as np

audio_dir = Path("data/audio")

durations = []
for txt_path in sorted(audio_dir.glob("*.txt")):
    for line in txt_path.read_text().strip().splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        start_t = float(parts[0])
        end_t = float(parts[1])
        dur = end_t - start_t
        if dur > 0:
            durations.append(dur)

durations = np.array(durations)
print(f"Total cycles: {len(durations)}")
print(f"Duration stats:")
print(f"  min:    {durations.min():.2f}s")
print(f"  max:    {durations.max():.2f}s")
print(f"  mean:   {durations.mean():.2f}s")
print(f"  median: {np.median(durations):.2f}s")
print(f"  std:    {durations.std():.2f}s")
print()
for p in [50, 75, 80, 85, 90, 95, 99]:
    val = np.percentile(durations, p)
    samples_at_4k = int(val * 4000)
    frames_at_hop128 = samples_at_4k // 128
    print(f"  {p}th percentile: {val:.1f}s → {samples_at_4k} samples → ~{frames_at_hop128} frames")
print()
# Distribution of durations
bins = [0, 1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30]
print("Duration distribution:")
for i in range(len(bins) - 1):
    count = ((durations >= bins[i]) & (durations < bins[i+1])).sum()
    pct = 100 * count / len(durations)
    bar = "█" * int(pct)
    print(f"  {bins[i]:2d}-{bins[i+1]:2d}s: {count:5d} ({pct:4.1f}%) {bar}")
count = (durations >= bins[-1]).sum()
print(f"  {bins[-1]:2d}s+:    {count:5d} ({100*count/len(durations):4.1f}%)")
