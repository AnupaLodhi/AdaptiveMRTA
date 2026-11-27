"""Generate source-grounded baseline and ablation comparison."""
from pathlib import Path
import csv

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "manuscript/tables"
OUT.mkdir(parents=True, exist_ok=True)

rows = [
    (
        "Main", "lock_aware", "Lock-Aware Hungarian",
        "Valid reservation preservation and subset Hungarian repair",
        "Global Hungarian reassignment when all remaining tasks are reserved",
        "A* cost matrix", "No",
        "src/mrta/coordinated_allocators.py"
    ),
    (
        "Main", "adaptive", "Adaptive Reassignment",
        "Progress-dependent PALR saturation delay",
        "Task-completion stagnation shortens delay from 5 to 2 "
        "after a 3-iteration window",
        "A* cost matrix", "Mission-level",
        "src/mrta/adaptive_reassignment_allocator.py"
    ),
    (
        "Main", "palr_5", "PALR-5",
        "Reservation preservation and delayed global replanning",
        "Five consecutive saturation requests are delayed",
        "A* cost matrix", "No",
        "src/mrta/progress_aware_allocator.py"
    ),
    (
        "Main", "casr_0", "CASR-0",
        "Normal selective reassignment, margin 0",
        "After saturation delay; strictly positive path-cost gain",
        "A* path-cost improvement", "No",
        "src/mrta/cost_aware_selective_allocator.py"
    ),
    (
        "Main", "casr_3", "CASR-3",
        "Normal selective reassignment, margin 3",
        "After saturation delay; positive gain of at least 3 steps",
        "A* path-cost improvement", "No",
        "src/mrta/cost_aware_selective_allocator.py"
    ),
    (
        "Main", "casr_v2_s3", "CASR-V2-S3",
        "CASR with stall-conditioned early reassignment",
        "Saturation; owner unchanged-state count >= 3; positive gain",
        "A* path-cost improvement", "Individual robot",
        "src/mrta/stall_aware_allocator.py"
    ),
    (
        "Main", "casr_v2_s2", "CASR-V2-S2 (Proposed)",
        "CASR with stall-conditioned margin relaxation",
        "Saturation; owner unchanged-state count >= 2; positive gain",
        "A* path-cost improvement", "Individual robot",
        "src/mrta/stall_aware_allocator.py"
    ),
    (
        "Ablation", "v2_fixed_margin", "Fixed-Margin Variant",
        "Early stall checks retain conventional margin 3",
        "Saturation; owner stall count >= 2; gain >= 3",
        "A* path-cost improvement", "Individual robot",
        "src/mrta/fixed_margin_allocator.py"
    ),
    (
        "Ablation", "v2_no_recovery", "No-Recovery Variant",
        "Early stall transfers retained; normal rejection recovery disabled",
        "No global fallback after repeated rejected normal checks",
        "A* path-cost improvement", "Individual robot",
        "src/mrta/no_recovery_stall_allocator.py"
    ),
]

for row in rows:
    assert (ROOT / row[-1]).is_file(), row[-1]

path = OUT / "table_4_baseline_comparison.csv"
with path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "group", "method_id", "display_name",
        "allocation_strategy", "adaptation_trigger",
        "cost_basis", "progress_monitoring", "source_file"
    ])
    writer.writerows(rows)

print("===== TABLE IV GENERATED =====")
print("Main comparison methods:", sum(r[0] == "Main" for r in rows))
print("Additional ablation variants:", sum(r[0] == "Ablation" for r in rows))
print("Total rows:", len(rows))
print("Saved:", path)
