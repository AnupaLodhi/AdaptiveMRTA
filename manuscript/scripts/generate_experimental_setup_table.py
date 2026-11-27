"""Generate Table III from verified experiment configuration."""
from pathlib import Path
import csv

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments"
OUT = ROOT / "manuscript/tables"
OUT.mkdir(parents=True, exist_ok=True)

phases = [
    ("Development", "run_casr_v2_development.py", 2000, 2019, 7, 1, 280),
    ("Independent holdout", "run_casr_v2_holdout.py", 3000, 3019, 7, 1, 280),
    ("Ablation", "run_casr_v2_ablation.py", 4000, 4029, 5, 1, 300),
    ("Scalability", "run_casr_v2_scalability.py", 5000, 5009, 5, 1, 600),
    ("Controlled runtime", "run_casr_runtime.py", 6000, 6004, 5, 3, 300),
]

scalability = [
    ("small", 5, 10, 52, 30, 0.20),
    ("medium", 15, 20, 52, 30, 0.20),
    ("large", 30, 50, 52, 30, 0.20),
    ("extra_large", 50, 100, 52, 30, 0.20),
    ("expanded", 50, 100, 100, 60, 0.20),
    ("dense_obstacles", 15, 20, 52, 30, 0.30),
]

rows = []

def add(section, item, value, detail, source):
    rows.append([section, item, str(value), detail, source])

for name, script, first, last, methods, repetitions, expected in phases:
    source = (EXP / script).read_text()

    assert f"range({first}, {last + 1})" in source, script
    assert "max_iterations=500" in source, script
    assert len(source) > 100

    configs = 6 if name == "Scalability" else (
        2 if name == "Controlled runtime" else 1
    )
    actual = (last - first + 1) * 2 * methods * repetitions * configs
    assert actual == expected, (name, actual, expected)

    add("Phase", name, expected,
        f"Seeds {first}-{last}; {methods} methods; "
        f"{repetitions} repetition(s); 2 failure conditions",
        script)

scalability_source = (
    EXP / "run_casr_v2_scalability.py"
).read_text()

for name, robots, tasks, width, height, density in scalability:
    expected_tuple = (
        f'("{name}", {robots}, {tasks}, '
        f'{width}, {height}, {density:.2f})'
    )
    assert expected_tuple in scalability_source, name

    add(
        "Scalability", name, f"{robots} robots / {tasks} tasks",
        f"{width}x{height}; obstacle density={density:.0%}",
        "run_casr_v2_scalability.py"
    )

for item, value, detail, source in [
    ("Maximum iterations", 500, "Per simulation", "Experiment scripts"),
    ("Failure rates", "0%, 20%", "Same rates across all phases", "Experiment scripts"),
    ("Failure iteration", 5, "Applied to selected robots", "Experiment scripts"),
    ("Failure sampling seed offset", 50000, "seed + 50000", "Experiment scripts"),
    ("Default medium grid", "52x30", "15 robots, 20 tasks", "src/mrta/config.py"),
    ("Normal cost margin", 3, "Grid steps", "src/mrta/cost_aware_selective_allocator.py"),
    ("Saturation delay", 5, "Normal selective check after five delayed requests", "src/mrta/cost_aware_selective_allocator.py"),
    ("Stall threshold", 2, "Unchanged-state checks", "src/mrta/stall_aware_allocator.py"),
    ("Global recovery threshold", 2, "Rejected normal selective checks", "src/mrta/cost_aware_selective_allocator.py"),
    ("Runtime repetitions", 3, "Shuffled method order per repetition", "run_casr_runtime.py"),
]:
    add("Shared setting", item, value, detail, source)

assert sum(p[6] for p in phases) == 1760

path = OUT / "table_3_experimental_setup.csv"
with path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["section", "item", "value", "detail", "source"])
    writer.writerows(rows)

print("===== TABLE III GENERATED =====")
print("Rows:", len(rows))
print("Recorded executions:", sum(p[6] for p in phases))
print("Saved:", path)
