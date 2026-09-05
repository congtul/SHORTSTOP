"""Aggregates every CALVIN baseline's own EVAL-cohort `results.json` (each
produced by its own dedicated script, run with NO `--tuning` flag) into
one combined table matching the paper's own Table II shape -- the single
place to pull final numbers from for the paper, instead of hand-copying
from each script's separate log.

Purely a JSON aggregator -- no CALVIN env/policy/checkpoint needed, no
WSL2 requirement of its own (though it obviously can't produce anything
meaningful until the baseline scripts it reads from have actually been
run there). Computes `recovery_rate`/`conservatism_cost` uniformly from
each baseline's own serialized `sequence_results`, INSTEAD OF trusting
whatever a baseline script may or may not have already computed inline --
one consistent source of truth, and the exact reason every run_calvin_*.py
script serializes full `sequence_results` in the first place (see
docs/PARAMETERS_REFERENCE.md's metrics-gap note).

Run each baseline's own eval command first (see each script's own
docstring), THEN this:

    cd SHORTSTOP
    python scripts/run_calvin_unshielded.py
    python scripts/run_calvin_shielded.py
    python scripts/run_calvin_stl_monitor.py
    python scripts/run_calvin_mpc_filter.py
    python scripts/run_calvin_shortstop.py
    python scripts/run_calvin_full_eval.py

Picks the MOST RECENT `outputs/<baseline>_runs/run_*_eval/results.json`
for each baseline automatically (sorted by the run-folder's own timestamp
in its name) -- rerun a baseline and this script picks up the new result
without any argument needed, but ALSO logs which exact run-folder it read
for each row, so a stale/wrong pick is always visible, not silent.

NOT the ablation table (Stage 1/2/4) -- that's scripts/run_calvin_
shortstop_ablation.py's own `results.json`, already self-contained (every
stage's own violation/success/etc side by side), no aggregation needed.
This script is specifically Table II's baseline-comparison row set.
"""
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402
sys.path.insert(0, str(REPO_ROOT))

from shortstop.calvin_metrics import conservatism_cost, recovery_rate  # noqa: E402

# Must match every baseline script's own OBSTACLE_RADIUS constant.
OBSTACLE_RADIUS = 0.06

BASELINES = {
    "unshielded": {
        "run_dir": "calvin_unshielded_runs",
        "select_entry": lambda results: next(e for e in results["results"] if e.get("radius") == OBSTACLE_RADIUS),
    },
    "conf_thresh": {
        "run_dir": "calvin_shielded_runs",
        "select_entry": lambda results: results["results"][0],
    },
    "stl_monitor": {
        "run_dir": "calvin_stl_monitor_runs",
        "select_entry": lambda results: results["results"][0],
    },
    "mpc_filter": {
        "run_dir": "calvin_mpc_filter_runs",
        "select_entry": lambda results: results["results"][0],
    },
    "shortstop": {
        "run_dir": "calvin_shortstop_runs",
        "select_entry": lambda results: results["results"][0],
    },
}


def _find_latest_eval_run(run_dir_name):
    base = REPO_ROOT / "outputs" / run_dir_name
    if not base.exists():
        return None
    candidates = sorted(p for p in base.glob("run_*_eval") if (p / "results.json").exists())
    return candidates[-1] if candidates else None


def _load_entry(baseline_name, spec):
    run_path = _find_latest_eval_run(spec["run_dir"])
    if run_path is None:
        raise FileNotFoundError(
            f"no eval run found for '{baseline_name}' under outputs/{spec['run_dir']}/run_*_eval/ -- "
            f"run its own script first (no --tuning flag)."
        )
    with open(run_path / "results.json", encoding="utf-8") as f:
        results = json.load(f)
    if results.get("tuning_mode"):
        raise ValueError(f"{run_path} was a --tuning run, not eval -- re-run without --tuning")
    entry = spec["select_entry"](results)
    return entry, run_path


def main():
    entries = {}
    run_paths = {}
    for name, spec in BASELINES.items():
        entry, run_path = _load_entry(name, spec)
        entries[name] = entry
        run_paths[name] = run_path
        print(f"[{name}] read from {run_path.relative_to(REPO_ROOT)}")

    unshielded_sequence_results = entries["unshielded"]["sequence_results"]

    rows = []
    for name in BASELINES:
        entry = entries[name]
        row = {
            "baseline": name,
            "violation_rate": entry.get("violation_rate"),
            "success_rate": entry.get("success_rate"),
            "avg_seq_len": entry.get("avg_seq_len"),
            "n_sequences": entry.get("n_sequences"),
            "shield_activation_rate": entry.get("shield_activation_rate"),
            "fallback_rate": entry.get("fallback_rate"),
            "intervention_precision": entry.get("intervention_precision"),
            "latency_ms_mean": (entry.get("latency_ms") or {}).get("mean"),
            "latency_ms_p95": (entry.get("latency_ms") or {}).get("p95"),
        }
        if name == "unshielded":
            row["recovery_rate"] = None
            row["conservatism_cost"] = None
        else:
            row["recovery_rate"] = recovery_rate(entry["sequence_results"])
            row["conservatism_cost"] = conservatism_cost(unshielded_sequence_results, entry["sequence_results"])
        rows.append(row)

    header = (
        f"{'baseline':<14}{'violation':>10}{'success':>10}{'avg_len':>9}{'activation':>11}"
        f"{'fallback':>10}{'precision':>10}{'lat_mean':>9}{'recovery':>10}{'consv_cost':>11}"
    )
    print()
    print(header)
    print("-" * len(header))
    for row in rows:
        def _fmt(x, spec="{:.3f}"):
            return spec.format(x) if x is not None else "n/a"

        print(
            f"{row['baseline']:<14}{_fmt(row['violation_rate']):>10}{_fmt(row['success_rate']):>10}"
            f"{_fmt(row['avg_seq_len'], '{:.2f}'):>9}{_fmt(row['shield_activation_rate']):>11}"
            f"{_fmt(row['fallback_rate']):>10}{_fmt(row['intervention_precision']):>10}"
            f"{_fmt(row['latency_ms_mean'], '{:.2f}'):>9}{_fmt(row['recovery_rate']):>10}"
            f"{_fmt(row['conservatism_cost']):>11}"
        )

    out_path = REPO_ROOT / "outputs" / "calvin_full_eval_table.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "obstacle_radius": OBSTACLE_RADIUS,
            "source_runs": {name: str(p.relative_to(REPO_ROOT)) for name, p in run_paths.items()},
            "rows": rows,
        }, f, indent=2)
    print(f"\n[run] wrote combined table to: {out_path}")


if __name__ == "__main__":
    main()
