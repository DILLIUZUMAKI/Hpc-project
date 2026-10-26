import csv
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import re
import uuid


def rows_for(experiment):
    rows = []
    for candidate in experiment.get("candidates", []):
        for row in candidate.get("measurements", []):
            if candidate.get("status") != "valid":
                continue
            rows.append({"experiment_id": experiment["id"], "benchmark": experiment["config"]["benchmark"],
                         "paradigm": experiment["config"]["paradigm"], "candidate": candidate["id"],
                         "origin": candidate["origin"], "size": experiment["config"]["size"],
                         **{k:v for k,v in row.items() if k != "samples_seconds"},
                         "samples_seconds": json.dumps(row["samples_seconds"])})
    return rows


def to_csv(experiment):
    rows = rows_for(experiment)
    buffer = io.StringIO()
    fields = list(dict.fromkeys(key for row in rows for key in row)) if rows else ["experiment_id", "candidate", "workers", "median_seconds", "speedup", "efficiency"]
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def report(experiment):
    c = experiment["config"]
    lines = ["# Experiment report", f"ID: {experiment['id']}", f"Status: {experiment['status']}",
             f"Benchmark: {c['benchmark']}; paradigm: {c['paradigm']}; size: {c['size']}",
             f"Best valid AI candidate: {experiment.get('best_ai') or 'none'}",
             "", "Timings are external process wall times, including process/MPI startup and output.",
             "This is single-node testing. No multi-node scalability claim is made.", "",
             "| Candidate | Status | Workers | Median (s) | Speedup | Efficiency |",
             "|---|---|---:|---:|---:|---:|"]
    for candidate in experiment.get("candidates", []):
        if not candidate.get("measurements"):
            lines.append(f"| {candidate['id']} | {candidate['status']} | — | — | — | — |")
        for row in candidate.get("measurements", []):
            lines.append(f"| {candidate['id']} | {candidate['status']} | {row['workers']} | {row['median_seconds']:.6g} | {row.get('speedup',1):.3f} | {row.get('efficiency',1):.3f} |")
    lines += ["", "## Interpretation", "Small-case full-vector validation and large-case fingerprints are empirical checks, not a proof of equivalence or race freedom.",
              "Curated reference sources were assistant-authored; obtain an independently human-authored/reviewed baseline before labeling it a human reference in a paper.",
              "Use a held-out timing run to confirm the selected candidate; this search result alone is not statistical significance."]
    return "\n".join(lines)+"\n"


class Store:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def directory(self, ident):
        if not re.fullmatch(r"[0-9]{8}T[0-9]{6}Z-[a-f0-9]{8}", ident):
            raise ValueError("Invalid experiment ID.")
        return self.root/ident

    def create(self, config):
        now = datetime.now(timezone.utc)
        experiment = {"id": now.strftime("%Y%m%dT%H%M%SZ-")+uuid.uuid4().hex[:8],
                      "created_at": now.isoformat(), "config": config.to_dict(), "status": "running",
                      "candidates": [], "events": [], "best_ai": None}
        self.save(experiment)
        return experiment

    def save(self, experiment):
        directory = self.directory(experiment["id"])
        directory.mkdir(parents=True, exist_ok=True)
        # Atomic result replacement means a stopped run retains its previous checkpoint.
        target = directory/"experiment.json"
        temporary = directory/(".checkpoint-"+uuid.uuid4().hex+".tmp")
        temporary.write_text(json.dumps(experiment, indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(target)
        (directory/"results.csv").write_text(to_csv(experiment), encoding="utf-8")
        (directory/"report.md").write_text(report(experiment), encoding="utf-8")
        for candidate in experiment.get("candidates", []):
            ident = candidate["id"]
            if not re.fullmatch(r"[a-z0-9_]+", ident):
                raise ValueError("Invalid candidate ID.")
            if candidate.get("source"):
                (directory/(ident+".c")).write_text(candidate["source"], encoding="utf-8")
            if candidate.get("prompt"):
                (directory/(ident+"_prompt.json")).write_text(json.dumps(candidate["prompt"], indent=2), encoding="utf-8")

    def list(self):
        values = []
        for file in sorted(self.root.glob("*/experiment.json"), reverse=True):
            try:
                experiment = json.loads(file.read_text())
                if self.directory(experiment["id"]) == file.parent:
                    values.append(experiment)
            except (OSError, ValueError, KeyError):
                continue
        return values

    def load(self, ident):
        return json.loads((self.directory(ident)/"experiment.json").read_text())
