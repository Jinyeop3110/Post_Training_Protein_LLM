#!/usr/bin/env python3
"""
Cross-source protein sequence overlap analysis for the combined SFT dataset.

Analyzes how many unique protein sequences appear across multiple data sources,
quantifying redundancy and effective dataset inflation.
"""

import json
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

# Configuration
DATA_DIR = Path("/orcd/pool/006/yeopjin/workspace/Post_Training_Protein_LLM/data/processed/combined_sft_260225")
OUTPUT_DIR = Path("/orcd/pool/006/yeopjin/workspace/Post_Training_Protein_LLM/blog/data/03-02")
OUTPUT_FILE = OUTPUT_DIR / "cross_source_overlap.json"

# Source prefixes
SOURCE_PREFIXES = {
    "mol_": "Mol-Instructions",
    "sp_": "SwissProt",
    "plm_": "ProteinLM",
    "clap_": "SwissProtCLAP",
    "pd_": "ProtDescribe",
    "p2t_": "Protein2Text",
}

# Regex to extract protein sequence from triple-backtick code blocks
SEQ_PATTERN = re.compile(r"```\n?([A-Z]+)\n?```")


def get_source(filename: str) -> str:
    """Map filename to source group."""
    for prefix, name in SOURCE_PREFIXES.items():
        if filename.startswith(prefix):
            return name
    return "Unknown"


def extract_sequence(input_text: str) -> str | None:
    """Extract protein sequence from input field."""
    match = SEQ_PATTERN.search(input_text)
    if match:
        return match.group(1)
    return None


def process_file(filepath: Path, filename: str) -> tuple[str, str, set, list]:
    """
    Process a single JSON file. Returns (source, task, unique_sequences_set, all_sequences_list).

    Uses ijson-style streaming if available, falls back to full load.
    """
    source = get_source(filename)
    task = filename.replace(".json", "")

    # Remove source prefix for cleaner task name
    for prefix in SOURCE_PREFIXES:
        if task.startswith(prefix):
            task = task[len(prefix):]
            break

    real_path = os.path.realpath(filepath)

    # Load and process
    t0 = time.time()
    print(f"  Loading {filename}...", end=" ", flush=True)

    with open(real_path, "r") as f:
        data = json.load(f)

    unique_seqs = set()
    all_seqs = []  # (sequence, task) for redundancy analysis
    no_match = 0

    for rec in data:
        inp = rec.get("input", "")
        seq = extract_sequence(inp)
        if seq:
            unique_seqs.add(seq)
            all_seqs.append(seq)
        else:
            no_match += 1

    elapsed = time.time() - t0
    print(f"{len(data):>8d} records, {len(unique_seqs):>7d} unique seqs, "
          f"{no_match} no-match, {elapsed:.1f}s")

    del data  # Free memory

    return source, task, unique_seqs, all_seqs


def main():
    print("=" * 80)
    print("Cross-Source Protein Sequence Overlap Analysis")
    print("=" * 80)
    print(f"Data directory: {DATA_DIR}")
    print()

    # Collect all JSON files (excluding manifest)
    json_files = sorted([
        f for f in os.listdir(DATA_DIR)
        if f.endswith(".json") and f != "assembly_manifest.json"
    ])

    print(f"Found {len(json_files)} dataset files\n")

    # Per-source data
    source_sequences: dict[str, set] = defaultdict(set)  # source -> set of sequences
    file_sequences: dict[str, set] = {}  # filename -> set of sequences
    file_to_source: dict[str, str] = {}
    file_to_task: dict[str, str] = {}

    # For redundancy: (sequence, task) pairs
    seq_task_pairs: set[tuple[str, str]] = set()
    total_records = 0
    per_file_records: dict[str, int] = {}

    # For top-protein analysis (accumulated during first pass)
    seq_record_count: Counter = Counter()
    seq_source_record_count: dict[str, dict] = defaultdict(lambda: defaultdict(int))

    # Process each file
    print("Processing files:")
    t_start = time.time()

    for filename in json_files:
        filepath = DATA_DIR / filename
        source, task, unique_seqs, all_seqs = process_file(filepath, filename)

        source_sequences[source].update(unique_seqs)
        file_sequences[filename] = unique_seqs
        file_to_source[filename] = source
        file_to_task[filename] = task
        per_file_records[filename] = len(all_seqs)
        total_records += len(all_seqs)

        # Track (seq, task) pairs and per-sequence record counts
        for seq in all_seqs:
            seq_task_pairs.add((seq, task))
            seq_record_count[seq] += 1
            seq_source_record_count[seq][source] += 1

    t_total = time.time() - t_start
    print(f"\nTotal processing time: {t_total:.1f}s")

    # =========================================================================
    # Analysis
    # =========================================================================

    sources = sorted(source_sequences.keys())

    # 1. Global unique sequences
    all_unique = set()
    for seqs in source_sequences.values():
        all_unique.update(seqs)

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    print(f"\nTotal records:           {total_records:>10,d}")
    print(f"Total unique sequences:  {len(all_unique):>10,d}")
    print(f"Unique (seq, task) pairs:{len(seq_task_pairs):>10,d}")
    print(f"Record-to-unique ratio:  {total_records / len(all_unique):>10.2f}x")
    print(f"Record-to-(seq,task):    {total_records / len(seq_task_pairs):>10.2f}x")

    # 2. Per-source stats
    print("\n--- Per-Source Unique Protein Counts ---")
    per_source_stats = {}
    for source in sources:
        n_unique = len(source_sequences[source])
        # Count records for this source
        source_records = sum(
            per_file_records[f] for f in json_files if file_to_source[f] == source
        )
        per_source_stats[source] = {
            "unique_sequences": n_unique,
            "total_records": source_records,
            "files": [f for f in json_files if file_to_source[f] == source],
        }
        print(f"  {source:20s}: {n_unique:>8,d} unique seqs / {source_records:>8,d} records "
              f"({source_records / n_unique:.1f}x inflation)")

    # 3. Per-file stats
    print("\n--- Per-File Unique Protein Counts ---")
    per_file_stats = {}
    for filename in json_files:
        n_unique = len(file_sequences[filename])
        n_records = per_file_records[filename]
        source = file_to_source[filename]
        task = file_to_task[filename]
        per_file_stats[filename] = {
            "source": source,
            "task": task,
            "unique_sequences": n_unique,
            "total_records": n_records,
        }
        print(f"  {filename:45s}: {n_unique:>7,d} unique / {n_records:>7,d} records")

    # 4. Cross-source overlap matrix
    print("\n--- Cross-Source Overlap Matrix ---")
    print("(Number of proteins appearing in BOTH sources)")
    overlap_matrix = {}

    # Header
    short_names = {s: s[:8] for s in sources}
    header = f"{'':20s}" + "".join(f"{short_names[s]:>10s}" for s in sources)
    print(header)

    for s1 in sources:
        overlap_matrix[s1] = {}
        row = f"{s1:20s}"
        for s2 in sources:
            if s1 == s2:
                overlap = len(source_sequences[s1])
            else:
                overlap = len(source_sequences[s1] & source_sequences[s2])
            overlap_matrix[s1][s2] = overlap
            row += f"{overlap:>10,d}"
        print(row)

    # 5. Distribution: proteins by number of sources
    print("\n--- Proteins by Number of Sources ---")
    seq_source_count = Counter()

    # Build: sequence -> set of sources
    seq_to_sources: dict[str, set] = defaultdict(set)
    for source, seqs in source_sequences.items():
        for seq in seqs:
            seq_to_sources[seq].add(source)

    for seq, src_set in seq_to_sources.items():
        seq_source_count[len(src_set)] += 1

    source_dist = {}
    for n_sources in sorted(seq_source_count.keys()):
        count = seq_source_count[n_sources]
        pct = 100.0 * count / len(all_unique)
        source_dist[n_sources] = count
        print(f"  In {n_sources} source(s): {count:>8,d} proteins ({pct:5.1f}%)")

    # 6. Top overrepresented proteins (uses data accumulated in first pass)
    print("\n--- Top 20 Most Cross-Referenced Proteins ---")
    print("(Appear in most sources, with most total records)")

    # Sort by (num_sources desc, num_records desc)
    top_seqs = sorted(
        seq_to_sources.keys(),
        key=lambda s: (len(seq_to_sources[s]), seq_record_count[s]),
        reverse=True,
    )[:20]

    top_proteins = []
    print(f"{'Seq (first 30)':32s} {'Len':>5s} {'Sources':>7s} {'Records':>8s}  Source breakdown")
    for seq in top_seqs:
        n_src = len(seq_to_sources[seq])
        n_rec = seq_record_count[seq]
        breakdown = ", ".join(
            f"{src}:{cnt}" for src, cnt in sorted(seq_source_record_count[seq].items(), key=lambda x: -x[1])
        )
        print(f"  {seq[:30]:30s} {len(seq):>5d} {n_src:>7d} {n_rec:>8,d}  {breakdown}")
        top_proteins.append({
            "sequence_prefix": seq[:50],
            "length": len(seq),
            "num_sources": n_src,
            "total_records": n_rec,
            "source_breakdown": dict(seq_source_record_count[seq]),
        })

    # 7. Cross-source overlap as percentage matrix
    print("\n--- Cross-Source Overlap (% of row source) ---")
    header = f"{'':20s}" + "".join(f"{short_names[s]:>10s}" for s in sources)
    print(header)

    overlap_pct_matrix = {}
    for s1 in sources:
        overlap_pct_matrix[s1] = {}
        row = f"{s1:20s}"
        for s2 in sources:
            if len(source_sequences[s1]) > 0:
                pct = 100.0 * overlap_matrix[s1][s2] / len(source_sequences[s1])
            else:
                pct = 0
            overlap_pct_matrix[s1][s2] = round(pct, 1)
            row += f"{pct:>9.1f}%"
        print(row)

    # 8. Task-level redundancy
    print("\n--- Task-Level Redundancy ---")
    print(f"Total records:                    {total_records:>10,d}")
    print(f"Unique (protein, task) pairs:     {len(seq_task_pairs):>10,d}")
    print(f"Duplicate records (same seq+task):{total_records - len(seq_task_pairs):>10,d}")
    print(f"Effective redundancy ratio:       {total_records / len(seq_task_pairs):>10.2f}x")

    # =========================================================================
    # Save results
    # =========================================================================

    results = {
        "summary": {
            "total_records": total_records,
            "total_unique_sequences": len(all_unique),
            "unique_seq_task_pairs": len(seq_task_pairs),
            "record_to_unique_ratio": round(total_records / len(all_unique), 2),
            "record_to_seq_task_ratio": round(total_records / len(seq_task_pairs), 2),
            "num_sources": len(sources),
            "num_files": len(json_files),
        },
        "per_source": {
            source: {
                "unique_sequences": stats["unique_sequences"],
                "total_records": stats["total_records"],
                "inflation_ratio": round(stats["total_records"] / stats["unique_sequences"], 2),
                "files": stats["files"],
            }
            for source, stats in per_source_stats.items()
        },
        "per_file": per_file_stats,
        "overlap_matrix": {
            s1: {s2: overlap_matrix[s1][s2] for s2 in sources}
            for s1 in sources
        },
        "overlap_pct_matrix": overlap_pct_matrix,
        "proteins_by_num_sources": {str(k): v for k, v in sorted(source_dist.items())},
        "top_overrepresented_proteins": top_proteins,
        "source_names": sources,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {OUTPUT_FILE}")
    print("Done.")


if __name__ == "__main__":
    main()
