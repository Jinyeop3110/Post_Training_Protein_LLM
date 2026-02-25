#!/usr/bin/env python3
"""
Data Audit: Combined Dataset Protein Extraction Analysis

Tests how _extract_protein_sequence() from mol_instructions.py handles
each of the 12 JSON files in data/processed/combined/.

Generates a Markdown report at blog/02-21/data_conversion_report.md
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.mol_instructions import MolInstructionsDataset

DATA_DIR = Path(
    "/home/yeopjin/orcd/pool/workspace/Post_Training_Protein_LLM"
    "/data/processed/combined"
)
REPORT_PATH = Path(__file__).parent / "data_conversion_report.md"

AA_CHARS = set("ACDEFGHIKLMNPQRSTVWY")
NUM_SAMPLES = 5  # Per-file detailed samples


def classify_input(input_text: str) -> str:
    """Classify what the input field contains."""
    cleaned = input_text.strip()
    if not cleaned:
        return "empty"

    # Check for backtick-wrapped sequence
    if cleaned.startswith("```"):
        inner = cleaned.strip("`").strip()
        if inner and all(c in AA_CHARS for c in inner.upper()):
            return "backtick_protein"
        return "backtick_mixed"

    # Check if raw protein sequence
    upper = cleaned.upper()
    if len(upper) >= 10:
        aa_frac = sum(1 for c in upper if c in AA_CHARS) / len(upper)
        if aa_frac > 0.9:
            return "raw_protein"

    # Text description (e.g., protein design requirements)
    return "text_description"


def extract_protein_for_audit(input_text: str) -> dict:
    """Run the actual extractor and analyze the result."""
    # Instantiate a temporary dataset object to access _extract_protein_sequence
    # We'll just call the static-like method directly
    ds = MolInstructionsDataset.__new__(MolInstructionsDataset)
    extracted = ds._extract_protein_sequence(input_text)

    # Analyze quality
    if not extracted:
        quality = "empty"
    else:
        upper = extracted.upper()
        if len(upper) >= 10:
            aa_frac = sum(1 for c in upper if c in AA_CHARS) / len(upper)
            if aa_frac > 0.95:
                quality = "valid_protein"
            elif aa_frac > 0.5:
                quality = "partial_protein"
            else:
                quality = "not_protein"
        else:
            quality = "too_short"

    return {
        "extracted": extracted[:100] + ("..." if len(extracted) > 100 else ""),
        "length": len(extracted),
        "quality": quality,
        "is_valid_protein": quality == "valid_protein",
    }


def audit_file(filepath: Path) -> dict:
    """Audit a single JSON data file."""
    with open(filepath) as f:
        records = json.load(f)

    filename = filepath.name
    prefix = filename.split("_")[0]  # mol, sp, wp

    results = {
        "filename": filename,
        "prefix": prefix,
        "total_records": len(records),
        "input_types": Counter(),
        "extraction_quality": Counter(),
        "protein_lengths": [],
        "samples": [],
        "issues": [],
    }

    for i, record in enumerate(records):
        input_text = record.get("input", "")
        output_text = record.get("output", "")

        # Classify input type
        input_type = classify_input(input_text)
        results["input_types"][input_type] += 1

        # Run extraction
        extraction = extract_protein_for_audit(input_text)
        results["extraction_quality"][extraction["quality"]] += 1

        if extraction["is_valid_protein"]:
            results["protein_lengths"].append(extraction["length"])

        # Collect detailed samples
        if i < NUM_SAMPLES:
            results["samples"].append({
                "index": i,
                "instruction": record.get("instruction", "")[:120],
                "input_type": input_type,
                "input_preview": input_text[:200],
                "output_preview": output_text[:150],
                "extraction": extraction,
            })

    # Check for issues
    not_protein = results["extraction_quality"].get("not_protein", 0)
    too_short = results["extraction_quality"].get("too_short", 0)
    empty = results["extraction_quality"].get("empty", 0)
    text_desc = results["input_types"].get("text_description", 0)

    if not_protein > 0:
        results["issues"].append(
            f"{not_protein:,} records extracted non-protein text"
        )
    if text_desc > 0:
        results["issues"].append(
            f"{text_desc:,} records have text descriptions as input "
            f"(not protein sequences)"
        )
    if too_short > 0:
        results["issues"].append(f"{too_short:,} records have too-short extractions")

    # Check if protein is in output instead (design tasks)
    if text_desc > 0:
        output_has_protein = 0
        for record in records[:1000]:  # Sample first 1000
            out = record.get("output", "")
            if "```" in out:
                inner = out.split("```")[1].strip() if "```" in out else ""
                if inner and len(inner) > 20:
                    aa_frac = sum(1 for c in inner.upper() if c in AA_CHARS) / len(inner)
                    if aa_frac > 0.9:
                        output_has_protein += 1
        if output_has_protein > 0:
            ratio = output_has_protein / min(1000, len(records))
            results["issues"].append(
                f"~{ratio:.0%} of records have protein in OUTPUT "
                f"(design task: input=requirements, output=sequence)"
            )

    return results


def generate_report(all_results: list) -> str:
    """Generate the Markdown report."""
    lines = []
    lines.append("# Combined Dataset: Protein Extraction Audit")
    lines.append("")
    lines.append("**Date**: 2026-02-21")
    lines.append("**Data dir**: `data/processed/combined/`")
    lines.append("**Config**: `data=combined` (sampling_temperature=0.5)")
    lines.append("**Extractor**: `MolInstructionsDataset._extract_protein_sequence()`")
    lines.append("")

    # Summary table
    lines.append("## Summary")
    lines.append("")
    lines.append(
        "| File | Records | Input Type | Extraction Quality "
        "| Avg Protein Len | Issues |"
    )
    lines.append(
        "|------|---------|------------|-------------------|"
        "----------------|--------|"
    )

    total_records = 0
    total_valid = 0
    total_issues = 0

    for r in all_results:
        total_records += r["total_records"]
        valid = r["extraction_quality"].get("valid_protein", 0)
        total_valid += valid

        # Dominant input type
        dominant_input = r["input_types"].most_common(1)[0][0] if r["input_types"] else "?"
        # Dominant quality
        dominant_quality = r["extraction_quality"].most_common(1)[0][0] if r["extraction_quality"] else "?"

        avg_len = (
            f"{sum(r['protein_lengths'])/len(r['protein_lengths']):.0f}"
            if r["protein_lengths"]
            else "N/A"
        )

        issue_count = len(r["issues"])
        total_issues += issue_count
        issue_str = f"{issue_count} issue(s)" if issue_count > 0 else "OK"

        lines.append(
            f"| `{r['filename']}` | {r['total_records']:,} "
            f"| {dominant_input} | {dominant_quality} ({valid:,}/{r['total_records']:,}) "
            f"| {avg_len} | {issue_str} |"
        )

    lines.append("")
    lines.append(
        f"**Total**: {total_records:,} records, "
        f"{total_valid:,} valid protein extractions "
        f"({total_valid/total_records:.1%}), "
        f"{total_issues} issues"
    )
    lines.append("")

    # Data source breakdown
    lines.append("## Data Sources")
    lines.append("")
    source_counts = defaultdict(int)
    for r in all_results:
        source_counts[r["prefix"]] += r["total_records"]
    for prefix, count in sorted(source_counts.items()):
        label = {"mol": "Mol-Instructions", "sp": "Swiss-Prot", "wp": "Wikipedia Protein"}
        lines.append(f"- **{label.get(prefix, prefix)}** (`{prefix}_*`): {count:,} records")
    lines.append("")

    # Issues detail
    lines.append("## Issues Found")
    lines.append("")
    any_issues = False
    for r in all_results:
        if r["issues"]:
            any_issues = True
            lines.append(f"### `{r['filename']}`")
            for issue in r["issues"]:
                lines.append(f"- {issue}")
            lines.append("")

    if not any_issues:
        lines.append("No issues found.")
        lines.append("")

    # Per-file detailed analysis
    lines.append("## Per-File Detailed Analysis")
    lines.append("")

    for r in all_results:
        lines.append(f"### `{r['filename']}`")
        lines.append("")
        lines.append(f"- **Records**: {r['total_records']:,}")
        lines.append(f"- **Source**: `{r['prefix']}_`")
        lines.append(f"- **Input type distribution**: {dict(r['input_types'])}")
        lines.append(f"- **Extraction quality**: {dict(r['extraction_quality'])}")
        if r["protein_lengths"]:
            lengths = r["protein_lengths"]
            lines.append(
                f"- **Protein lengths**: min={min(lengths)}, "
                f"max={max(lengths)}, mean={sum(lengths)/len(lengths):.0f}, "
                f"median={sorted(lengths)[len(lengths)//2]}"
            )
        lines.append("")

        # Show samples
        lines.append("**Samples:**")
        lines.append("")
        for s in r["samples"]:
            lines.append(f"<details><summary>Record {s['index']}: {s['input_type']} → {s['extraction']['quality']}</summary>")
            lines.append("")
            lines.append(f"- **Instruction**: `{s['instruction']}`")
            lines.append(f"- **Input** ({s['input_type']}): `{s['input_preview'][:150]}...`")
            lines.append(f"- **Output preview**: `{s['output_preview'][:100]}...`")
            lines.append(f"- **Extracted protein**: `{s['extraction']['extracted']}` (len={s['extraction']['length']})")
            lines.append(f"- **Quality**: **{s['extraction']['quality']}**")
            lines.append("")
            lines.append("</details>")
            lines.append("")

    # Recommendations
    lines.append("## Recommendations")
    lines.append("")

    return "\n".join(lines)


def main():
    print("=" * 60)
    print("Combined Dataset Protein Extraction Audit")
    print("=" * 60)

    json_files = sorted(DATA_DIR.glob("*.json"))
    print(f"Found {len(json_files)} JSON files in {DATA_DIR}")

    all_results = []
    for filepath in json_files:
        print(f"\nAuditing: {filepath.name}...")
        result = audit_file(filepath)
        all_results.append(result)

        valid = result["extraction_quality"].get("valid_protein", 0)
        print(
            f"  {result['total_records']:,} records, "
            f"{valid:,} valid proteins, "
            f"{len(result['issues'])} issues"
        )
        for issue in result["issues"]:
            print(f"  ⚠ {issue}")

    # Generate report
    report = generate_report(all_results)

    with open(REPORT_PATH, "w") as f:
        f.write(report)

    print(f"\n{'=' * 60}")
    print(f"Report written to: {REPORT_PATH}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
