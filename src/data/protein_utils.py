"""Protein sequence utility functions.

Consolidates amino-acid detection logic previously duplicated in
mol_instructions.py and scripts/prepare_arrow.py.
"""

import re

# Standard 20 amino acids + IUPAC ambiguous codes (X=unknown, B=D/N, Z=E/Q, U=Sec, O=Pyl)
AA_CHARS = frozenset("ACDEFGHIKLMNPQRSTVWYBXZUO")

# Pattern for <seq>M K M R F...</seq> tags (space-separated AAs)
_SEQ_TAG_PATTERN = re.compile(r"<seq>\s*(.*?)\s*</seq>", re.DOTALL)


def extract_seq_tag_proteins(text: str) -> list[str]:
    """Extract protein sequences from <seq>...</seq> tags.

    SSL data uses ``<seq>M K M R F L G...</seq>`` format with space-separated
    single-letter amino acid codes. This function extracts and joins them into
    standard contiguous sequences.

    Args:
        text: Text possibly containing one or more <seq>...</seq> tags.

    Returns:
        List of protein sequences (uppercase, no spaces). Empty if no tags found.
    """
    matches = _SEQ_TAG_PATTERN.findall(text)
    sequences = []
    for match in matches:
        # Join space-separated AA letters: "M K M R F" -> "MKMRF"
        aa = match.replace(" ", "").replace("\n", "").replace("\t", "").upper()
        # Validate that it's actually amino acids
        if aa and all(c in AA_CHARS for c in aa):
            sequences.append(aa)
    return sequences


def extract_protein_sequence(text: str) -> str:
    """Extract protein sequence from text.

    Handles:
    - Pure AA sequences (standard 20 + ambiguous XBZUO codes)
    - Sequences wrapped in triple backticks (```...```)
    - Short sequences (< 10 AA)
    - Multi-line text where one line is a protein sequence

    Args:
        text: Input text that may contain a protein sequence.

    Returns:
        Extracted protein sequence (uppercase), or original text if no
        clear sequence found.
    """
    if not text:
        return ""

    # Strip markdown code fences that wrap sequences in many datasets
    cleaned = text.strip()
    if cleaned.startswith("```") and cleaned.endswith("```"):
        cleaned = cleaned[3:-3].strip()

    # Check if the entire input is a protein sequence
    upper = cleaned.upper()
    if upper and all(c in AA_CHARS for c in upper):
        return upper

    # Try to extract sequence from structured input
    # Sometimes sequences are on their own line or after a colon
    for line in text.split("\n"):
        line = line.strip().upper()
        if not line or line.startswith("```"):
            continue
        # Accept any line that's mostly AA characters (>= 4 chars to catch short seqs)
        if len(line) >= 4:
            aa_count = sum(1 for c in line if c in AA_CHARS)
            if aa_count / len(line) > 0.9:
                return line

    # Return the original input if no clear sequence is found
    return text.strip()


def protein_sequence_length(text: str) -> int:
    """Return length of protein sequence in text, or len(text) if none found.

    Uses the same heuristics as extract_protein_sequence but returns length
    directly (avoids allocating the full sequence string when only the
    length is needed).

    Args:
        text: Input text that may contain a protein sequence.

    Returns:
        Length of the detected protein sequence, or len(text) as fallback.
    """
    if not text:
        return 0

    cleaned = text.strip().upper()
    if cleaned.startswith("```") and cleaned.endswith("```"):
        cleaned = cleaned[3:-3].strip()

    if cleaned and all(c in AA_CHARS for c in cleaned):
        return len(cleaned)

    # Fall back to longest AA-like line
    for line in text.split("\n"):
        line = line.strip().upper()
        if len(line) >= 4:
            aa_count = sum(1 for c in line if c in AA_CHARS)
            if aa_count / len(line) > 0.9:
                return len(line)

    return len(text)


def protein_level_split(
    inputs: list[str],
    train_ratio: float = 0.9,
    val_ratio: float = 0.05,
    seed: int = 42,
    extract_fn=None,
) -> dict[str, list[int]]:
    """Split row indices by unique protein so no protein spans multiple splits.

    Groups rows by their extracted protein sequence, shuffles the unique
    proteins, and assigns each protein (with all its rows) to train, val, or
    test.  The split ratios target *row* counts, not protein counts.

    Args:
        inputs: List of raw ``input`` field strings (one per row).
        train_ratio: Fraction of rows for training.
        val_ratio: Fraction of rows for validation (remainder → test).
        seed: Random seed for reproducibility.
        extract_fn: Callable to extract protein sequence from input text.
            Defaults to :func:`extract_protein_sequence`.

    Returns:
        Dict with keys ``"train"``, ``"validation"``, ``"test"``, each
        mapping to a sorted list of row indices.
    """
    import random

    if extract_fn is None:
        extract_fn = extract_protein_sequence

    # Group row indices by canonical protein sequence
    protein_to_rows: dict[str, list[int]] = {}
    for i, text in enumerate(inputs):
        seq = extract_fn(text)
        protein_to_rows.setdefault(seq, []).append(i)

    # Deterministic shuffle of unique proteins
    proteins = list(protein_to_rows.keys())
    rng = random.Random(seed)
    rng.shuffle(proteins)

    total = len(inputs)
    train_target = int(total * train_ratio)
    val_target = int(total * val_ratio)

    split_indices: dict[str, list[int]] = {"train": [], "validation": [], "test": []}
    train_count = 0
    val_count = 0

    for prot in proteins:
        rows = protein_to_rows[prot]
        if train_count < train_target:
            split_indices["train"].extend(rows)
            train_count += len(rows)
        elif val_count < val_target:
            split_indices["validation"].extend(rows)
            val_count += len(rows)
        else:
            split_indices["test"].extend(rows)

    # Sort within each split for deterministic ordering
    for key in split_indices:
        split_indices[key].sort()

    return split_indices
