"""Protein sequence utility functions.

Consolidates amino-acid detection logic previously duplicated in
mol_instructions.py and scripts/prepare_arrow.py.
"""

# Standard 20 amino acids + IUPAC ambiguous codes (X=unknown, B=D/N, Z=E/Q, U=Sec, O=Pyl)
AA_CHARS = frozenset("ACDEFGHIKLMNPQRSTVWYBXZUO")


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
