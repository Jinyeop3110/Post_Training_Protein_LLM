"""
Wikipedia-Enriched Protein SFT Data Pipeline
=============================================

Fetches rich biological descriptions from Wikipedia for proteins in the
IPD-PDB dataset, producing high-quality instruction-following pairs with
real biological knowledge (function, structure, disease associations).

Pipeline stages:
    1. Extract unique PDB IDs from IPD-PDB list.csv (~60K unique IDs)
    2. Fetch RCSB PDB metadata (protein name, UniProt ID, organism, method)
    3. Deduplicate by UniProt ID (~30K unique proteins)
    4. Search Wikipedia for each protein, fetch article wikitext
    5. Parse sections (intro, function, structure, disease) into SFT records
    6. Write/append output JSON files with deduplication

Output files (one per task type):
    - protein_overview.json    — General protein descriptions from Wikipedia intro
    - protein_function.json    — Molecular/biological function descriptions
    - protein_structure.json   — 3D structure and domain information
    - disease_association.json — Clinical significance and disease links
    - conversion_stats.json    — Cumulative processing statistics

Each record follows the Mol-Instructions schema:
    {"instruction": "...", "input": "```\\nSEQUENCE\\n```", "output": "...",
     "metadata": {"pdb_id": "...", "uniprot_id": "...", "task": "...", ...}}

Incremental processing:
    The script supports incremental runs via --offset and --limit. Stage 6
    appends new records to existing output files and deduplicates by pdb_id,
    so you can safely expand the dataset without re-processing old entries.

Parallel processing:
    Use --parallel N to split the PDB ID range into N non-overlapping chunks,
    each running as a separate subprocess with its own cache directory.
    After all workers finish, results are merged into the main output dir.

Usage examples:
    # First run: process PDB IDs 0-20000
    python src/data/wikipedia_protein_converter.py \\
        --data-dir data/raw/pdb_2021aug02_sample \\
        --output data/processed/wikipedia_protein \\
        --limit 20000

    # Expand: append PDB IDs 20000-60000 (existing records preserved)
    python src/data/wikipedia_protein_converter.py \\
        --data-dir data/raw/pdb_2021aug02_sample \\
        --output data/processed/wikipedia_protein \\
        --offset 20000 --limit 60000

    # Parallel: 4 workers processing PDB IDs 20000-100000
    python src/data/wikipedia_protein_converter.py \\
        --data-dir data/raw/pdb_2021aug02_sample \\
        --output data/processed/wikipedia_protein \\
        --offset 20000 --limit 100000 --parallel 4

    # Small test run (10 PDB IDs)
    python src/data/wikipedia_protein_converter.py \\
        --data-dir data/raw/pdb_2021aug02_sample \\
        --output /tmp/wiki_test \\
        --offset 0 --limit 10 --show-samples 2

API rate limits (no GPU or paid API required):
    - RCSB PDB: ~15 req/s (0.07s delay) — free public REST API
    - Wikipedia: ~5 req/s (0.2s delay) — free MediaWiki Action API

Estimated runtime:
    - ~20K PDB IDs: ~4-5 hours (serial)
    - ~80K PDB IDs: ~18 hours (serial), ~4.5 hours (4 workers)
"""

import csv
import json
import logging
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Instruction templates
# ---------------------------------------------------------------------------

OVERVIEW_INSTRUCTIONS = [
    "Describe this protein's biological role and key properties.",
    "Provide an overview of this protein, including its function and biological significance.",
    "What is known about this protein? Summarize its key biological characteristics.",
    "Analyze the following protein sequence and describe what is known about its biological role:",
    "Give a comprehensive overview of this protein's properties and biological importance.",
]

FUNCTION_INSTRUCTIONS = [
    "What is the function of this protein?",
    "Describe the biological function of the protein with the following sequence:",
    "Explain what this protein does in the cell.",
    "What role does this protein play in biological processes?",
    "Analyze the following protein sequence and describe its molecular function:",
]

STRUCTURE_INSTRUCTIONS = [
    "Describe the structure of this protein.",
    "What are the structural features of the protein with the following sequence?",
    "Provide details about this protein's three-dimensional structure.",
    "Characterize the structural properties of this protein.",
    "What is known about the structure and structural domains of this protein?",
]

DISEASE_INSTRUCTIONS = [
    "What diseases are associated with this protein?",
    "Describe the clinical significance of this protein.",
    "What is the role of this protein in human disease?",
    "Are there any disease associations known for this protein? Explain.",
    "Discuss the medical relevance and disease connections of this protein.",
]


# ---------------------------------------------------------------------------
# Wikitext cleanup
# ---------------------------------------------------------------------------

def _clean_wikitext(text: str) -> str:
    """Strip wikitext markup to produce plain text."""
    # Remove references: <ref>...</ref> and <ref ... />
    text = re.sub(r'<ref[^>]*>.*?</ref>', '', text, flags=re.DOTALL)
    text = re.sub(r'<ref[^/]*/>', '', text)
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Convert wiki links: [[Target|Display]] → Display, [[Target]] → Target
    text = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]', r'\1', text)
    # Remove templates: {{...}} (handle nested up to 2 levels)
    for _ in range(3):
        text = re.sub(r'\{\{[^{}]*\}\}', '', text)
    # Remove remaining curly braces
    text = re.sub(r'[{}]', '', text)
    # Remove bold/italic markup
    text = re.sub(r"'{2,5}", '', text)
    # Remove category links
    text = re.sub(r'\[\[Category:[^\]]+\]\]', '', text)
    # Remove external links: [http://... display] → display
    text = re.sub(r'\[https?://\S+\s+([^\]]+)\]', r'\1', text)
    text = re.sub(r'\[https?://\S+\]', '', text)
    # Collapse whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


def _parse_sections(wikitext: str) -> Dict[str, str]:
    """Parse wikitext into {section_name: cleaned_text} dict.

    Returns the intro (key="intro") and all level-2 sections.
    """
    sections: Dict[str, str] = {}

    # Split on level-2 headings: == Section Name ==
    parts = re.split(r'^(=={1,3}\s*.+?\s*=={1,3})\s*$', wikitext, flags=re.MULTILINE)

    # First part is intro (before any heading)
    intro = _clean_wikitext(parts[0]) if parts else ""
    if intro:
        sections["intro"] = intro

    # Subsequent parts alternate: heading, content, heading, content, ...
    i = 1
    while i < len(parts) - 1:
        heading_raw = parts[i].strip()
        content_raw = parts[i + 1] if i + 1 < len(parts) else ""
        # Extract heading text
        heading = re.sub(r'^=+\s*|\s*=+$', '', heading_raw).strip().lower()
        cleaned = _clean_wikitext(content_raw)
        if cleaned:
            sections[heading] = cleaned
        i += 2

    return sections


# ---------------------------------------------------------------------------
# API clients
# ---------------------------------------------------------------------------

RCSB_BASE = "https://data.rcsb.org/rest/v1/core"
WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"

RCSB_DELAY = 0.07   # ~15 req/s
WIKI_DELAY = 0.2     # ~5 req/s


def _rcsb_fetch_entry(pdb_id: str, session: requests.Session) -> Optional[Dict[str, Any]]:
    """Fetch entry metadata from RCSB PDB."""
    url = f"{RCSB_BASE}/entry/{pdb_id}"
    try:
        resp = session.get(url, timeout=10)
        if resp.status_code != 200:
            return None
        data = resp.json()

        struct = data.get("struct", {})
        title = struct.get("title", "")

        # Organism from entity source
        exptl = data.get("exptl", [{}])
        method = exptl[0].get("method", "") if exptl else ""

        cell = data.get("cell", {})

        # Resolution
        refine = data.get("rcsb_entry_info", {})
        resolution = refine.get("resolution_combined", [None])
        if isinstance(resolution, list):
            resolution = resolution[0] if resolution else None

        return {
            "title": title,
            "method": method,
            "resolution": resolution,
        }
    except (requests.RequestException, ValueError, KeyError):
        return None


def _rcsb_fetch_uniprot(pdb_id: str, session: requests.Session) -> List[str]:
    """Fetch UniProt accessions for a PDB entry."""
    url = f"{RCSB_BASE}/uniprot/{pdb_id}"
    try:
        resp = session.get(url, timeout=10)
        if resp.status_code != 200:
            return []
        data = resp.json()
        # Response is a list of uniprot entries
        if isinstance(data, list):
            accessions = []
            for entry in data:
                acc = entry.get("rcsb_uniprot_container_identifiers", {}).get(
                    "uniprot_id"
                )
                if acc:
                    accessions.append(acc)
            return accessions
        return []
    except (requests.RequestException, ValueError, KeyError):
        return []


def _rcsb_fetch_organism(pdb_id: str, session: requests.Session) -> Optional[str]:
    """Fetch organism from RCSB polymer entity."""
    url = f"{RCSB_BASE}/polymer_entity/{pdb_id}/1"
    try:
        resp = session.get(url, timeout=10)
        if resp.status_code != 200:
            return None
        data = resp.json()
        sources = data.get("rcsb_entity_source_organism", [])
        if sources:
            return sources[0].get("scientific_name")
        return None
    except (requests.RequestException, ValueError, KeyError):
        return None


def _wikipedia_search(query: str, session: requests.Session) -> Optional[str]:
    """Search Wikipedia for a protein article. Returns page title or None."""
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srnamespace": "0",
        "srlimit": "1",
        "format": "json",
    }
    try:
        resp = session.get(WIKIPEDIA_API, params=params, timeout=10)
        if resp.status_code != 200:
            return None
        data = resp.json()
        results = data.get("query", {}).get("search", [])
        if results:
            return results[0]["title"]
        return None
    except (requests.RequestException, ValueError, KeyError):
        return None


def _wikipedia_get_wikitext(title: str, session: requests.Session) -> Optional[str]:
    """Fetch raw wikitext for a Wikipedia page."""
    params = {
        "action": "parse",
        "page": title,
        "prop": "wikitext",
        "format": "json",
    }
    try:
        resp = session.get(WIKIPEDIA_API, params=params, timeout=15)
        if resp.status_code != 200:
            return None
        data = resp.json()
        return data.get("parse", {}).get("wikitext", {}).get("*")
    except (requests.RequestException, ValueError, KeyError):
        return None


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _load_cache(path: Path) -> Dict:
    """Load JSON cache file, return empty dict if missing."""
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return {}


def _save_cache(path: Path, data: Dict) -> None:
    """Save dict to JSON cache file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=1)


# ---------------------------------------------------------------------------
# SFT record generation
# ---------------------------------------------------------------------------

def _format_input(sequence: str) -> str:
    """Wrap sequence in backtick block matching Mol-Instructions format."""
    return f"```\n{sequence}\n```"


# Section keys we look for (case-insensitive after lowering)
_FUNCTION_KEYS = {"function", "biological function", "molecular function", "activity",
                  "mechanism", "mechanism of action", "biochemistry"}
_STRUCTURE_KEYS = {"structure", "protein structure", "3d structure", "structural features",
                   "crystal structure", "tertiary structure", "domain structure",
                   "structure and function"}
_DISEASE_KEYS = {"clinical significance", "disease", "clinical relevance",
                 "pathology", "disease association", "disease associations",
                 "medical significance", "role in disease", "disease relevance",
                 "mutations"}


def _find_section(sections: Dict[str, str], keys: set) -> Optional[str]:
    """Find first matching section by key set."""
    for key in keys:
        if key in sections:
            return sections[key]
    return None


def _make_overview_record(
    protein_name: str,
    sequence: str,
    intro: str,
    organism: Optional[str],
    method: Optional[str],
    pdb_id: str,
    uniprot_id: str,
    rng: random.Random,
) -> Dict[str, Any]:
    """Create protein_overview SFT record from Wikipedia intro."""
    instruction = rng.choice(OVERVIEW_INSTRUCTIONS)

    parts = [intro]
    if organism:
        parts.append(f"This protein is found in {organism}.")
    if method:
        parts.append(f"Its structure has been determined by {method}.")
    output = " ".join(parts)

    return {
        "instruction": instruction,
        "input": _format_input(sequence),
        "output": output,
        "metadata": {
            "seq_len": len(sequence),
            "task": "protein_overview",
            "source": "wikipedia_pdb",
            "pdb_id": pdb_id,
            "uniprot_id": uniprot_id,
            "organism": organism or "",
            "protein_name": protein_name,
        },
    }


def _make_section_record(
    task: str,
    instructions: List[str],
    section_text: str,
    protein_name: str,
    sequence: str,
    pdb_id: str,
    uniprot_id: str,
    organism: Optional[str],
    rng: random.Random,
) -> Dict[str, Any]:
    """Create a section-based SFT record."""
    instruction = rng.choice(instructions)

    return {
        "instruction": instruction,
        "input": _format_input(sequence),
        "output": section_text,
        "metadata": {
            "seq_len": len(sequence),
            "task": task,
            "source": "wikipedia_pdb",
            "pdb_id": pdb_id,
            "uniprot_id": uniprot_id,
            "organism": organism or "",
            "protein_name": protein_name,
        },
    }


# ---------------------------------------------------------------------------
# Main conversion pipeline
# ---------------------------------------------------------------------------

def convert_wikipedia_protein(
    data_dir: Path,
    output_dir: Path,
    cache_dir: Optional[Path] = None,
    min_length: int = 50,
    max_length: int = 1000,
    offset: int = 0,
    limit: Optional[int] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Convert IPD-PDB data to Wikipedia-enriched SFT instruction pairs.

    This is the main conversion function. It runs through all 6 pipeline
    stages sequentially. For parallel execution, use ``run_parallel()``
    instead, which spawns multiple instances of this function.

    Incremental processing:
        Use ``offset`` and ``limit`` to process a specific slice of PDB IDs.
        Stage 6 appends new records to any existing output files and
        deduplicates by pdb_id, making it safe to run incrementally:

        - Run 1: offset=0, limit=20000  → processes first 20K PDB IDs
        - Run 2: offset=20000, limit=60000 → appends next 40K, keeps the 20K

    Caching:
        API responses are cached in ``cache_dir`` (default: ``output_dir/.cache``).
        If a PDB ID is already in the cache, the API call is skipped. This
        makes re-runs fast for already-processed ranges.

    Args:
        data_dir: Path to pdb_2021aug02_sample directory (must contain list.csv).
        output_dir: Directory to write output JSON files.
        cache_dir: Directory for API response caches. Default: ``output_dir/.cache``.
            When running in parallel, each worker should use a separate cache_dir.
        min_length: Minimum protein sequence length to include (default: 50).
        max_length: Maximum protein sequence length to include (default: 1000).
        offset: Start index into the sorted PDB ID list. PDB IDs before this
            index are skipped entirely (no API calls). Default: 0.
        limit: End index into the PDB ID list (exclusive). None means process
            all remaining PDB IDs. Used with offset for incremental runs.
        seed: Random seed for instruction template selection (default: 42).

    Returns:
        Dict with conversion statistics including record counts per task,
        total chains, unique proteins, and skip counts.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if cache_dir is None:
        cache_dir = output_dir / ".cache"
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    csv_path = Path(data_dir) / "list.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"list.csv not found in {data_dir}")

    # --- Stage 1: Extract unique PDB IDs and representative sequences ---
    logger.info("Stage 1: Extracting unique PDB IDs from list.csv...")
    pdb_sequences: Dict[str, Tuple[str, str]] = {}  # pdb_id -> (chain_id, sequence)
    total_chains = 0
    skipped_length = 0

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_chains += 1
            chain_id = row["CHAINID"]
            sequence = row["SEQUENCE"]
            pdb_id = chain_id.split("_")[0].lower()

            if len(sequence) < min_length or len(sequence) > max_length:
                skipped_length += 1
                continue

            # Keep first (longest) sequence per PDB ID
            if pdb_id not in pdb_sequences or len(sequence) > len(pdb_sequences[pdb_id][1]):
                pdb_sequences[pdb_id] = (chain_id, sequence)

    all_pdb_ids = list(pdb_sequences.keys())
    all_pdb_ids = all_pdb_ids[offset:limit]
    logger.info(f"  {total_chains} chains → {len(all_pdb_ids)} PDB IDs "
                f"(offset={offset}, limit={limit}, skipped {skipped_length} by length)")

    # --- Stage 2: Fetch RCSB metadata ---
    logger.info("Stage 2: Fetching RCSB metadata...")
    rcsb_cache_path = cache_dir / "rcsb_metadata.json"
    rcsb_cache = _load_cache(rcsb_cache_path)

    session = requests.Session()
    session.headers.update({"User-Agent": "ProteinLLM-DataPipeline/1.0"})

    new_fetches = 0
    for i, pdb_id in enumerate(all_pdb_ids):
        if pdb_id in rcsb_cache:
            continue

        entry = _rcsb_fetch_entry(pdb_id, session)
        time.sleep(RCSB_DELAY)

        if entry is None:
            rcsb_cache[pdb_id] = None
            new_fetches += 1
            if new_fetches % 100 == 0:
                _save_cache(rcsb_cache_path, rcsb_cache)
            continue

        uniprot_ids = _rcsb_fetch_uniprot(pdb_id, session)
        time.sleep(RCSB_DELAY)

        organism = _rcsb_fetch_organism(pdb_id, session)
        time.sleep(RCSB_DELAY)

        rcsb_cache[pdb_id] = {
            "title": entry.get("title", ""),
            "method": entry.get("method", ""),
            "resolution": entry.get("resolution"),
            "uniprot_ids": uniprot_ids,
            "organism": organism,
        }
        new_fetches += 1

        if new_fetches % 100 == 0:
            _save_cache(rcsb_cache_path, rcsb_cache)
            logger.info(f"  RCSB: {new_fetches} new fetches, {i+1}/{len(all_pdb_ids)} total")

    if new_fetches > 0:
        _save_cache(rcsb_cache_path, rcsb_cache)
    logger.info(f"  RCSB metadata: {new_fetches} new fetches, "
                f"{sum(1 for v in rcsb_cache.values() if v)} entries with data")

    # --- Stage 3: Deduplicate by UniProt ID ---
    logger.info("Stage 3: Deduplicating by UniProt ID...")
    seen_uniprot: Dict[str, str] = {}  # uniprot_id -> pdb_id
    unique_pdb_ids = []

    for pdb_id in all_pdb_ids:
        meta = rcsb_cache.get(pdb_id)
        if meta is None:
            continue

        uniprot_ids = meta.get("uniprot_ids", [])
        if uniprot_ids:
            primary_uniprot = uniprot_ids[0]
            if primary_uniprot in seen_uniprot:
                continue
            seen_uniprot[primary_uniprot] = pdb_id

        unique_pdb_ids.append(pdb_id)

    logger.info(f"  {len(all_pdb_ids)} PDB IDs → {len(unique_pdb_ids)} unique proteins")

    # --- Stage 4: Fetch Wikipedia articles ---
    logger.info("Stage 4: Fetching Wikipedia articles...")
    wiki_cache_path = cache_dir / "wikipedia_articles.json"
    wiki_cache = _load_cache(wiki_cache_path)

    new_wiki_fetches = 0
    wiki_found = 0

    for i, pdb_id in enumerate(unique_pdb_ids):
        meta = rcsb_cache.get(pdb_id)
        if meta is None:
            continue

        protein_name = meta.get("title", "")
        if not protein_name:
            continue

        # Use protein title as cache key
        cache_key = protein_name.lower().strip()
        if cache_key in wiki_cache:
            if wiki_cache[cache_key] is not None:
                wiki_found += 1
            continue

        # Search Wikipedia
        # Try protein name first, then add "protein" suffix
        page_title = _wikipedia_search(f"{protein_name} protein", session)
        time.sleep(WIKI_DELAY)

        if page_title is None:
            # Retry with just the protein name
            page_title = _wikipedia_search(protein_name, session)
            time.sleep(WIKI_DELAY)

        if page_title is None:
            wiki_cache[cache_key] = None
            new_wiki_fetches += 1
            if new_wiki_fetches % 50 == 0:
                _save_cache(wiki_cache_path, wiki_cache)
            continue

        # Fetch article wikitext
        wikitext = _wikipedia_get_wikitext(page_title, session)
        time.sleep(WIKI_DELAY)

        if wikitext is None:
            wiki_cache[cache_key] = None
        else:
            sections = _parse_sections(wikitext)
            # Check if article has enough content (not a stub)
            total_text = " ".join(sections.values())
            if len(total_text) < 200:
                wiki_cache[cache_key] = None
            else:
                wiki_cache[cache_key] = {
                    "page_title": page_title,
                    "sections": sections,
                }
                wiki_found += 1

        new_wiki_fetches += 1
        if new_wiki_fetches % 50 == 0:
            _save_cache(wiki_cache_path, wiki_cache)
            logger.info(f"  Wikipedia: {new_wiki_fetches} new fetches, "
                        f"{wiki_found} articles found, "
                        f"{i+1}/{len(unique_pdb_ids)} total")

    if new_wiki_fetches > 0:
        _save_cache(wiki_cache_path, wiki_cache)
    logger.info(f"  Wikipedia: {new_wiki_fetches} new fetches, {wiki_found} articles found")

    # --- Stage 5: Generate SFT records ---
    logger.info("Stage 5: Generating SFT records...")
    records_by_task: Dict[str, List] = {
        "protein_overview": [],
        "protein_function": [],
        "protein_structure": [],
        "disease_association": [],
    }
    stats = {
        "total_chains": total_chains,
        "unique_pdb_ids": len(all_pdb_ids),
        "unique_proteins": len(unique_pdb_ids),
        "wikipedia_articles_found": wiki_found,
        "skipped_length": skipped_length,
        "skipped_no_rcsb": 0,
        "skipped_no_wikipedia": 0,
    }

    for pdb_id in unique_pdb_ids:
        meta = rcsb_cache.get(pdb_id)
        if meta is None:
            stats["skipped_no_rcsb"] += 1
            continue

        protein_name = meta.get("title", "")
        if not protein_name:
            stats["skipped_no_rcsb"] += 1
            continue

        cache_key = protein_name.lower().strip()
        wiki_entry = wiki_cache.get(cache_key)
        if wiki_entry is None:
            stats["skipped_no_wikipedia"] += 1
            continue

        chain_id, sequence = pdb_sequences[pdb_id]
        sections = wiki_entry.get("sections", {})
        organism = meta.get("organism")
        method = meta.get("method")
        uniprot_ids = meta.get("uniprot_ids", [])
        uniprot_id = uniprot_ids[0] if uniprot_ids else ""

        # Overview from intro
        intro = sections.get("intro", "")
        if intro and len(intro) >= 50:
            records_by_task["protein_overview"].append(
                _make_overview_record(
                    protein_name, sequence, intro, organism, method,
                    pdb_id, uniprot_id, rng
                )
            )

        # Function section
        func_text = _find_section(sections, _FUNCTION_KEYS)
        if func_text and len(func_text) >= 50:
            records_by_task["protein_function"].append(
                _make_section_record(
                    "protein_function", FUNCTION_INSTRUCTIONS, func_text,
                    protein_name, sequence, pdb_id, uniprot_id, organism, rng
                )
            )

        # Structure section
        struct_text = _find_section(sections, _STRUCTURE_KEYS)
        if struct_text and len(struct_text) >= 50:
            records_by_task["protein_structure"].append(
                _make_section_record(
                    "protein_structure", STRUCTURE_INSTRUCTIONS, struct_text,
                    protein_name, sequence, pdb_id, uniprot_id, organism, rng
                )
            )

        # Disease/clinical significance section
        disease_text = _find_section(sections, _DISEASE_KEYS)
        if disease_text and len(disease_text) >= 50:
            records_by_task["disease_association"].append(
                _make_section_record(
                    "disease_association", DISEASE_INSTRUCTIONS, disease_text,
                    protein_name, sequence, pdb_id, uniprot_id, organism, rng
                )
            )

    # --- Stage 6: Write output (append to existing files, deduplicate) ---
    # This stage supports incremental runs: if output files already exist
    # from a previous run, new records are appended rather than overwriting.
    # Deduplication by pdb_id prevents duplicates when ranges overlap.
    logger.info("Stage 6: Writing output files (appending to existing)...")
    for task_name, new_records in records_by_task.items():
        output_path = output_dir / f"{task_name}.json"

        # Load any existing records from a previous run
        if output_path.exists():
            with open(output_path, "r") as f:
                existing = json.load(f)
        else:
            existing = []

        # Build a set of already-seen pdb_ids to skip duplicates.
        # This handles the case where offset ranges overlap between runs.
        seen_pdb_ids = {r["metadata"]["pdb_id"] for r in existing}
        deduped = [r for r in new_records if r["metadata"]["pdb_id"] not in seen_pdb_ids]

        # Append new unique records after existing ones
        combined = existing + deduped

        with open(output_path, "w") as f:
            json.dump(combined, f, indent=2)

        stats[f"records_{task_name}"] = len(combined)
        logger.info(f"  {output_path.name}: {len(existing)} existing + "
                     f"{len(deduped)} new = {len(combined)} total")

    total_records = sum(
        stats.get(f"records_{t}", 0) for t in records_by_task
    )
    stats["total_records"] = total_records
    logger.info(f"Wikipedia protein conversion complete: {total_records} total records")

    # Update cumulative stats by merging with any existing stats file.
    # This ensures stats reflect ALL runs, not just the current one.
    # Additive keys (unique_pdb_ids, etc.) are summed across runs.
    # total_chains is the same constant from the CSV, so we take max.
    stats_path = output_dir / "conversion_stats.json"
    if stats_path.exists():
        with open(stats_path, "r") as f:
            prev_stats = json.load(f)
        for key in ("unique_pdb_ids", "unique_proteins", "wikipedia_articles_found",
                     "skipped_length", "skipped_no_rcsb", "skipped_no_wikipedia"):
            if key in prev_stats and key in stats:
                stats[key] = prev_stats[key] + stats[key]
        stats["total_chains"] = max(stats["total_chains"], prev_stats.get("total_chains", 0))

    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    return stats


# ---------------------------------------------------------------------------
# prepare_data.py integration
# ---------------------------------------------------------------------------

def prepare_wikipedia_protein(
    raw_dir: Path, processed_dir: Path, cfg: Any = None
) -> Dict[str, Any]:
    """Entry point for ``prepare_data.py`` integration (Hydra config-driven).

    Wraps ``convert_wikipedia_protein()`` with config extraction from Hydra.
    Supports ``offset`` and ``limit`` via ``cfg.filters.offset`` / ``cfg.filters.limit``.

    Args:
        raw_dir: Path to pdb_2021aug02_sample directory (must contain list.csv).
        processed_dir: Output directory for JSON files.
        cfg: Optional Hydra DictConfig. Looks for ``filters.min_length``,
            ``filters.max_length``, ``filters.offset``, ``filters.limit``.

    Returns:
        Conversion statistics dict.
    """
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)

    csv_path = raw_dir / "list.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"list.csv not found in {raw_dir}. "
            "Run: python src/data/download.py --dataset ipd_pdb_sample"
        )

    # Extract config filters
    min_length = 50
    max_length = 1000
    offset = 0
    limit = None

    if cfg is not None:
        filters = cfg.get("filters", cfg.get("data", {}).get("filters", {}))
        if hasattr(filters, "to_container"):
            filters = filters.to_container(resolve=True)
        elif not isinstance(filters, dict):
            filters = {}
        min_length = filters.get("min_length", 50)
        max_length = filters.get("max_length", 1000)
        offset = filters.get("offset", 0)
        limit = filters.get("limit", None)

    return convert_wikipedia_protein(
        data_dir=raw_dir,
        output_dir=processed_dir,
        min_length=min_length,
        max_length=max_length,
        offset=offset,
        limit=limit,
    )


# ---------------------------------------------------------------------------
# Parallel execution + merge
# ---------------------------------------------------------------------------

# Output JSON file names (one per SFT task type). Used by merge_shards()
# and run_parallel() to iterate over all output files consistently.
TASK_FILES = ["protein_overview", "protein_function", "protein_structure",
              "disease_association"]


def merge_shards(output_dir: Path, shard_dirs: List[Path]) -> Dict[str, Any]:
    """Merge shard output directories into the main output directory.

    Called automatically by ``run_parallel()`` after all workers finish.
    Can also be called manually if you ran shards independently.

    For each task file (protein_overview, protein_function, etc.):
      1. Load existing records from ``output_dir/{task}.json`` (if any)
      2. Load records from each shard directory
      3. Deduplicate by ``metadata.pdb_id`` — existing records take priority
      4. Write combined records back to ``output_dir/{task}.json``

    Statistics from ``conversion_stats.json`` in each shard are accumulated
    and merged with the main output directory's existing stats.

    Args:
        output_dir: Main output directory. May already contain records from
            previous runs — these are preserved and not duplicated.
        shard_dirs: List of shard directories to merge from. Each should
            contain the same JSON file structure as output_dir.

    Returns:
        Dict with cumulative merged statistics.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    merged_stats: Dict[str, Any] = {}

    # --- Merge task JSON files ---
    for task_name in TASK_FILES:
        output_path = output_dir / f"{task_name}.json"

        # Start with existing records in the main output directory
        if output_path.exists():
            with open(output_path, "r") as f:
                combined = json.load(f)
        else:
            combined = []

        # Track seen pdb_ids to prevent duplicates across shards
        seen_pdb_ids = {r["metadata"]["pdb_id"] for r in combined}
        total_new = 0

        # Append unique records from each shard
        for shard_dir in shard_dirs:
            shard_path = shard_dir / f"{task_name}.json"
            if not shard_path.exists():
                continue
            with open(shard_path, "r") as f:
                shard_records = json.load(f)
            for r in shard_records:
                pid = r["metadata"]["pdb_id"]
                if pid not in seen_pdb_ids:
                    seen_pdb_ids.add(pid)
                    combined.append(r)
                    total_new += 1

        with open(output_path, "w") as f:
            json.dump(combined, f, indent=2)

        merged_stats[f"records_{task_name}"] = len(combined)
        logger.info(f"  {task_name}: {total_new} new from shards → {len(combined)} total")

    merged_stats["total_records"] = sum(
        merged_stats.get(f"records_{t}", 0) for t in TASK_FILES
    )

    # --- Merge conversion statistics ---
    # These keys are additive across shards (each shard processes different PDB IDs)
    accum_keys = ("unique_pdb_ids", "unique_proteins", "wikipedia_articles_found",
                  "skipped_length", "skipped_no_rcsb", "skipped_no_wikipedia")

    # Sum stats from all shards
    for shard_dir in shard_dirs:
        shard_stats_path = shard_dir / "conversion_stats.json"
        if not shard_stats_path.exists():
            continue
        with open(shard_stats_path, "r") as f:
            shard_stats = json.load(f)
        for key in accum_keys:
            merged_stats[key] = merged_stats.get(key, 0) + shard_stats.get(key, 0)
        # total_chains is the same across all shards (full CSV), so take max
        merged_stats["total_chains"] = max(
            merged_stats.get("total_chains", 0), shard_stats.get("total_chains", 0)
        )

    # Add to any existing main stats from previous runs
    main_stats_path = output_dir / "conversion_stats.json"
    if main_stats_path.exists():
        with open(main_stats_path, "r") as f:
            prev_stats = json.load(f)
        for key in accum_keys:
            merged_stats[key] = merged_stats.get(key, 0) + prev_stats.get(key, 0)
        merged_stats["total_chains"] = max(
            merged_stats.get("total_chains", 0), prev_stats.get("total_chains", 0)
        )

    with open(main_stats_path, "w") as f:
        json.dump(merged_stats, f, indent=2)

    return merged_stats


def run_parallel(
    data_dir: Path,
    output_dir: Path,
    offset: int,
    limit: int,
    workers: int = 4,
    min_length: int = 50,
    max_length: int = 1000,
) -> Dict[str, Any]:
    """Run conversion in parallel using separate shard directories.

    Splits the PDB ID range [offset, limit) into ``workers`` non-overlapping
    chunks. Each chunk runs as a subprocess with its own output and cache
    directory under ``output_dir/.shards/shard_N/``. This avoids file
    collisions — each worker reads/writes only its own shard.

    After all workers finish, ``merge_shards()`` combines results into the
    main ``output_dir``, deduplicating by pdb_id. Shard directories are then
    cleaned up.

    Directory layout during execution::

        output_dir/
        ├── protein_overview.json      ← existing records (preserved)
        ├── ...
        └── .shards/                   ← temporary, deleted after merge
            ├── shard_0/
            │   ├── protein_overview.json
            │   ├── ...
            │   └── .cache/            ← separate cache per worker
            ├── shard_0.log
            ├── shard_1/
            │   └── ...
            └── shard_1.log

    Worker failure handling:
        If a worker fails, its shard is excluded from the merge but other
        workers' results are still merged. Check shard_N.log for errors.
        You can re-run with the failed worker's offset/limit range to retry.

    Args:
        data_dir: Path to IPD-PDB directory (must contain list.csv).
        output_dir: Main output directory for final merged results.
            Existing records in this directory are preserved.
        offset: Start index into PDB ID list (0-based).
        limit: End index into PDB ID list (exclusive). Required.
        workers: Number of parallel workers (default: 4).
        min_length: Minimum protein sequence length filter (default: 50).
        max_length: Maximum protein sequence length filter (default: 1000).

    Returns:
        Dict with merged statistics from all successful shards.

    Example:
        >>> # Process PDB IDs 20000-100000 with 4 parallel workers
        >>> stats = run_parallel(
        ...     data_dir=Path("data/raw/pdb_2021aug02_sample"),
        ...     output_dir=Path("data/processed/wikipedia_protein"),
        ...     offset=20000, limit=100000, workers=4,
        ... )
    """
    import shutil
    import subprocess

    # Divide the [offset, limit) range into equal chunks, one per worker.
    # ceiling division ensures all PDB IDs are covered.
    chunk_size = (limit - offset + workers - 1) // workers
    shard_base = output_dir / ".shards"
    shard_base.mkdir(parents=True, exist_ok=True)

    shard_dirs = []
    processes = []

    # Re-invoke this same script as a subprocess for each shard
    script_path = Path(__file__).resolve()

    for i in range(workers):
        chunk_start = offset + i * chunk_size
        chunk_end = min(offset + (i + 1) * chunk_size, limit)
        if chunk_start >= limit:
            break  # fewer chunks than workers (small range)

        # Each shard gets its own output dir + cache dir to avoid collisions
        shard_dir = shard_base / f"shard_{i}"
        shard_cache = shard_dir / ".cache"
        shard_dirs.append(shard_dir)

        # Build CLI command for the worker subprocess (serial mode, no --parallel)
        cmd = [
            "python3", str(script_path),
            "--data-dir", str(data_dir),
            "--output", str(shard_dir),
            "--cache-dir", str(shard_cache),
            "--offset", str(chunk_start),
            "--limit", str(chunk_end),
            "--min-length", str(min_length),
            "--max-length", str(max_length),
        ]

        log_path = shard_base / f"shard_{i}.log"
        logger.info(f"  Worker {i}: PDB IDs [{chunk_start}, {chunk_end}) → {shard_dir}")
        log_file = open(log_path, "w")
        proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
        processes.append((proc, log_file, i))

    logger.info(f"Launched {len(processes)} workers. Waiting for completion...")

    # Block until all workers finish
    failed = []
    for proc, log_file, idx in processes:
        proc.wait()
        log_file.close()
        if proc.returncode != 0:
            failed.append(idx)
            logger.error(f"  Worker {idx} failed (exit code {proc.returncode}). "
                         f"See {shard_base / f'shard_{idx}.log'}")
        else:
            logger.info(f"  Worker {idx} finished successfully")

    if failed:
        logger.error(f"Workers {failed} failed. Merging successful shards only.")

    # Merge all successful shards into the main output directory
    logger.info("Merging shards into main output...")
    successful_shards = [d for i, d in enumerate(shard_dirs) if i not in failed]
    merged_stats = merge_shards(output_dir, successful_shards)

    # Remove temporary shard directories (logs + data)
    logger.info("Cleaning up shard directories...")
    shutil.rmtree(shard_base, ignore_errors=True)

    return merged_stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Convert IPD-PDB proteins to Wikipedia-enriched SFT instruction pairs.",
        epilog="""
Examples:
  # First run (serial, first 20K PDB IDs):
  %(prog)s --limit 20000

  # Expand to 60K (appends to existing, skips first 20K):
  %(prog)s --offset 20000 --limit 60000

  # Parallel expansion (4 workers, ~4x faster):
  %(prog)s --offset 20000 --limit 100000 --parallel 4

  # Quick test (10 PDB IDs, show 2 sample records):
  %(prog)s --output /tmp/wiki_test --limit 10 --show-samples 2
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-dir", type=str,
        default="data/raw/pdb_2021aug02_sample",
        help="Path to IPD-PDB directory containing list.csv "
             "(default: data/raw/pdb_2021aug02_sample)",
    )
    parser.add_argument(
        "--output", type=str,
        default="data/processed/wikipedia_protein",
        help="Output directory for JSON files. Existing files are preserved "
             "and new records are appended (default: data/processed/wikipedia_protein)",
    )
    parser.add_argument(
        "--cache-dir", type=str, default=None,
        help="Cache directory for RCSB/Wikipedia API responses. "
             "Default: OUTPUT/.cache. Use separate cache dirs for parallel workers.",
    )
    parser.add_argument(
        "--min-length", type=int, default=50,
        help="Minimum protein sequence length to include (default: 50)",
    )
    parser.add_argument(
        "--max-length", type=int, default=1000,
        help="Maximum protein sequence length to include (default: 1000)",
    )
    parser.add_argument(
        "--offset", type=int, default=0,
        help="Start index into sorted PDB ID list (0-based). "
             "PDB IDs before this index are skipped entirely (default: 0)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="End index into PDB ID list (exclusive). None = process all. "
             "Required when using --parallel.",
    )
    parser.add_argument(
        "--parallel", type=int, default=0,
        help="Number of parallel workers (default: 0 = serial mode). "
             "Each worker gets its own shard directory and cache. "
             "Results are merged after all workers finish.",
    )
    parser.add_argument(
        "--show-samples", type=int, default=0,
        help="Print N sample records from each output file after conversion",
    )
    args = parser.parse_args()

    # Dispatch to parallel or serial mode
    if args.parallel > 0:
        if args.limit is None:
            parser.error("--limit is required when using --parallel")
        stats = run_parallel(
            data_dir=Path(args.data_dir),
            output_dir=Path(args.output),
            offset=args.offset,
            limit=args.limit,
            workers=args.parallel,
            min_length=args.min_length,
            max_length=args.max_length,
        )
    else:
        stats = convert_wikipedia_protein(
            data_dir=Path(args.data_dir),
            output_dir=Path(args.output),
            cache_dir=Path(args.cache_dir) if args.cache_dir else None,
            min_length=args.min_length,
            max_length=args.max_length,
            offset=args.offset,
            limit=args.limit,
        )

    print("\n=== Conversion Statistics ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    if args.show_samples > 0:
        output_dir = Path(args.output)
        for json_file in sorted(output_dir.glob("*.json")):
            if json_file.name == "conversion_stats.json":
                continue
            with open(json_file) as f:
                records = json.load(f)
            print(f"\n=== {json_file.name} (showing {min(args.show_samples, len(records))} of {len(records)}) ===")
            for rec in records[:args.show_samples]:
                print(json.dumps(rec, indent=2))
                print()
