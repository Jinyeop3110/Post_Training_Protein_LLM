"""Structural property extraction from ESMFold predictions.

Converts ESMFold atom37 coordinates to biotite AtomArray and extracts:
- Secondary structure (SS3: H=helix, E=sheet, C=coil)
- Backbone dihedral angles (phi, psi)
- Solvent-accessible surface area (SASA / RSA)
- Contact map (Cβ-Cβ distance < 8Å)

Usage:
    from src.models.esmfold_wrapper import get_esmfold_predictor
    from src.utils.structure_properties import extract_properties

    predictor = get_esmfold_predictor()
    full_output = predictor.predict_full("MKTAYIAK...")
    props = extract_properties(full_output)
    print(props["ss3_string"])      # "CCHHHHHHEEEECC"
    print(props["helix_fraction"])  # 0.43
"""

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

# ESMFold atom37 atom names (from openfold residue_constants)
_ATOM37_NAMES = [
    "N", "CA", "C", "CB", "O", "CG", "CG1", "CG2", "OG", "OG1", "SG",
    "CD", "CD1", "CD2", "ND1", "ND2", "OD1", "OD2", "SD", "CE", "CE1",
    "CE2", "CE3", "NE", "NE1", "NE2", "OE1", "OE2", "CH2", "NH1", "NH2",
    "OH", "CZ", "CZ2", "CZ3", "NZ", "OXT",
]

# Standard 1-letter to 3-letter amino acid mapping
_AA_1TO3 = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}

# Max SASA per residue type (Tien et al. 2013, theoretical) for RSA normalization
_MAX_ASA = {
    "ALA": 129.0, "ARG": 274.0, "ASN": 195.0, "ASP": 193.0, "CYS": 167.0,
    "GLN": 225.0, "GLU": 223.0, "GLY": 104.0, "HIS": 224.0, "ILE": 197.0,
    "LEU": 201.0, "LYS": 236.0, "MET": 224.0, "PHE": 240.0, "PRO": 159.0,
    "SER": 155.0, "THR": 172.0, "TRP": 285.0, "TYR": 263.0, "VAL": 174.0,
}

# Ramachandran region classification
_RAMA_REGIONS = {
    "alpha_helix": (-120, -40, -80, -10),   # phi_min, phi_max, psi_min, psi_max
    "beta_sheet": (-180, -50, 80, 180),
    "left_helix": (40, 100, 10, 80),
}


def atom37_to_biotite(
    positions: torch.Tensor,
    atom37_mask: torch.Tensor,
    sequence: str,
):
    """Convert ESMFold atom37 output to biotite AtomArray.

    Args:
        positions: Atom coordinates, shape [L, 37, 3].
        atom37_mask: Atom existence mask, shape [L, 37].
        sequence: Amino acid sequence (1-letter codes).

    Returns:
        biotite AtomArray with all existing atoms.
    """
    import biotite.structure as struc

    coords_np = positions.numpy() if isinstance(positions, torch.Tensor) else positions
    mask_np = atom37_mask.numpy() if isinstance(atom37_mask, torch.Tensor) else atom37_mask

    # Collect atoms
    atom_coords = []
    atom_names = []
    res_ids = []
    res_names = []
    elements = []
    chain_ids = []

    L = len(sequence)
    for i in range(L):
        aa_1 = sequence[i]
        aa_3 = _AA_1TO3.get(aa_1, "UNK")
        for j in range(37):
            if mask_np[i, j] > 0.5:
                atom_coords.append(coords_np[i, j])
                atom_names.append(_ATOM37_NAMES[j])
                res_ids.append(i + 1)  # 1-indexed
                res_names.append(aa_3)
                elements.append(_ATOM37_NAMES[j][0])  # First char = element
                chain_ids.append("A")

    if not atom_coords:
        return None

    atom_array = struc.AtomArray(len(atom_coords))
    atom_array.coord = np.array(atom_coords, dtype=np.float32)
    atom_array.atom_name = np.array(atom_names)
    atom_array.res_id = np.array(res_ids)
    atom_array.res_name = np.array(res_names)
    atom_array.element = np.array(elements)
    atom_array.chain_id = np.array(chain_ids)

    return atom_array


def compute_secondary_structure(atom_array) -> Tuple[str, Dict[str, float]]:
    """Compute per-residue secondary structure using P-SEA algorithm.

    Args:
        atom_array: biotite AtomArray.

    Returns:
        Tuple of (ss3_string, fractions_dict).
        ss3_string: Per-residue SS as "H"=helix, "E"=sheet, "C"=coil.
        fractions_dict: {"helix_fraction", "sheet_fraction", "coil_fraction"}.
    """
    import biotite.structure as struc

    sse = struc.annotate_sse(atom_array)

    # Map biotite codes: 'a'→H, 'b'→E, 'c'/''→C
    ss_map = {"a": "H", "b": "E", "c": "C", "": "C"}
    ss3_list = [ss_map.get(s, "C") for s in sse]
    ss3_string = "".join(ss3_list)

    L = len(ss3_list)
    if L == 0:
        return "", {"helix_fraction": 0.0, "sheet_fraction": 0.0, "coil_fraction": 0.0}

    fractions = {
        "helix_fraction": round(ss3_list.count("H") / L, 4),
        "sheet_fraction": round(ss3_list.count("E") / L, 4),
        "coil_fraction": round(ss3_list.count("C") / L, 4),
    }
    return ss3_string, fractions


def compute_dihedrals(atom_array) -> Tuple[np.ndarray, np.ndarray]:
    """Compute backbone phi/psi dihedral angles.

    Args:
        atom_array: biotite AtomArray.

    Returns:
        Tuple of (phi, psi) arrays in degrees. NaN for terminal residues.
    """
    import biotite.structure as struc

    phi, psi, _ = struc.dihedral_backbone(atom_array)
    # Convert radians to degrees
    phi_deg = np.degrees(phi)
    psi_deg = np.degrees(psi)
    return phi_deg, psi_deg


def classify_ramachandran(phi: float, psi: float) -> str:
    """Classify a phi/psi pair into a Ramachandran region.

    Returns one of: "alpha", "beta", "left_helix", "other".
    """
    if np.isnan(phi) or np.isnan(psi):
        return "other"

    if -120 <= phi <= -40 and -80 <= psi <= -10:
        return "alpha"
    if -180 <= phi <= -50 and (80 <= psi <= 180 or -180 <= psi <= -120):
        return "beta"
    if 40 <= phi <= 100 and 10 <= psi <= 80:
        return "left_helix"
    return "other"


def compute_ramachandran_fractions(
    phi: np.ndarray, psi: np.ndarray,
) -> Dict[str, float]:
    """Compute fraction of residues in each Ramachandran region."""
    regions = [classify_ramachandran(p, s) for p, s in zip(phi, psi)]
    L = len(regions)
    if L == 0:
        return {"alpha": 0.0, "beta": 0.0, "left_helix": 0.0, "other": 0.0}
    return {
        "alpha": round(regions.count("alpha") / L, 4),
        "beta": round(regions.count("beta") / L, 4),
        "left_helix": round(regions.count("left_helix") / L, 4),
        "other": round(regions.count("other") / L, 4),
    }


def compute_sasa(atom_array, sequence: str) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-residue SASA and relative solvent accessibility (RSA).

    Args:
        atom_array: biotite AtomArray.
        sequence: 1-letter amino acid sequence.

    Returns:
        Tuple of (sasa_per_residue, rsa_per_residue) arrays.
    """
    import biotite.structure as struc

    atom_sasa = struc.sasa(atom_array)

    # Aggregate to per-residue
    res_ids = struc.get_residues(atom_array)[0]
    residue_sasa = np.zeros(len(res_ids))
    for i, rid in enumerate(res_ids):
        mask = atom_array.res_id == rid
        residue_sasa[i] = atom_sasa[mask].sum()

    # Compute RSA
    rsa = np.zeros(len(res_ids))
    for i, rid in enumerate(res_ids):
        idx = i if i < len(sequence) else len(sequence) - 1
        aa_3 = _AA_1TO3.get(sequence[idx], "ALA")
        max_asa = _MAX_ASA.get(aa_3, 200.0)
        rsa[i] = min(residue_sasa[i] / max_asa, 1.0) if max_asa > 0 else 0.0

    return residue_sasa, rsa


def compute_contact_map(
    atom_array, threshold: float = 8.0,
) -> Tuple[List[Tuple[int, int]], int]:
    """Compute Cβ-Cβ contact map (Cα for glycine).

    Args:
        atom_array: biotite AtomArray.
        threshold: Distance cutoff in Angstroms.

    Returns:
        Tuple of (contact_list, num_contacts).
        contact_list: List of (res_i, res_j) pairs (0-indexed, i < j, |i-j| > 5).
    """
    import biotite.structure as struc

    res_ids = struc.get_residues(atom_array)[0]
    L = len(res_ids)

    # Get CB coordinates (CA for glycine)
    cb_coords = np.full((L, 3), np.nan)
    for i, rid in enumerate(res_ids):
        res_mask = atom_array.res_id == rid
        res_atoms = atom_array[res_mask]

        # Try CB first, fall back to CA
        cb_mask = res_atoms.atom_name == "CB"
        if cb_mask.any():
            cb_coords[i] = res_atoms.coord[cb_mask][0]
        else:
            ca_mask = res_atoms.atom_name == "CA"
            if ca_mask.any():
                cb_coords[i] = res_atoms.coord[ca_mask][0]

    # Compute pairwise distances and find contacts
    contacts = []
    for i in range(L):
        if np.isnan(cb_coords[i]).any():
            continue
        for j in range(i + 6, L):  # Skip sequence neighbors (|i-j| > 5)
            if np.isnan(cb_coords[j]).any():
                continue
            dist = np.linalg.norm(cb_coords[i] - cb_coords[j])
            if dist < threshold:
                contacts.append((i, j))

    return contacts, len(contacts)


def extract_properties(
    esmfold_output: Dict[str, Any],
    contact_threshold: float = 8.0,
) -> Dict[str, Any]:
    """Extract all structural properties from ESMFold full output.

    This is the main entry point. Takes output from
    ESMFoldPredictor.predict_full() and returns a comprehensive
    dictionary of structural properties.

    Args:
        esmfold_output: Dictionary from predict_full().
        contact_threshold: Cβ-Cβ distance threshold for contacts.

    Returns:
        Dictionary with all structural properties, or dict with
        "error" key if extraction failed.
    """
    if "error" in esmfold_output:
        return {"error": esmfold_output["error"]}

    sequence = esmfold_output["sequence"]
    positions = esmfold_output["positions"]
    atom37_mask = esmfold_output["atom37_mask"]

    # Convert to biotite
    atom_array = atom37_to_biotite(positions, atom37_mask, sequence)
    if atom_array is None:
        return {"error": "Failed to build AtomArray (no valid atoms)"}

    result = {
        "sequence": sequence,
        "sequence_length": len(sequence),
        "plddt": esmfold_output["plddt"],
        "ptm": esmfold_output["ptm"],
        "plddt_per_residue": esmfold_output["plddt_per_residue"].tolist()
        if isinstance(esmfold_output["plddt_per_residue"], torch.Tensor)
        else esmfold_output["plddt_per_residue"],
        "truncated": esmfold_output.get("truncated", False),
    }

    # Secondary structure
    try:
        ss3_string, ss_fractions = compute_secondary_structure(atom_array)
        result["ss3_string"] = ss3_string
        result.update(ss_fractions)
    except Exception as e:
        logger.warning(f"SS computation failed: {e}")
        result["ss3_string"] = "C" * len(sequence)
        result["helix_fraction"] = 0.0
        result["sheet_fraction"] = 0.0
        result["coil_fraction"] = 1.0

    # Backbone dihedrals
    try:
        phi, psi = compute_dihedrals(atom_array)
        result["phi"] = np.nan_to_num(phi, nan=0.0).tolist()
        result["psi"] = np.nan_to_num(psi, nan=0.0).tolist()
        result["ramachandran"] = compute_ramachandran_fractions(phi, psi)
    except Exception as e:
        logger.warning(f"Dihedral computation failed: {e}")
        result["phi"] = [0.0] * len(sequence)
        result["psi"] = [0.0] * len(sequence)
        result["ramachandran"] = {"alpha": 0.0, "beta": 0.0, "left_helix": 0.0, "other": 1.0}

    # SASA / RSA
    try:
        sasa, rsa = compute_sasa(atom_array, sequence)
        result["sasa_per_residue"] = sasa.tolist()
        result["rsa_per_residue"] = rsa.tolist()
        result["mean_rsa"] = round(float(rsa.mean()), 4)
        # Buried vs exposed fractions
        result["buried_fraction"] = round(float((rsa < 0.25).mean()), 4)
        result["exposed_fraction"] = round(float((rsa >= 0.25).mean()), 4)
    except Exception as e:
        logger.warning(f"SASA computation failed: {e}")
        result["mean_rsa"] = 0.5
        result["buried_fraction"] = 0.5
        result["exposed_fraction"] = 0.5

    # Contact map
    try:
        contacts, num_contacts = compute_contact_map(atom_array, contact_threshold)
        result["contacts"] = contacts
        result["num_contacts"] = num_contacts
        L = len(sequence)
        max_possible = max(1, (L - 5) * (L - 6) // 2)
        result["contact_density"] = round(num_contacts / max_possible, 4)
    except Exception as e:
        logger.warning(f"Contact map computation failed: {e}")
        result["contacts"] = []
        result["num_contacts"] = 0
        result["contact_density"] = 0.0

    return result


def properties_to_summary(props: Dict[str, Any]) -> str:
    """Format structural properties as a human-readable summary string.

    Used as ground truth output in instruction-format training data.
    """
    if "error" in props:
        return f"Error: {props['error']}"

    lines = []

    # SS composition
    h = props.get("helix_fraction", 0)
    e = props.get("sheet_fraction", 0)
    c = props.get("coil_fraction", 0)
    lines.append(
        f"Secondary structure: {h*100:.1f}% helix, {e*100:.1f}% sheet, {c*100:.1f}% coil."
    )

    # SS string
    ss3 = props.get("ss3_string", "")
    if ss3:
        lines.append(f"SS3: {ss3}")

    # Ramachandran
    rama = props.get("ramachandran", {})
    if rama:
        lines.append(
            f"Ramachandran: {rama.get('alpha', 0)*100:.1f}% alpha, "
            f"{rama.get('beta', 0)*100:.1f}% beta, "
            f"{rama.get('left_helix', 0)*100:.1f}% left-helix, "
            f"{rama.get('other', 0)*100:.1f}% other."
        )

    # RSA
    mean_rsa = props.get("mean_rsa", 0)
    buried = props.get("buried_fraction", 0)
    lines.append(
        f"Mean RSA: {mean_rsa:.2f}. Buried: {buried*100:.1f}%."
    )

    # Contacts
    nc = props.get("num_contacts", 0)
    cd = props.get("contact_density", 0)
    lines.append(f"Long-range contacts: {nc} (density {cd:.4f}).")

    # pLDDT
    plddt = props.get("plddt", 0)
    ptm = props.get("ptm", 0)
    lines.append(f"pLDDT: {plddt:.1f}. pTM: {ptm:.3f}.")

    return " ".join(lines)
