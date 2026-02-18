"""
Dataset Download Utilities

Centralized download functions for various protein datasets used in training.
"""

import os
import subprocess
import urllib.request
from pathlib import Path
from typing import List, Optional


def download_ipd_pdb_sample(target_dir: str = "./data") -> str:
    """
    Download IPD PDB training set (RoseTTAFold/ProteinMPNN format).

    This is a preprocessed subset of PDB structures with:
    - ~556K chains
    - Pre-computed coordinates in .pt format
    - Sequence clustering at 30% identity

    Args:
        target_dir: Directory to download to

    Returns:
        Path to extracted dataset directory
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    url = "https://files.ipd.uw.edu/pub/training_sets/pdb_2021aug02_sample.tar.gz"
    tar_file = target_dir / "pdb_2021aug02_sample.tar.gz"
    extract_dir = target_dir / "pdb_2021aug02_sample"

    if extract_dir.exists():
        print(f"Dataset already exists at {extract_dir}")
        return str(extract_dir)

    print(f"Downloading IPD PDB sample (~47MB)...")
    print(f"URL: {url}")
    subprocess.run(["wget", "-q", "--show-progress", url, "-O", str(tar_file)], check=True)

    print("Extracting...")
    subprocess.run(["tar", "xzf", str(tar_file), "-C", str(target_dir)], check=True)

    print("Cleaning up...")
    tar_file.unlink()

    print(f"Dataset ready at {extract_dir}")
    return str(extract_dir)


def download_rcsb_structures(
    pdb_ids: List[str],
    output_dir: str = "./data/pdb_files",
    file_format: str = "pdb",
    verbose: bool = True,
) -> List[str]:
    """
    Download structures from RCSB Protein Data Bank.

    Args:
        pdb_ids: List of PDB IDs (e.g., ["1crn", "1ubq"])
        output_dir: Output directory
        file_format: "pdb" or "cif"
        verbose: Print progress

    Returns:
        List of downloaded file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    downloaded = []
    failed = []

    for pdb_id in pdb_ids:
        pdb_id = pdb_id.lower()

        if file_format == "cif":
            url = f"https://files.rcsb.org/download/{pdb_id.upper()}.cif.gz"
            local_path = output_dir / f"{pdb_id}.cif.gz"
        else:
            url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb.gz"
            local_path = output_dir / f"{pdb_id}.pdb.gz"

        if local_path.exists():
            if verbose:
                print(f"  {pdb_id}: already exists")
            downloaded.append(str(local_path))
            continue

        try:
            if verbose:
                print(f"  {pdb_id}: downloading...")
            urllib.request.urlretrieve(url, local_path)
            downloaded.append(str(local_path))
        except Exception as e:
            if verbose:
                print(f"  {pdb_id}: FAILED - {e}")
            failed.append(pdb_id)

    if verbose:
        print(f"\nDownloaded: {len(downloaded)}, Failed: {len(failed)}")

    return downloaded


def download_mol_instructions(target_dir: str = "./data") -> str:
    """
    Download Mol-Instructions dataset (protein subset).

    Contains ~505K instruction-following pairs for protein tasks.
    Source: https://huggingface.co/datasets/zjunlp/Mol-Instructions

    Args:
        target_dir: Directory to download to

    Returns:
        Path to dataset directory
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    dataset_dir = target_dir / "mol_instructions"

    if dataset_dir.exists():
        print(f"Dataset already exists at {dataset_dir}")
        return str(dataset_dir)

    print("Downloading Mol-Instructions (protein subset)...")
    print("This requires huggingface-cli to be installed and logged in.")

    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id="zjunlp/Mol-Instructions",
            repo_type="dataset",
            local_dir=str(dataset_dir),
            allow_patterns=["*protein*"],
        )
        print(f"Dataset ready at {dataset_dir}")
        return str(dataset_dir)
    except ImportError:
        print("Please install huggingface_hub: pip install huggingface_hub")
        return ""
    except Exception as e:
        print(f"Download failed: {e}")
        return ""


def download_swissprot_sequences(target_dir: str = "./data") -> str:
    """
    Download Swiss-Prot reviewed protein sequences.

    Contains ~570K curated protein sequences with annotations.

    Args:
        target_dir: Directory to download to

    Returns:
        Path to FASTA file
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    url = "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz"
    output_file = target_dir / "uniprot_sprot.fasta.gz"

    if output_file.exists():
        print(f"Swiss-Prot already exists at {output_file}")
        return str(output_file)

    print("Downloading Swiss-Prot sequences (~90MB)...")
    subprocess.run(["wget", "-q", "--show-progress", url, "-O", str(output_file)], check=True)

    print(f"Dataset ready at {output_file}")
    return str(output_file)


def download_alphafold_structures(
    uniprot_ids: List[str],
    output_dir: str = "./data/alphafold",
    version: int = 4,
) -> List[str]:
    """
    Download AlphaFold predicted structures.

    Args:
        uniprot_ids: List of UniProt IDs
        output_dir: Output directory
        version: AlphaFold DB version (default: 4)

    Returns:
        List of downloaded file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    downloaded = []

    for uniprot_id in uniprot_ids:
        url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v{version}.pdb"
        local_path = output_dir / f"AF-{uniprot_id}-F1-model_v{version}.pdb"

        if local_path.exists():
            print(f"  {uniprot_id}: already exists")
            downloaded.append(str(local_path))
            continue

        try:
            print(f"  {uniprot_id}: downloading...")
            urllib.request.urlretrieve(url, local_path)
            downloaded.append(str(local_path))
        except Exception as e:
            print(f"  {uniprot_id}: FAILED - {e}")

    return downloaded


def list_available_datasets() -> dict:
    """List all available datasets with descriptions."""
    return {
        "ipd_pdb_sample": {
            "description": "IPD PDB training set (RoseTTAFold/ProteinMPNN format)",
            "size": "~47MB compressed, ~556K chains",
            "format": "PyTorch .pt files",
            "download_fn": "download_ipd_pdb_sample",
        },
        "rcsb_pdb": {
            "description": "RCSB Protein Data Bank structures",
            "size": "Variable (per structure)",
            "format": "PDB/mmCIF files",
            "download_fn": "download_rcsb_structures",
        },
        "mol_instructions": {
            "description": "Mol-Instructions protein instruction dataset",
            "size": "~505K instruction pairs",
            "format": "HuggingFace dataset",
            "download_fn": "download_mol_instructions",
        },
        "swissprot": {
            "description": "Swiss-Prot curated protein sequences",
            "size": "~90MB, ~570K sequences",
            "format": "FASTA",
            "download_fn": "download_swissprot_sequences",
        },
        "alphafold": {
            "description": "AlphaFold predicted structures",
            "size": "Variable (per structure)",
            "format": "PDB files",
            "download_fn": "download_alphafold_structures",
        },
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download protein datasets")
    parser.add_argument("--dataset", type=str, choices=[
        "ipd_pdb_sample", "swissprot", "mol_instructions", "list"
    ], default="list", help="Dataset to download")
    parser.add_argument("--output_dir", type=str, default="./data",
                        help="Output directory")
    parser.add_argument("--pdb_ids", type=str, nargs="+",
                        help="PDB IDs for RCSB download")
    args = parser.parse_args()

    if args.dataset == "list":
        print("Available datasets:\n")
        for name, info in list_available_datasets().items():
            print(f"  {name}:")
            print(f"    {info['description']}")
            print(f"    Size: {info['size']}")
            print(f"    Format: {info['format']}")
            print()
    elif args.dataset == "ipd_pdb_sample":
        download_ipd_pdb_sample(args.output_dir)
    elif args.dataset == "swissprot":
        download_swissprot_sequences(args.output_dir)
    elif args.dataset == "mol_instructions":
        download_mol_instructions(args.output_dir)
