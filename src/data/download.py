"""
Dataset Download Utilities

Centralized download functions for various protein datasets used in training.
"""

import subprocess
import urllib.request
from pathlib import Path
from typing import List


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

    print("Downloading IPD PDB sample (~47MB)...")
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


def download_proteinlm(target_dir: str = "./data") -> str:
    """Download ProteinLMDataset from HuggingFace.

    Contains ~893K instruction pairs across 7 protein tasks.
    Source: https://huggingface.co/datasets/tsynbio/ProteinLMDataset

    Args:
        target_dir: Directory to download to

    Returns:
        Path to dataset directory
    """
    target_dir = Path(target_dir)
    raw_dir = target_dir / "raw" / "proteinlm"

    if raw_dir.exists() and any(raw_dir.iterdir()):
        print(f"Dataset already exists at {raw_dir}")
        return str(raw_dir)

    raw_dir.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import hf_hub_download
        hf_hub_download(
            repo_id="tsynbio/ProteinLMDataset",
            repo_type="dataset",
            filename="swissProt2Text.json",
            local_dir=str(raw_dir),
        )
        print(f"Dataset ready at {raw_dir}")
        return str(raw_dir)
    except ImportError:
        print("Please install huggingface_hub: pip install huggingface_hub")
        return ""
    except Exception as e:
        print(f"Download failed: {e}")
        return ""


def download_swissprotclap(target_dir: str = "./data") -> str:
    """Download SwissProtCLAP from HuggingFace (chao1224/ProteinDT).

    Contains ~441K protein-text pairs for contrastive learning,
    repurposed here for instruction-following SFT.

    Args:
        target_dir: Directory to download to

    Returns:
        Path to dataset directory
    """
    target_dir = Path(target_dir)
    raw_dir = target_dir / "raw" / "swissprotclap"

    if raw_dir.exists() and (raw_dir / "SwissProtCLAP" / "protein_sequence.txt").exists():
        print(f"Dataset already exists at {raw_dir}")
        return str(raw_dir)

    raw_dir.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import hf_hub_download
        for filename in ["SwissProtCLAP/protein_sequence.txt", "SwissProtCLAP/text_sequence.txt"]:
            hf_hub_download(
                repo_id="chao1224/ProteinDT",
                repo_type="dataset",
                filename=filename,
                local_dir=str(raw_dir),
            )
        print(f"Dataset ready at {raw_dir}")
        return str(raw_dir)
    except ImportError:
        print("Please install huggingface_hub: pip install huggingface_hub")
        return ""
    except Exception as e:
        print(f"Download failed: {e}")
        return ""


def download_protdescribe(target_dir: str = "./data") -> str:
    """Download ProtDescribe dataset from HuggingFace.

    Contains ~549K protein entries with function, location, and similarity annotations.
    Source: https://huggingface.co/datasets/katarinayuan/ProtDescribe

    Args:
        target_dir: Directory to download to

    Returns:
        Path to dataset directory
    """
    target_dir = Path(target_dir)
    raw_dir = target_dir / "raw" / "protdescribe"

    if raw_dir.exists() and any(raw_dir.iterdir()):
        print(f"Dataset already exists at {raw_dir}")
        return str(raw_dir)

    raw_dir.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id="katarinayuan/ProtDescribe",
            repo_type="dataset",
            local_dir=str(raw_dir),
        )
        print(f"Dataset ready at {raw_dir}")
        return str(raw_dir)
    except ImportError:
        print("Please install huggingface_hub: pip install huggingface_hub")
        return ""
    except Exception as e:
        print(f"Download failed: {e}")
        return ""


def download_protein2text_qa(target_dir: str = "./data") -> str:
    """Download Protein2Text-QA dataset from HuggingFace.

    Contains ~56.6K protein QA pairs from tumorailab/Protein2Text-QA.

    Args:
        target_dir: Directory to download to

    Returns:
        Path to dataset directory
    """
    target_dir = Path(target_dir)
    raw_dir = target_dir / "raw" / "protein2text_qa"

    if raw_dir.exists() and any(raw_dir.iterdir()):
        print(f"Dataset already exists at {raw_dir}")
        return str(raw_dir)

    raw_dir.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id="tumorailab/Protein2Text-QA",
            repo_type="dataset",
            local_dir=str(raw_dir),
        )
        print(f"Dataset ready at {raw_dir}")
        return str(raw_dir)
    except ImportError:
        print("Please install huggingface_hub: pip install huggingface_hub")
        return ""
    except Exception as e:
        print(f"Download failed: {e}")
        return ""


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
        "proteinlm": {
            "description": "ProteinLMDataset (7 protein annotation tasks)",
            "size": "~893K instruction pairs",
            "format": "HuggingFace dataset (swissProt2Text.json)",
            "download_fn": "download_proteinlm",
        },
        "swissprotclap": {
            "description": "SwissProtCLAP protein-text pairs (from ProteinDT)",
            "size": "~441K pairs",
            "format": "Parallel text files",
            "download_fn": "download_swissprotclap",
        },
        "protdescribe": {
            "description": "ProtDescribe multi-annotation protein dataset",
            "size": "~549K entries",
            "format": "TSV/Parquet",
            "download_fn": "download_protdescribe",
        },
        "protein2text_qa": {
            "description": "Protein2Text-QA protein question-answering",
            "size": "~56.6K QA pairs",
            "format": "JSON conversations",
            "download_fn": "download_protein2text_qa",
        },
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download protein datasets")
    parser.add_argument("--dataset", type=str, choices=[
        "ipd_pdb_sample", "swissprot", "mol_instructions",
        "proteinlm", "swissprotclap", "protdescribe", "protein2text_qa",
        "list",
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
    elif args.dataset == "proteinlm":
        download_proteinlm(args.output_dir)
    elif args.dataset == "swissprotclap":
        download_swissprotclap(args.output_dir)
    elif args.dataset == "protdescribe":
        download_protdescribe(args.output_dir)
    elif args.dataset == "protein2text_qa":
        download_protein2text_qa(args.output_dir)
