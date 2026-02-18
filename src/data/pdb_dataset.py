"""
PDB Protein Structure Dataset

Dataset loader for the IPD PDB training set (pdb_2021aug02_sample).
Contains protein structures with sequences, 3D coordinates, and metadata.

Data format per chain (.pt file):
    - seq: amino acid sequence (str)
    - xyz: atomic coordinates [L, 14, 3] (N, CA, C, O, CB, ...)
    - mask: valid atom mask [L, 14]
    - bfac: temperature factors [L, 14]
    - occ: occupancy [L, 14]

Download:
    wget https://files.ipd.uw.edu/pub/training_sets/pdb_2021aug02_sample.tar.gz
    tar xvf pdb_2021aug02_sample.tar.gz
"""

import os
import csv
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from torch.utils.data import Dataset, DataLoader


class PDBProteinDataset(Dataset):
    """
    PyTorch Dataset for PDB protein structures.

    Each sample contains:
        - sequence: amino acid sequence string
        - coords: atomic coordinates [L, 14, 3]
        - mask: valid atom mask [L, 14]
        - chain_id: PDB chain identifier (e.g., "5l3g_A")
        - metadata: dict with resolution, deposition date, cluster info
    """

    def __init__(
        self,
        data_dir: str,
        csv_path: Optional[str] = None,
        max_length: Optional[int] = None,
        min_length: int = 10,
        max_resolution: Optional[float] = None,
        return_coords: bool = True,
        transform: Optional[callable] = None,
    ):
        """
        Args:
            data_dir: Path to pdb_2021aug02_sample directory
            csv_path: Path to list.csv (default: data_dir/list.csv)
            max_length: Maximum sequence length (None = no limit)
            min_length: Minimum sequence length
            max_resolution: Maximum resolution in Angstroms (None = no limit)
            return_coords: Whether to load 3D coordinates
            transform: Optional transform to apply to samples
        """
        self.data_dir = Path(data_dir)
        self.pdb_dir = self.data_dir / "pdb"
        self.csv_path = Path(csv_path) if csv_path else self.data_dir / "list.csv"
        self.max_length = max_length
        self.min_length = min_length
        self.max_resolution = max_resolution
        self.return_coords = return_coords
        self.transform = transform

        # Load and filter entries from CSV
        self.entries = self._load_entries()

    def _load_entries(self) -> List[Dict[str, Any]]:
        """Load entries from list.csv and apply filters."""
        entries = []

        with open(self.csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                chain_id = row['CHAINID']
                sequence = row['SEQUENCE']
                seq_len = len(sequence)

                # Apply length filters
                if seq_len < self.min_length:
                    continue
                if self.max_length and seq_len > self.max_length:
                    continue

                # Apply resolution filter
                resolution = float(row['RESOLUTION']) if row['RESOLUTION'] else None
                if self.max_resolution and resolution and resolution > self.max_resolution:
                    continue

                # Get .pt file path
                pdb_id = chain_id.split('_')[0].lower()
                subdir = pdb_id[1:3]  # e.g., "5l3g" -> "l3"
                pt_path = self.pdb_dir / subdir / f"{chain_id}.pt"

                if not pt_path.exists():
                    continue

                entries.append({
                    'chain_id': chain_id,
                    'pt_path': str(pt_path),
                    'sequence': sequence,
                    'deposition': row['DEPOSITION'],
                    'resolution': resolution,
                    'hash': row['HASH'],
                    'cluster': row['CLUSTER'],
                })

        return entries

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        entry = self.entries[idx]

        sample = {
            'chain_id': entry['chain_id'],
            'sequence': entry['sequence'],
            'length': len(entry['sequence']),
            'resolution': entry['resolution'],
            'deposition': entry['deposition'],
            'cluster': entry['cluster'],
        }

        if self.return_coords:
            # Load structure data from .pt file
            data = torch.load(entry['pt_path'], weights_only=False)
            sample['coords'] = data['xyz']      # [L, 14, 3]
            sample['mask'] = data['mask']       # [L, 14]
            sample['bfac'] = data['bfac']       # [L, 14]
            sample['occ'] = data['occ']         # [L, 14]

        if self.transform:
            sample = self.transform(sample)

        return sample


def collate_proteins(batch: List[Dict]) -> Dict[str, Any]:
    """
    Collate function for variable-length protein sequences.

    Pads sequences and coordinates to the maximum length in the batch.
    """
    max_len = max(item['length'] for item in batch)
    batch_size = len(batch)

    # Collect non-tensor fields
    chain_ids = [item['chain_id'] for item in batch]
    sequences = [item['sequence'] for item in batch]
    lengths = torch.tensor([item['length'] for item in batch])
    resolutions = [item['resolution'] for item in batch]

    result = {
        'chain_id': chain_ids,
        'sequence': sequences,
        'length': lengths,
        'resolution': resolutions,
    }

    # Pad coordinate tensors if present
    if 'coords' in batch[0]:
        coords = torch.zeros(batch_size, max_len, 14, 3)
        mask = torch.zeros(batch_size, max_len, 14)

        for i, item in enumerate(batch):
            L = item['length']
            coords[i, :L] = item['coords']
            mask[i, :L] = item['mask']

        result['coords'] = coords
        result['mask'] = mask

    return result


def get_pdb_dataloader(
    data_dir: str,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    max_length: Optional[int] = 512,
    min_length: int = 10,
    max_resolution: Optional[float] = 3.0,
    return_coords: bool = True,
    **kwargs
) -> DataLoader:
    """
    Create a DataLoader for the PDB protein dataset.

    Args:
        data_dir: Path to pdb_2021aug02_sample directory
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of data loading workers
        max_length: Maximum sequence length
        min_length: Minimum sequence length
        max_resolution: Maximum resolution filter
        return_coords: Whether to load 3D coordinates
        **kwargs: Additional arguments for DataLoader

    Returns:
        DataLoader instance
    """
    dataset = PDBProteinDataset(
        data_dir=data_dir,
        max_length=max_length,
        min_length=min_length,
        max_resolution=max_resolution,
        return_coords=return_coords,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_proteins,
        pin_memory=True,
        **kwargs
    )


def download_pdb_sample(target_dir: str = ".") -> str:
    """
    Download and extract the PDB sample dataset.

    Args:
        target_dir: Directory to download to

    Returns:
        Path to extracted dataset directory
    """
    import subprocess

    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    url = "https://files.ipd.uw.edu/pub/training_sets/pdb_2021aug02_sample.tar.gz"
    tar_file = target_dir / "pdb_2021aug02_sample.tar.gz"
    extract_dir = target_dir / "pdb_2021aug02_sample"

    if extract_dir.exists():
        print(f"Dataset already exists at {extract_dir}")
        return str(extract_dir)

    print(f"Downloading {url}...")
    subprocess.run(["wget", "-q", url, "-O", str(tar_file)], check=True)

    print("Extracting...")
    subprocess.run(["tar", "xzf", str(tar_file), "-C", str(target_dir)], check=True)

    print("Cleaning up...")
    tar_file.unlink()

    print(f"Dataset ready at {extract_dir}")
    return str(extract_dir)


if __name__ == "__main__":
    # Example usage and test
    import argparse

    parser = argparse.ArgumentParser(description="Test PDB Dataset")
    parser.add_argument("--data_dir", type=str, default="./pdb_2021aug02_sample",
                        help="Path to dataset directory")
    parser.add_argument("--download", action="store_true",
                        help="Download dataset if not present")
    args = parser.parse_args()

    if args.download and not Path(args.data_dir).exists():
        args.data_dir = download_pdb_sample(Path(args.data_dir).parent)

    print("Loading dataset...")
    dataset = PDBProteinDataset(
        data_dir=args.data_dir,
        max_length=512,
        max_resolution=3.0,
    )
    print(f"Dataset size: {len(dataset)} chains")

    # Test single sample
    sample = dataset[0]
    print(f"\nSample 0:")
    print(f"  Chain ID: {sample['chain_id']}")
    print(f"  Sequence length: {sample['length']}")
    print(f"  Sequence: {sample['sequence'][:50]}...")
    print(f"  Resolution: {sample['resolution']} Å")
    if 'coords' in sample:
        print(f"  Coords shape: {sample['coords'].shape}")
        print(f"  Mask shape: {sample['mask'].shape}")

    # Test dataloader
    print("\nTesting DataLoader...")
    dataloader = get_pdb_dataloader(
        data_dir=args.data_dir,
        batch_size=4,
        max_length=256,
        num_workers=0,
    )

    batch = next(iter(dataloader))
    print(f"Batch keys: {list(batch.keys())}")
    print(f"Batch sequences: {len(batch['sequence'])}")
    print(f"Batch lengths: {batch['length']}")
    if 'coords' in batch:
        print(f"Batch coords shape: {batch['coords'].shape}")
