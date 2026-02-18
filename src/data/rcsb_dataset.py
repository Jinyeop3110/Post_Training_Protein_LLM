"""
RCSB PDB Dataset Loader

Dataset loader for standard PDB/mmCIF files from the RCSB Protein Data Bank.
Supports downloading structures directly from RCSB and parsing local files.

Supported formats:
    - .pdb (legacy PDB format)
    - .cif / .mmcif (mmCIF format)
    - .ent (PDB format with .ent extension)
"""

import os
import gzip
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader

try:
    from Bio.PDB import PDBParser, MMCIFParser
    from Bio.PDB.Polypeptide import is_aa, protein_letters_3to1
    HAS_BIOPYTHON = True

    def three_to_one(resname: str) -> str:
        """Convert 3-letter amino acid code to 1-letter code."""
        return protein_letters_3to1.get(resname, 'X')
except ImportError:
    HAS_BIOPYTHON = False
    three_to_one = None


# Standard backbone + CB atoms (same order as IPD format)
ATOM_ORDER = ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CG1', 'CG2', 'CD', 'CD1', 'CD2',
              'CE', 'CE1', 'CE2', 'CE3', 'CZ', 'CZ2', 'CZ3', 'CH2', 'ND1', 'ND2',
              'NE', 'NE1', 'NE2', 'NH1', 'NH2', 'NZ', 'OD1', 'OD2', 'OE1', 'OE2',
              'OG', 'OG1', 'OH', 'SD', 'SG']

# Simplified 14-atom representation (backbone + common heavy atoms)
ATOM_ORDER_14 = ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2', 'OD1', 'OE1', 'SG']


class RCSBProteinDataset(Dataset):
    """
    PyTorch Dataset for standard PDB/mmCIF protein structures.

    Each sample contains:
        - sequence: amino acid sequence string
        - coords: atomic coordinates [L, 14, 3] (backbone + key atoms)
        - mask: valid atom mask [L, 14]
        - pdb_id: PDB identifier
        - chain_id: chain identifier
    """

    def __init__(
        self,
        pdb_ids: Optional[List[str]] = None,
        pdb_dir: Optional[str] = None,
        pdb_files: Optional[List[str]] = None,
        chains: Optional[Dict[str, List[str]]] = None,
        max_length: Optional[int] = None,
        min_length: int = 10,
        download: bool = True,
        file_format: str = "pdb",
        return_coords: bool = True,
        transform: Optional[callable] = None,
    ):
        """
        Args:
            pdb_ids: List of PDB IDs to load (e.g., ["1abc", "2xyz"])
            pdb_dir: Directory containing PDB files or to download to
            pdb_files: List of paths to PDB/mmCIF files (alternative to pdb_ids)
            chains: Dict mapping PDB ID to list of chains (e.g., {"1abc": ["A", "B"]})
                    If None, all chains are loaded
            max_length: Maximum sequence length
            min_length: Minimum sequence length
            download: Whether to download missing structures from RCSB
            file_format: "pdb" or "cif" for downloads
            return_coords: Whether to return 3D coordinates
            transform: Optional transform to apply to samples
        """
        if not HAS_BIOPYTHON:
            raise ImportError("BioPython is required for RCSB dataset. Install with: pip install biopython")

        self.pdb_dir = Path(pdb_dir) if pdb_dir else Path("./pdb_files")
        self.pdb_dir.mkdir(parents=True, exist_ok=True)
        self.max_length = max_length
        self.min_length = min_length
        self.download = download
        self.file_format = file_format
        self.return_coords = return_coords
        self.transform = transform
        self.chains_filter = chains or {}

        self.pdb_parser = PDBParser(QUIET=True)
        self.cif_parser = MMCIFParser(QUIET=True)

        # Build list of (pdb_file, chain_id) entries
        self.entries = []

        if pdb_files:
            for pdb_file in pdb_files:
                self._add_entries_from_file(pdb_file)
        elif pdb_ids:
            for pdb_id in pdb_ids:
                pdb_file = self._get_pdb_file(pdb_id.lower())
                if pdb_file:
                    allowed_chains = self.chains_filter.get(pdb_id.lower())
                    self._add_entries_from_file(pdb_file, allowed_chains)

    def _get_pdb_file(self, pdb_id: str) -> Optional[Path]:
        """Get path to PDB file, downloading if necessary."""
        pdb_id = pdb_id.lower()

        # Check for existing files
        for ext in ['.pdb', '.cif', '.ent', '.pdb.gz', '.cif.gz', '.ent.gz']:
            path = self.pdb_dir / f"{pdb_id}{ext}"
            if path.exists():
                return path

        if not self.download:
            return None

        # Download from RCSB
        return self._download_pdb(pdb_id)

    def _download_pdb(self, pdb_id: str) -> Optional[Path]:
        """Download structure from RCSB PDB."""
        pdb_id = pdb_id.lower()

        if self.file_format == "cif":
            url = f"https://files.rcsb.org/download/{pdb_id.upper()}.cif.gz"
            local_path = self.pdb_dir / f"{pdb_id}.cif.gz"
        else:
            url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb.gz"
            local_path = self.pdb_dir / f"{pdb_id}.pdb.gz"

        try:
            print(f"Downloading {pdb_id} from RCSB...")
            urllib.request.urlretrieve(url, local_path)
            return local_path
        except Exception as e:
            print(f"Failed to download {pdb_id}: {e}")
            return None

    def _add_entries_from_file(
        self,
        pdb_file: Union[str, Path],
        allowed_chains: Optional[List[str]] = None
    ):
        """Parse PDB file and add valid chain entries."""
        pdb_file = Path(pdb_file)

        try:
            structure = self._parse_structure(pdb_file)
        except Exception as e:
            print(f"Failed to parse {pdb_file}: {e}")
            return

        pdb_id = pdb_file.stem.split('.')[0].lower()

        for model in structure:
            for chain in model:
                chain_id = chain.id

                if allowed_chains and chain_id not in allowed_chains:
                    continue

                # Extract sequence and coords
                seq, coords, mask = self._extract_chain_data(chain)

                if len(seq) < self.min_length:
                    continue
                if self.max_length and len(seq) > self.max_length:
                    continue

                self.entries.append({
                    'pdb_file': str(pdb_file),
                    'pdb_id': pdb_id,
                    'chain_id': chain_id,
                    'sequence': seq,
                    'coords': coords,
                    'mask': mask,
                })
            break  # Only first model

    def _parse_structure(self, pdb_file: Path):
        """Parse PDB or mmCIF file."""
        file_str = str(pdb_file)

        # Handle gzipped files
        if file_str.endswith('.gz'):
            import tempfile
            with gzip.open(pdb_file, 'rt') as f:
                content = f.read()

            suffix = '.cif' if '.cif' in file_str else '.pdb'
            with tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False) as tmp:
                tmp.write(content)
                tmp_path = tmp.name

            try:
                if '.cif' in file_str:
                    structure = self.cif_parser.get_structure('protein', tmp_path)
                else:
                    structure = self.pdb_parser.get_structure('protein', tmp_path)
            finally:
                os.unlink(tmp_path)

            return structure

        # Regular files
        if file_str.endswith('.cif') or file_str.endswith('.mmcif'):
            return self.cif_parser.get_structure('protein', file_str)
        else:
            return self.pdb_parser.get_structure('protein', file_str)

    def _extract_chain_data(self, chain) -> Tuple[str, torch.Tensor, torch.Tensor]:
        """Extract sequence and coordinates from a chain."""
        residues = []

        for residue in chain:
            # Skip non-amino acid residues (water, ligands, etc.)
            if not is_aa(residue, standard=True):
                continue
            residues.append(residue)

        n_residues = len(residues)
        seq = ""
        coords = torch.zeros(n_residues, 14, 3)
        mask = torch.zeros(n_residues, 14)

        for i, residue in enumerate(residues):
            # Get one-letter code
            try:
                seq += three_to_one(residue.resname)
            except KeyError:
                seq += 'X'  # Unknown residue

            # Extract atom coordinates
            for j, atom_name in enumerate(ATOM_ORDER_14):
                if atom_name in residue:
                    atom = residue[atom_name]
                    coords[i, j] = torch.tensor(atom.coord)
                    mask[i, j] = 1.0

        return seq, coords, mask

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        entry = self.entries[idx]

        sample = {
            'pdb_id': entry['pdb_id'],
            'chain_id': entry['chain_id'],
            'sequence': entry['sequence'],
            'length': len(entry['sequence']),
        }

        if self.return_coords:
            sample['coords'] = entry['coords']
            sample['mask'] = entry['mask']

        if self.transform:
            sample = self.transform(sample)

        return sample


def collate_rcsb_proteins(batch: List[Dict]) -> Dict[str, Any]:
    """Collate function for variable-length protein sequences."""
    max_len = max(item['length'] for item in batch)
    batch_size = len(batch)

    result = {
        'pdb_id': [item['pdb_id'] for item in batch],
        'chain_id': [item['chain_id'] for item in batch],
        'sequence': [item['sequence'] for item in batch],
        'length': torch.tensor([item['length'] for item in batch]),
    }

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


def get_rcsb_dataloader(
    pdb_ids: List[str],
    pdb_dir: str = "./pdb_files",
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    max_length: Optional[int] = 512,
    download: bool = True,
    **kwargs
) -> DataLoader:
    """
    Create a DataLoader for RCSB PDB structures.

    Args:
        pdb_ids: List of PDB IDs to load
        pdb_dir: Directory for PDB files
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of workers
        max_length: Maximum sequence length
        download: Whether to download missing structures
        **kwargs: Additional DataLoader arguments

    Returns:
        DataLoader instance
    """
    dataset = RCSBProteinDataset(
        pdb_ids=pdb_ids,
        pdb_dir=pdb_dir,
        max_length=max_length,
        download=download,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_rcsb_proteins,
        pin_memory=True,
        **kwargs
    )


def download_rcsb_structures(
    pdb_ids: List[str],
    output_dir: str = "./pdb_files",
    file_format: str = "pdb",
) -> List[str]:
    """
    Download multiple structures from RCSB PDB.

    Args:
        pdb_ids: List of PDB IDs
        output_dir: Output directory
        file_format: "pdb" or "cif"

    Returns:
        List of downloaded file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    downloaded = []

    for pdb_id in pdb_ids:
        pdb_id = pdb_id.lower()

        if file_format == "cif":
            url = f"https://files.rcsb.org/download/{pdb_id.upper()}.cif.gz"
            local_path = output_dir / f"{pdb_id}.cif.gz"
        else:
            url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb.gz"
            local_path = output_dir / f"{pdb_id}.pdb.gz"

        if local_path.exists():
            print(f"{pdb_id} already exists")
            downloaded.append(str(local_path))
            continue

        try:
            print(f"Downloading {pdb_id}...")
            urllib.request.urlretrieve(url, local_path)
            downloaded.append(str(local_path))
        except Exception as e:
            print(f"Failed to download {pdb_id}: {e}")

    return downloaded


if __name__ == "__main__":
    # Example usage
    print("Testing RCSB Dataset...")

    # Test with a few example PDB IDs
    test_ids = ["1crn", "1ubq", "2gb1"]

    dataset = RCSBProteinDataset(
        pdb_ids=test_ids,
        pdb_dir="./pdb_files",
        download=True,
        max_length=512,
    )

    print(f"Dataset size: {len(dataset)} chains")

    if len(dataset) > 0:
        sample = dataset[0]
        print(f"\nSample 0:")
        print(f"  PDB ID: {sample['pdb_id']}")
        print(f"  Chain: {sample['chain_id']}")
        print(f"  Sequence length: {sample['length']}")
        print(f"  Sequence: {sample['sequence'][:50]}...")
        if 'coords' in sample:
            print(f"  Coords shape: {sample['coords'].shape}")

        # Test dataloader
        print("\nTesting DataLoader...")
        dataloader = get_rcsb_dataloader(
            pdb_ids=test_ids,
            batch_size=2,
            num_workers=0,
        )

        batch = next(iter(dataloader))
        print(f"Batch keys: {list(batch.keys())}")
        print(f"Batch lengths: {batch['length']}")
        if 'coords' in batch:
            print(f"Batch coords shape: {batch['coords'].shape}")
