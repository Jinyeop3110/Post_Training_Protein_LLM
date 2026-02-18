# Data loading and processing utilities

# IPD PDB dataset (RoseTTAFold/ProteinMPNN format)
from .pdb_dataset import (
    PDBProteinDataset,
    get_pdb_dataloader,
    collate_proteins,
    download_pdb_sample,
)

# RCSB PDB dataset (standard PDB/mmCIF files)
from .rcsb_dataset import (
    RCSBProteinDataset,
    get_rcsb_dataloader,
    collate_rcsb_proteins,
)

# Mol-Instructions dataset (instruction-following for proteins)
from .mol_instructions import (
    MolInstructionsDataset,
    MolInstructionsCollator,
    MolInstructionsConfig,
    get_mol_instructions_dataloader,
)

# Download utilities
from .download import (
    download_ipd_pdb_sample,
    download_rcsb_structures,
    download_mol_instructions,
    download_swissprot_sequences,
    download_alphafold_structures,
    list_available_datasets,
)
