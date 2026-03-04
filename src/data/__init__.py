# Data loading and processing utilities

# IPD PDB dataset (RoseTTAFold/ProteinMPNN format)
# Download utilities
# Assembly
from .assemble_combined import assemble_combined
from .download import (
    download_alphafold_structures,
    download_ipd_pdb_sample,
    download_mol_instructions,
    download_protdescribe,
    download_protein2text_qa,
    download_proteinlm,
    download_proteinlm_bench,
    download_rcsb_structures,
    download_swissprot_sequences,
    download_swissprotclap,
    list_available_datasets,
)
from .ipd_pdb_converter import convert_ipd_pdb, prepare_ipd_pdb

# Mol-Instructions dataset (instruction-following for proteins)
from .mol_instructions import (
    MolInstructionsCollator,
    MolInstructionsConfig,
    MolInstructionsDataset,
    get_mol_instructions_dataloader,
)
from .pdb_dataset import (
    PDBProteinDataset,
    collate_proteins,
    download_pdb_sample,
    get_pdb_dataloader,
)

# Data converters
from .protdescribe_converter import convert_protdescribe, prepare_protdescribe
from .protein2text_qa_converter import convert_protein2text_qa, prepare_protein2text_qa
from .proteinlm_converter import convert_proteinlm, prepare_proteinlm

# RCSB PDB dataset (standard PDB/mmCIF files)
from .rcsb_dataset import (
    RCSBProteinDataset,
    collate_rcsb_proteins,
    get_rcsb_dataloader,
)
from .swissprot_converter import convert_swissprot, prepare_swissprot
from .swissprotclap_converter import convert_swissprotclap, prepare_swissprotclap
from .wikipedia_protein_converter import convert_wikipedia_protein, prepare_wikipedia_protein
