# Protein Datasets and Evaluation Benchmarks
## Comprehensive Research Summary for Post-Training Protein LLMs

*Research Date: February 2026*

---

## Table of Contents
1. [Protein-Text Datasets](#1-protein-text-datasets)
2. [Instruction-Following Datasets for Proteins](#2-instruction-following-datasets-for-proteins)
3. [Evaluation Benchmarks](#3-evaluation-benchmarks)
4. [Evaluation Metrics](#4-evaluation-metrics)
5. [Dataset Recommendations](#5-dataset-recommendations)
6. [Evaluation Plan](#6-evaluation-plan)
7. [Sources](#7-sources)

---

## 1. Protein-Text Datasets

### 1.1 UniProt Descriptions and Function Annotations

**UniProtKB/Swiss-Prot** is the world's most widely used protein information resource:

| Dataset | Size | Description |
|---------|------|-------------|
| **UniProtKB/Swiss-Prot** | ~570,000+ proteins | Manually curated, high-quality protein sequences with function annotations |
| **UniProtKB/TrEMBL** | ~250+ million proteins | Automatically annotated sequences |
| **UniRef50** | ~30 million sequences | Clustered sequences used for pre-training (e.g., ProtNLM) |

**Key Features:**
- Contains comprehensive function descriptions, domain structures, subcellular localization
- Post-translational modifications and functionally characterized variants
- Raw embeddings (per-protein and per-residue) available for ML training
- **ProtNLM** (Google Research collaboration): Annotated 28+ million previously "uncharacterized" proteins

**Availability:** https://www.uniprot.org/

### 1.2 Swiss-Prot Curated Data

Swiss-Prot provides the gold standard for protein annotation:

- **EnzChemRED Dataset**: Training/benchmarking dataset for NLP models extracting enzyme-substrate relationships
- **ProtNote Dataset**: Constructed from Swiss-Prot for multimodal protein-function annotation
  - Removes duplicated sequences and sequences >10,000 amino acids
  - Includes GO and EC annotations for 2019 and 2024 releases
  - Available on Zenodo

### 1.3 PDB Structure Descriptions

| Dataset | Size | Description |
|---------|------|-------------|
| **PDB (Protein Data Bank)** | 214,000+ structures | Experimentally determined 3D protein structures |
| **ProteinNet** | CASP 7-12 series | Standardized dataset for ML with sequences, structures, MSAs, PSSMs |
| **PPB-Affinity** | Large-scale | Protein-protein binding affinity with PDB crystal structures |
| **AlphaFold DB** | 200+ million structures | Predicted structures for all UniProt sequences |

**ProteinNet Features:**
- Standardized training/validation/test splits
- Multiple sequence alignments (MSAs)
- Position-specific scoring matrices (PSSMs)
- Available via TensorFlow Datasets

### 1.4 Protein-Protein Interaction Databases

| Database | Statistics | Description |
|----------|------------|-------------|
| **STRING 12.5** (2025) | Comprehensive network | Physical and functional protein associations with directionality |
| **BioGRID 5.0** | 2.9M+ interactions, 88K publications | Protein, chemical, and genetic interactions |
| **IntAct/IMEx** | Curated interactions | Primary repository for experimental data |
| **HIPPIE** | ~70% gold-standard coverage | Human integrated protein-protein interaction reference |

**STRING 12.5 New Features:**
- New "regulatory network" with directionality of interactions
- Downloadable network embeddings for ML applications
- Cross-species transfer capabilities

---

## 2. Instruction-Following Datasets for Proteins

### 2.1 Mol-Instructions (ICLR 2024)

The primary large-scale biomolecular instruction dataset:

| Component | Size | Tasks |
|-----------|------|-------|
| **Protein-oriented** | 505,000 instructions | 5 task categories (structure, function, activity prediction, protein design) |
| **Molecule-oriented** | 148,400 instructions | 6 tasks (chemical reactions, molecular design) |
| **Biomolecular text** | 53,000 instructions | 6 NLP tasks (information extraction, Q&A) |

**Availability:**
- Dataset: `zjunlp/Mol-Instructions` on HuggingFace
- Models: `zjunlp/llama-molinst-protein-7b`

### 2.2 ProteinChat Dataset

| Dataset | Size | Source |
|---------|------|--------|
| **ProteinChat Training** | 1,500,000+ triplets | Swiss-Prot |
| **Format** | (protein, prompt, answer) | Covering diverse functions |

**Model Architecture:**
- Protein encoder: xTrimoPGLM
- LLM: Vicuna-13B
- Training: Task-specific prompts for flexible prediction

### 2.3 InstructProtein

- Constructs instruction datasets from **knowledge graphs**
- Addresses annotation imbalance in protein-text datasets
- Enables bidirectional tasks:
  - Function prediction from sequences
  - Sequence design from natural language

### 2.4 ProteinGPT Dataset

| Dataset | Size | Description |
|---------|------|-------------|
| **ProteinGPT Training** | 132,092 proteins | Annotations optimized using GPT-4o |

### 2.5 Generating Instruction Data for Proteins

**Recommended Approaches:**

1. **Template-based Generation from Swiss-Prot:**
   - Extract function annotations → Create QA pairs
   - Use GO terms → Generate "What is the function of [protein]?" instructions
   - Leverage subcellular localization → "Where is [protein] located?"

2. **Knowledge Graph-based (InstructProtein approach):**
   - Build relationships between proteins and GO terms
   - Generate diverse instruction formats from graph traversal

3. **LLM-assisted Generation:**
   - Use GPT-4 to rephrase and expand existing annotations
   - Generate diverse question formats from structured data

4. **Cross-modal Pairing:**
   - Pair sequence/structure with text descriptions
   - Create instruction-response pairs for prediction tasks

---

## 3. Evaluation Benchmarks

### 3.1 Protein Function Prediction

| Benchmark | Tasks | Key Features |
|-----------|-------|--------------|
| **CAFA5** (2023) | GO term prediction | Global community challenge, blind evaluation |
| **BeProf** | 8 test cases, 17 methods | Novel metrics considering depth and IC |
| **PAD (Protein Annotation Dataset)** | EC numbers, GO terms | 4 categories of functional annotations |
| **PROBE** | 4 core tasks | Semantic similarity, function prediction, drug target classification, binding affinity |

**CAFA Key Information:**
- Uses GO terms across MFO, BPO, CCO (41,000+ terms)
- Multi-label classification problem
- Unbalanced dataset with limited samples per label

### 3.2 Protein-Protein Interaction Prediction

| Benchmark/Dataset | Size | Description |
|-------------------|------|-------------|
| **BioSNAP** | 13,830 positive / 13,634 negative | DrugBank-derived, balanced |
| **Human PPI** | 2,633 positive / 3,364 negative | High-accuracy negative interactions |
| **STRING physical** | Comprehensive | Experimental validation |

### 3.3 Drug-Target Interaction (DTI)

| Dataset | Size (DTI pairs/drugs/proteins) | Metric Type |
|---------|----------------------------------|-------------|
| **Davis** | 25,772 / 68 / 379 | Kd values |
| **KIBA** | Large-scale | IC50, Ki, Kd |
| **BindingDB (Kd)** | 52,284 / 10,665 / 1,413 | Binding affinity |
| **BindingDB (IC50)** | 991,486 / 549,205 / 5,078 | IC50 values |
| **DUD-E** | 102 targets / 22,886 actives | With 50 decoys per active |

**Evaluation Splits:**
- Random split
- Cold Drug split (unseen drugs)
- Cold Protein split (unseen proteins)

### 3.4 Protein Property Prediction

#### Stability Prediction

| Dataset | Size | Description |
|---------|------|-------------|
| **Megascale** | Large-scale | ΔΔG° from protease sensitivity |
| **S2648** | 2,648 mutations | Standard benchmark |
| **S669** | 669 mutations | Balanced blind test set |
| **ProThermDB** | 32,000 proteins | Comprehensive thermodynamic data |
| **FireProtDB** | Quality-controlled | Single-point mutant thermostability |

#### PEER Benchmark (Comprehensive)

17 biologically relevant tasks covering:

| Category | Tasks |
|----------|-------|
| **Function** | Fluorescence, stability, β-lactamase activity |
| **Structure** | Fold classification, secondary structure |
| **Localization** | Subcellular localization, binary localization |
| **Interaction** | PPI affinity, PDBbind, BindingDB |
| **Properties** | Solubility |

### 3.5 Biomedical QA Benchmarks

| Dataset | Size | Type |
|---------|------|------|
| **BioASQ 2020** | 3,243 train + 500 test | Factoid, list, yes/no, summary |
| **PubMedQA** | 1K expert + 61.2K unlabeled + 211.3K artificial | Yes/no/maybe |
| **COVID-QA** | 2,000 QA pairs | Expert-annotated |

---

## 4. Evaluation Metrics

### 4.1 Classification Tasks (Function Prediction, PPI, DTI)

| Metric | Description | Use Case |
|--------|-------------|----------|
| **F1-score** | Harmonic mean of precision/recall | Multi-label classification |
| **Accuracy** | Correct predictions / total | Balanced datasets |
| **MCC (Matthews Correlation Coefficient)** | Balanced measure | Imbalanced datasets |
| **AUPR** | Area under precision-recall curve | Imbalanced multi-label |
| **AUC-ROC** | Area under ROC curve | Binary classification |

### 4.2 Regression Tasks (Binding Affinity, Stability)

| Metric | Description |
|--------|-------------|
| **Pearson Correlation** | Linear correlation coefficient |
| **Spearman Correlation** | Rank correlation |
| **MSE / RMSE** | Mean squared error |
| **MAE** | Mean absolute error |

### 4.3 Text Generation / QA Tasks

| Metric | Description |
|--------|-------------|
| **BLEU** | N-gram overlap with reference |
| **ROUGE** | Recall-oriented understudy |
| **METEOR** | Alignment-based evaluation |
| **BERTScore** | Semantic similarity using embeddings |
| **Exact Match** | For factoid questions |

### 4.4 Structure Prediction (CASP)

| Metric | Description |
|--------|-------------|
| **GDT-TS** | Global Distance Test |
| **TM-score** | Template Modeling score |
| **RMSD** | Root mean square deviation |
| **lDDT** | Local Distance Difference Test |

---

## 5. Dataset Recommendations

### 5.1 For SFT Training Data Construction

**Priority 1: Core Datasets**

| Dataset | Size | Purpose |
|---------|------|---------|
| **Mol-Instructions (protein)** | 505K | Ready-to-use instruction format |
| **ProteinChat triplets** | 1.5M | Swiss-Prot derived QA |
| **Swiss-Prot annotations** | 570K+ proteins | Base for custom instructions |

**Priority 2: Supplementary Datasets**

| Dataset | Purpose |
|---------|---------|
| **GO annotations** | Function prediction instructions |
| **PDB descriptions** | Structure-function relationships |
| **STRING interactions** | PPI prediction training |

### 5.2 SFT Data Construction Strategy

```
1. Extract Swiss-Prot entries with:
   - Function descriptions
   - GO term annotations (MF, BP, CC)
   - Subcellular localization
   - Catalytic activity

2. Generate instruction templates:
   - "Describe the function of this protein: [sequence]"
   - "What cellular processes does this protein participate in?"
   - "Where is this protein located in the cell?"
   - "What molecular functions does this protein perform?"

3. Augment with:
   - Mol-Instructions protein component
   - InstructProtein knowledge graph data
   - Custom templates from domain experts
```

### 5.3 Recommended Dataset Sizes

| Training Stage | Recommended Size |
|----------------|------------------|
| **Initial SFT** | 500K - 1M instruction pairs |
| **Task-specific fine-tuning** | 50K - 100K per task |
| **Evaluation held-out** | 10-20% of training |

---

## 6. Evaluation Plan

### 6.1 Core Evaluation Tasks (Must-Have)

| Task | Benchmark | Metric |
|------|-----------|--------|
| **Function Prediction (GO)** | CAFA / PROBE | Fmax, AUPR |
| **Subcellular Localization** | PEER | Accuracy, F1 |
| **Stability Prediction** | S669, Megascale | Pearson, Spearman |
| **PPI Prediction** | BioSNAP, Human PPI | AUC, AUPR, MCC |

### 6.2 Extended Evaluation Tasks (Recommended)

| Task | Benchmark | Metric |
|------|-----------|--------|
| **Drug-Target Interaction** | Davis, KIBA | MSE, CI (Concordance Index) |
| **Protein QA** | Custom from PubMedQA | BLEU, BERTScore |
| **Secondary Structure** | PEER/ProteinGLUE | Accuracy (Q3, Q8) |
| **Fold Classification** | PEER | Accuracy |

### 6.3 Evaluation Protocol

```
Phase 1: Zero-shot Evaluation
- Test on held-out protein sets
- Measure generation quality (BLEU, ROUGE)
- Assess GO term prediction accuracy

Phase 2: Task-specific Evaluation
- Fine-tune on specific benchmarks
- Compare with state-of-the-art models
- Report on standard test splits

Phase 3: Cross-domain Transfer
- Train on one organism → test on others
- Evaluate cold-start scenarios
- Assess generalization capability
```

### 6.4 Baseline Models for Comparison

| Model | Type | Parameters |
|-------|------|------------|
| **ESM2-650M/3B** | Sequence PLM | 650M - 3B |
| **ESM3** | Multimodal | 98B |
| **ProtT5-XL** | Sequence PLM | 3B |
| **ProstT5** | Structure-aware | - |
| **SaProt** | Structure-aware | - |

---

## 7. Sources

### UniProt and Swiss-Prot
- [UniProt: the Universal Protein Knowledgebase in 2025](https://academic.oup.com/nar/article/53/D1/D609/7902999)
- [UniProtKB/Swiss-Prot - SIB Swiss Institute of Bioinformatics](https://www.expasy.org/resources/uniprotkb-swiss-prot)
- [ProtNote GitHub - Microsoft](https://github.com/microsoft/protnote)

### Mol-Instructions
- [Mol-Instructions GitHub](https://github.com/zjunlp/Mol-Instructions)
- [Mol-Instructions on HuggingFace](https://huggingface.co/datasets/zjunlp/Mol-Instructions)
- [ICLR 2024 Paper](https://openreview.net/forum?id=Tlsdsb6l9n)

### Benchmarks
- [PROBE Benchmark](https://www.biorxiv.org/content/10.1101/2025.04.10.648084v1.full)
- [BeProf Benchmark](https://academic.oup.com/bib/article/25/2/bbae050/7611938)
- [ProteinGLUE](https://www.nature.com/articles/s41598-022-19608-4)
- [PEER Benchmark](https://papers.neurips.cc/paper_files/paper/2022/file/e467582d42d9c13fa9603df16f31de6d-Paper-Datasets_and_Benchmarks.pdf)

### Protein-Protein Interaction
- [STRING Database 2025](https://academic.oup.com/nar/article/53/D1/D730/7903368)
- [BioGRID](https://thebiogrid.org/)

### Drug-Target Interaction
- [TDC DTI Tasks](https://tdcommons.ai/multi_pred_tasks/dti/)
- [GTB-DTI Benchmark](https://arxiv.org/html/2407.04055v1)

### Protein Language Models
- [ESM GitHub](https://github.com/facebookresearch/esm)
- [ProteinGPT](https://arxiv.org/html/2408.11363v1)
- [Prot2Chat](https://academic.oup.com/bioinformatics/article/41/8/btaf396/8215464)

### Stability and Properties
- [TemStaPro](https://academic.oup.com/bioinformatics/article/40/4/btae157/7632735)
- [Megascale Dataset Study](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012248)

### Biomedical QA
- [BioASQ-QA](https://www.nature.com/articles/s41597-023-02068-4)
- [PubMedQA](https://pubmedqa.github.io/)

### Gene Ontology
- [Gene Ontology Resource](https://geneontology.org/)

---

## Summary

For building a post-training pipeline for protein LLMs, the recommended approach is:

1. **Start with Mol-Instructions** (505K protein instructions) as the base SFT dataset
2. **Augment with Swiss-Prot** derived QA pairs (~1.5M potential pairs)
3. **Evaluate on CAFA/PROBE** for function prediction
4. **Include PEER benchmark** for comprehensive property evaluation
5. **Use STRING/BioGRID** for PPI prediction evaluation
6. **Benchmark against ESM2/ESM3** as baselines

The most critical tasks for evaluating protein understanding are:
- GO term prediction (molecular function, biological process, cellular component)
- Protein-protein interaction prediction
- Stability/thermostability prediction
- Subcellular localization

These tasks cover the fundamental aspects of protein biology that any protein-understanding model should master.
