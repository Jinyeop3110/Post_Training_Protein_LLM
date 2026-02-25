# Combined Dataset: Protein Extraction Audit

**Date**: 2026-02-21
**Data dir**: `data/processed/combined/`
**Config**: `data=combined` (sampling_temperature=0.5)
**Extractor**: `MolInstructionsDataset._extract_protein_sequence()`

## Data Flow

```
JSON file → MolInstructionsDataset.__getitem__()
  ├── item["input"] → _extract_protein_sequence() → protein_sequence
  │     └── ESM-3 encoder (approach=esm3) or <protein>...</protein> tag (approach=text)
  ├── item["instruction"] + item["input"] + item["output"] → formatted_prompt
  │     └── Tokenized by collator → input_ids, attention_mask, labels
  └── Collator combines both → batch dict for Trainer
```

The `protein_sequence` goes to ESM-3 for encoding. The `formatted_prompt` goes to the LLM.
For `approach=text`, the protein is embedded as text tokens in the prompt (no encoder).

### How `_extract_protein_sequence()` works

1. Check if entire input is a raw protein sequence → return it
2. Split by newlines, find first line with >90% amino acid characters and len>=10 → return it
3. **Fallback**: return entire input text as-is (even if it's not a protein!)

This works well for backtick-wrapped sequences (the protein line passes step 2), but fails for:
- **Text descriptions** (design tasks): returns the requirements text
- **X-masked sequences**: `X` not in AA char set → fraction drops below 0.9 → fallback returns the entire input with backticks

## Summary

| File | Records | Input Type | Extraction Quality | Avg Protein Len | Issues |
|------|---------|------------|-------------------|----------------|--------|
| `mol_catalytic_activity.json` | 53,174 | backtick_protein | valid_protein (53,020/53,174) | 417 | 1 issue(s) |
| `mol_domain_motif.json` | 45,100 | backtick_protein | valid_protein (44,938/45,100) | 477 | 1 issue(s) |
| `mol_general_function.json` | 86,572 | backtick_protein | valid_protein (86,244/86,572) | 381 | 1 issue(s) |
| `mol_protein_design.json` | 195,975 | text_description | partial_protein (0/195,975) | N/A | 3 issue(s) |
| `mol_protein_function.json` | 114,183 | backtick_protein | valid_protein (113,614/114,183) | 372 | 1 issue(s) |
| `sp_gene_prediction.json` | 263,061 | backtick_protein | valid_protein (263,028/263,061) | 331 | OK |
| `sp_general_function.json` | 542,287 | backtick_protein | valid_protein (542,162/542,287) | 328 | 1 issue(s) |
| `sp_organism_prediction.json` | 271,498 | backtick_protein | valid_protein (271,430/271,498) | 328 | OK |
| `wp_disease_association.json` | 1,763 | backtick_protein | valid_protein (1,760/1,763) | 307 | OK |
| `wp_protein_function.json` | 3,507 | backtick_protein | valid_protein (3,495/3,507) | 310 | OK |
| `wp_protein_overview.json` | 7,166 | backtick_protein | valid_protein (7,146/7,166) | 308 | 1 issue(s) |
| `wp_protein_structure.json` | 2,930 | backtick_protein | valid_protein (2,920/2,930) | 307 | OK |

**Total**: 1,587,216 records, 1,389,757 valid protein extractions (87.6%), 9 issues

## Data Sources

- **Mol-Instructions** (`mol_*`): 495,004 records
- **Swiss-Prot** (`sp_*`): 1,076,846 records
- **Wikipedia Protein** (`wp_*`): 15,366 records

## Issues Found

### `mol_catalytic_activity.json`
- 10 records extracted non-protein text

### `mol_domain_motif.json`
- 7 records extracted non-protein text

### `mol_general_function.json`
- 76 records extracted non-protein text

### `mol_protein_design.json`
- 25 records extracted non-protein text
- 195,975 records have text descriptions as input (not protein sequences)
- ~99% of records have protein in OUTPUT (design task: input=requirements, output=sequence)

### `mol_protein_function.json`
- 119 records extracted non-protein text

### `sp_general_function.json`
- 1 records extracted non-protein text

### `wp_protein_overview.json`
- 1 records extracted non-protein text

## Per-File Detailed Analysis

### `mol_catalytic_activity.json`

- **Records**: 53,174
- **Source**: `mol_`
- **Input type distribution**: {'backtick_protein': 52479, 'backtick_mixed': 695}
- **Extraction quality**: {'valid_protein': 53020, 'partial_protein': 144, 'not_protein': 10}
- **Protein lengths**: min=10, max=768, mean=417, median=405

**Samples:**

<details><summary>Record 0: backtick_protein → valid_protein</summary>

- **Instruction**: `Please evaluate the following protein sequence and provide an explanation of the enzyme's catalytic activity, including `
- **Input** (backtick_protein): ````
MPGRQLTELLTGLEEVKVQTAMEQKEMMIGGLTADSREVRPGDLFAALPGARVDGRDFIDQAVGRGADVVLAPVGTSLKDYGRPVSLVTSDEPRRTLAQMAARFHGRQPRTIAAVTGTSGKTSVADFLRQIWTLADRKAASLGTLG...`
- **Output preview**: `By examining the input protein sequence, the enzyme catalyzes the subsequent chemical reaction: ATP ...`
- **Extracted protein**: `MPGRQLTELLTGLEEVKVQTAMEQKEMMIGGLTADSREVRPGDLFAALPGARVDGRDFIDQAVGRGADVVLAPVGTSLKDYGRPVSLVTSDEPRRTLAQM...` (len=495)
- **Quality**: **valid_protein**

</details>

<details><summary>Record 1: backtick_protein → valid_protein</summary>

- **Instruction**: `Given the protein sequence below, please analyze and describe the catalytic activity of the corresponding enzyme, specif`
- **Input** (backtick_protein): ````
MKQVVIASRESPLAMWQAEHIRARLQALYPGLEVSILGITTQGDRILDKTLNKIGGKGLFVKELELAMQEGQADLAVHSIKDVPMDLPEGFALAAICEREDPRDAFVSSRYASLSELPAGAVVGTASLRRESQIRARYPHLLVKPL...`
- **Output preview**: `Based on the provided protein sequence, the enzyme appears to facilitate the chemical reaction: H2O ...`
- **Extracted protein**: `MKQVVIASRESPLAMWQAEHIRARLQALYPGLEVSILGITTQGDRILDKTLNKIGGKGLFVKELELAMQEGQADLAVHSIKDVPMDLPEGFALAAICERE...` (len=308)
- **Quality**: **valid_protein**

</details>

<details><summary>Record 2: backtick_protein → valid_protein</summary>

- **Instruction**: `Given the protein sequence below, please analyze and describe the catalytic activity of the corresponding enzyme, specif`
- **Input** (backtick_protein): ````
MKPVHIVSSAQMRWADMQTMQKTPSRTLMERAGYAVAEAVVHNMPDVGRVVVVAGGGNNGGDGYAAAFFLRRRLPVTVVSLVPVERHTEDARHWRDQAVAAGVKVRDACGDPRALLDRWCQRAVIIVDALFGTGLKRPLMGEMALA...`
- **Output preview**: `Upon reviewing the provided protein sequence, the corresponding enzyme's catalytic activity is ident...`
- **Extracted protein**: `MKPVHIVSSAQMRWADMQTMQKTPSRTLMERAGYAVAEAVVHNMPDVGRVVVVAGGGNNGGDGYAAAFFLRRRLPVTVVSLVPVERHTEDARHWRDQAVA...` (len=528)
- **Quality**: **valid_protein**

</details>

<details><summary>Record 3: backtick_protein → valid_protein</summary>

- **Instruction**: `Please evaluate the following protein sequence and provide an explanation of the enzyme's catalytic activity, including `
- **Input** (backtick_protein): ````
MRKKVTKYIFVTGGVMSSLGKGLASACIGALLEARGMRVSLQKLDPYLNIDPGTMNPYQHGEVFVTNDGAETDLDLGHYERFTNASLSQENNVTSGRVYDEVISQERKGEYLGQTVQVIPHITDEIIKRIYSLKQKQDVSIVEIGG...`
- **Output preview**: `An analysis of the protein sequence reveals that the enzyme's catalytic function corresponds to the ...`
- **Extracted protein**: `MRKKVTKYIFVTGGVMSSLGKGLASACIGALLEARGMRVSLQKLDPYLNIDPGTMNPYQHGEVFVTNDGAETDLDLGHYERFTNASLSQENNVTSGRVYD...` (len=534)
- **Quality**: **valid_protein**

</details>

<details><summary>Record 4: backtick_protein → valid_protein</summary>

- **Instruction**: `Examine the provided protein sequence and determine the catalytic activity of the enzyme it represents, focusing on the `
- **Input** (backtick_protein): ````
MRLRNKPWAKDKIAAYPQYVIPDPETKRGRWRELFGHDQPLHVEIGTGKGKFITEMAKLHPDVNFIGIELYPSVLVSALDKLIESGLANVKLLNANAKDLTAFFADGEVSRIYLNFSDPWPKKRHEKRRLTYRDFLALYDRILAED...`
- **Output preview**: `Evaluation of the protein sequence indicates that the associated enzyme exhibits catalytic activity ...`
- **Extracted protein**: `MRLRNKPWAKDKIAAYPQYVIPDPETKRGRWRELFGHDQPLHVEIGTGKGKFITEMAKLHPDVNFIGIELYPSVLVSALDKLIESGLANVKLLNANAKDL...` (len=217)
- **Quality**: **valid_protein**

</details>

### `mol_domain_motif.json`

- **Records**: 45,100
- **Source**: `mol_`
- **Input type distribution**: {'backtick_protein': 43849, 'backtick_mixed': 1251}
- **Extraction quality**: {'valid_protein': 44938, 'partial_protein': 155, 'not_protein': 7}
- **Protein lengths**: min=10, max=768, mean=477, median=466

**Samples:**

<details><summary>Record 0: backtick_protein → valid_protein</summary>

- **Instruction**: `Find and list any domains or motifs that are likely present in this protein sequence: `
- **Input** (backtick_protein): ````
MEFDTIAAISTFPGEAGIGIVRISGDEALEIISKIFRPFRKKDIKSVKSHTIHYGHIVDPETGEVYDEVLVTVMRKPNTYTREDVVEINCHGGIVVSSKILELVLKHGARLAEPGEFTKRAFLNGRIDLSQAEAVIDIITSKTMLA...`
- **Output preview**: `Our bioinformatics tools have processed the sequence you provided sequence. The prediction suggests ...`
- **Extracted protein**: `MEFDTIAAISTFPGEAGIGIVRISGDEALEIISKIFRPFRKKDIKSVKSHTIHYGHIVDPETGEVYDEVLVTVMRKPNTYTREDVVEINCHGGIVVSSKI...` (len=460)
- **Quality**: **valid_protein**

</details>

<details><summary>Record 1: backtick_protein → valid_protein</summary>

- **Instruction**: `Please examine the following protein sequence and predict any domains or motifs you can discern. The sequence is: `
- **Input** (backtick_protein): ````
MAEGDLAIAAEAAALRAKNASLEREVETLKEQLLALRAELGSAKTPEVASQDGAAEQRDGGATASTSAAPTWSSSAPAPPRAAPRHGLTRAQAERYSRHLLLPAFGVAAQERLVRGAVLVVGCGGLGSPAAMYLAAAGVGRLGLVD...`
- **Output preview**: `After analyzing the given sequence, the following protein domains or motifs are predicted: Rhodanese...`
- **Extracted protein**: `MAEGDLAIAAEAAALRAKNASLEREVETLKEQLLALRAELGSAKTPEVASQDGAAEQRDGGATASTSAAPTWSSSAPAPPRAAPRHGLTRAQAERYSRHL...` (len=566)
- **Quality**: **valid_protein**

</details>

<details><summary>Record 2: backtick_protein → valid_protein</summary>

- **Instruction**: `Can you analyze the protein sequence provided below and predict any recognizable domains or motifs it may contain? Here'`
- **Input** (backtick_protein): ````
MKVKGIQRNWQHLWTWWTLTLGLVIICSASNNLWVTVYYGVPVWEDADTPLFCASDAKAYSTEKHNVWATHACVPTDPNPQEIELKNVKENFNMWKNNMVEQMHEDIISLWDESLKPCVRLTPLCVTLNCTNVNTTSCTNNTTTKN...`
- **Output preview**: `The computational analysis of the sequence suggests the presence of the following protein domains or...`
- **Extracted protein**: `MKVKGIQRNWQHLWTWWTLTLGLVIICSASNNLWVTVYYGVPVWEDADTPLFCASDAKAYSTEKHNVWATHACVPTDPNPQEIELKNVKENFNMWKNNMV...` (len=768)
- **Quality**: **valid_protein**

</details>

<details><summary>Record 3: backtick_protein → valid_protein</summary>

- **Instruction**: `Please examine the following protein sequence and predict any domains or motifs you can discern. The sequence is: `
- **Input** (backtick_protein): ````
MKSIEVHTDGSCLGNPGPGGWAALLRYNGREKELAGGEANSTNNRMELMAAIMALETLTEPCQILLHTDSQYVRQGITEWMPGWVRRGWKTSGGDPVKNRELWERLHAATQRHSIEWRWVKGHNGDPDNERVDVLARNQAIAQRGG...`
- **Output preview**: `Our predictive analysis of the given protein sequence reveals possible domains or motifs. These incl...`
- **Extracted protein**: `MKSIEVHTDGSCLGNPGPGGWAALLRYNGREKELAGGEANSTNNRMELMAAIMALETLTEPCQILLHTDSQYVRQGITEWMPGWVRRGWKTSGGDPVKNR...` (len=150)
- **Quality**: **valid_protein**

</details>

<details><summary>Record 4: backtick_protein → valid_protein</summary>

- **Instruction**: `Could you inspect the provided protein sequence and determine any possible motifs or domains? Here is the protein sequen`
- **Input** (backtick_protein): ````
MFRIAKNLVRTFEQSVQDTLALSQDSSNLDAFFQSIPPNLLSAQLESPVDAVSEGVKHTNVNETLSGLRIVWVDEMQFQLQSFFDYIVGFNDDPVPVVSNQHGFSYPDYRRITSIFNEHCGRTLKVNIWSAKGGTFRDEYISIISK...`
- **Output preview**: `Our predictive analysis of the given protein sequence reveals possible domains or motifs. These incl...`
- **Extracted protein**: `MFRIAKNLVRTFEQSVQDTLALSQDSSNLDAFFQSIPPNLLSAQLESPVDAVSEGVKHTNVNETLSGLRIVWVDEMQFQLQSFFDYIVGFNDDPVPVVSN...` (len=372)
- **Quality**: **valid_protein**

</details>

### `mol_general_function.json`

- **Records**: 86,572
- **Source**: `mol_`
- **Input type distribution**: {'backtick_protein': 85126, 'backtick_mixed': 1446}
- **Extraction quality**: {'valid_protein': 86244, 'not_protein': 76, 'partial_protein': 252}
- **Protein lengths**: min=10, max=768, mean=381, median=363

**Samples:**

<details><summary>Record 0: backtick_protein → valid_protein</summary>

- **Instruction**: `Inspect the protein with the subsequent sequence and offer a concise description of its properties: `
- **Input** (backtick_protein): ````
MTDKKFNVAVVGATGAVGETMLSILEQRHFPVGEVYPLASSRSAGKRIEFNNKQLIVEDLETFDFSKVQIGLFSPGASVSAIHAPRAVEAGCVVVDNTSQFRYDDDIPLVVSEVNPHAVADYKNRGIIANPNCSTIQMLVALKPIR...`
- **Output preview**: `A summary of the protein's main attributes with the input amino acid sequence reveals: Catalyzes the...`
- **Extracted protein**: `MTDKKFNVAVVGATGAVGETMLSILEQRHFPVGEVYPLASSRSAGKRIEFNNKQLIVEDLETFDFSKVQIGLFSPGASVSAIHAPRAVEAGCVVVDNTSQ...` (len=341)
- **Quality**: **valid_protein**

</details>

<details><summary>Record 1: backtick_protein → valid_protein</summary>

- **Instruction**: `Please provide a summary of the key features and characteristics of the protein with the following amino acid sequence: `
- **Input** (backtick_protein): ````
MHQQSPVDDVTALNSSALTMSEYPEGESPLQLQDVDSSRVGGHILSPIFNSSSPSLPVESHPVCIQSPYTDLGHDFTTLPFYSPALLGYGTSPLSECSSVRQSLSPTLFWPPHSQVSSLALHQQHTRLQQNHPTGGTWTELTPHDH...`
- **Output preview**: `A short report on the protein with the given amino acid sequence highlights: Binds estrogens with an...`
- **Extracted protein**: `MHQQSPVDDVTALNSSALTMSEYPEGESPLQLQDVDSSRVGGHILSPIFNSSSPSLPVESHPVCIQSPYTDLGHDFTTLPFYSPALLGYGTSPLSECSSV...` (len=568)
- **Quality**: **valid_protein**

</details>

<details><summary>Record 2: backtick_protein → valid_protein</summary>

- **Instruction**: `Could you evaluate the protein with this amino acid sequence and present a summary of its features? The sequence is: `
- **Input** (backtick_protein): ````
MKSVHSSPQNTSHTIMTFYPTMEEFADFNTYVAYMESQGAHQAGLAKVIPPKEWKARQMYDDIEDILIATPLQQVTSGQGGVFTQYHKKKKAMRVGQYRRLANSKKYQTPPHQNFADLEQRYWKSHPGNPPIYGADISGSLFEEST...`
- **Output preview**: `A short report on the protein with the given amino acid sequence highlights: Histone demethylase tha...`
- **Extracted protein**: `MKSVHSSPQNTSHTIMTFYPTMEEFADFNTYVAYMESQGAHQAGLAKVIPPKEWKARQMYDDIEDILIATPLQQVTSGQGGVFTQYHKKKKAMRVGQYRR...` (len=506)
- **Quality**: **valid_protein**

</details>

<details><summary>Record 3: backtick_protein → valid_protein</summary>

- **Instruction**: `Inspect the protein with the subsequent sequence and offer a concise description of its properties: `
- **Input** (backtick_protein): ````
MSIEWLSEKLSEQGIELSNTQKEQFQKYYKLLVEWNKKMNLTSITDEHDVYLKHFYDSIAPSFYYDFNGQLSLCDIGAGAGFPSIPLKIVYPELKVTIVDSLNKRIQFLNHLAAELGLEDVSFVHDRAETYGKGVYRESYDIVTAR...`
- **Output preview**: `A brief overview of the protein with the provided amino acid sequence is as follows: Specifically me...`
- **Extracted protein**: `MSIEWLSEKLSEQGIELSNTQKEQFQKYYKLLVEWNKKMNLTSITDEHDVYLKHFYDSIAPSFYYDFNGQLSLCDIGAGAGFPSIPLKIVYPELKVTIVD...` (len=239)
- **Quality**: **valid_protein**

</details>

<details><summary>Record 4: backtick_protein → valid_protein</summary>

- **Instruction**: `Assess the following protein sequence and provide a brief report on its primary characteristics: `
- **Input** (backtick_protein): ````
MNRANDTSSNLKLISLVVLIVQTTALVLTLRYSQTQKSEGPRYLSSTAVVCAEIIKLITCFFVIYRNNGYRFSGMLNELNREIFASPQTRADSLKVAVPAIMYVIQNNLLFFALKKLDAATYQVTYQLKILTTAIFSVTMLGKSLH...`
- **Output preview**: `A brief overview of the protein with the provided amino acid sequence is as follows: Uridine diphosp...`
- **Extracted protein**: `MNRANDTSSNLKLISLVVLIVQTTALVLTLRYSQTQKSEGPRYLSSTAVVCAEIIKLITCFFVIYRNNGYRFSGMLNELNREIFASPQTRADSLKVAVPA...` (len=339)
- **Quality**: **valid_protein**

</details>

### `mol_protein_design.json`

- **Records**: 195,975
- **Source**: `mol_`
- **Input type distribution**: {'text_description': 195975}
- **Extraction quality**: {'partial_protein': 195950, 'not_protein': 25}

**Samples:**

<details><summary>Record 0: text_description → partial_protein</summary>

- **Instruction**: `Synthesize a protein sequence that has the desired solubility or other physical properties for the intended use.`
- **Input** (text_description): `1. The protein must exhibit the following characteristics: Also acts as a cofactor with GATA4, a key cardiac regulator ..
2. The protein should have m...`
- **Output preview**: `Here's a protein sequence with the above functions:
```
MIGMLESLQHESDLLQHDQIHTGEKPYECNECRKTFSLKQNLVE...`
- **Extracted protein**: `1. The protein must exhibit the following characteristics: Also acts as a cofactor with GATA4, a key...` (len=192)
- **Quality**: **partial_protein**

</details>

<details><summary>Record 1: text_description → partial_protein</summary>

- **Instruction**: `Create a protein sequence that satisfies the following specifications:`
- **Input** (text_description): `1. The protein should have a specific function in intein-mediated protein splicing, cell-cell signaling and should be localized to plasma membrane, en...`
- **Output preview**: `The protein you've requested has been designed with this amino acid sequence:
```
MPISCGSRRGERSYSYSV...`
- **Extracted protein**: `1. The protein should have a specific function in intein-mediated protein splicing, cell-cell signal...` (len=350)
- **Quality**: **partial_protein**

</details>

<details><summary>Record 2: text_description → partial_protein</summary>

- **Instruction**: `Design a protein sequence that is tailored to the desired application.`
- **Input** (text_description): `1. Ideal protein should have a binding affinity for key enzymes in the pyridoxal 5'-phosphate biosynthesis pathway to enhance the overall efficiency o...`
- **Output preview**: `The protein you've requested has been designed with this amino acid sequence:
```
MEKLTIGVLSLQGNFQSH...`
- **Extracted protein**: `1. Ideal protein should have a binding affinity for key enzymes in the pyridoxal 5'-phosphate biosyn...` (len=164)
- **Quality**: **partial_protein**

</details>

<details><summary>Record 3: text_description → partial_protein</summary>

- **Instruction**: `Synthesize a protein sequence that has the desired solubility or other physical properties for the intended use.`
- **Input** (text_description): `1. A protein with phosphopantothenoylcysteine decarboxylase activity, FMN binding, metal ion binding, phosphopantothenate--cysteine ligase activity is...`
- **Output preview**: `The designed protein's amino acid sequence is as follows:
```
MQDFANKKIVLGICGGIAAYKSAYLVRELSRLGAEIRV...`
- **Extracted protein**: `1. A protein with phosphopantothenoylcysteine decarboxylase activity, FMN binding, metal ion binding...` (len=1240)
- **Quality**: **partial_protein**

</details>

<details><summary>Record 4: text_description → partial_protein</summary>

- **Instruction**: `Generate a protein sequence optimized for the following function requirements.`
- **Input** (text_description): `1. The S-adenosyl-L-methionine binding site should be stable and able to withstand changes in environmental conditions.
2. The protein design should t...`
- **Output preview**: `Your protein design is complete, and the amino acid sequence is
```
MDEKSAQGTTHFGFRDVPVGEKKKLVGQVFSS...`
- **Extracted protein**: `1. The S-adenosyl-L-methionine binding site should be stable and able to withstand changes in enviro...` (len=856)
- **Quality**: **partial_protein**

</details>

### `mol_protein_function.json`

- **Records**: 114,183
- **Source**: `mol_`
- **Input type distribution**: {'backtick_protein': 112357, 'backtick_mixed': 1826}
- **Extraction quality**: {'valid_protein': 113614, 'partial_protein': 450, 'not_protein': 119}
- **Protein lengths**: min=10, max=768, mean=372, median=350

**Samples:**

<details><summary>Record 0: backtick_protein → valid_protein</summary>

- **Instruction**: `Using the given protein sequence, predict its functional role and the potential biological pathway it may be a part of: `
- **Input** (backtick_protein): ````
MTTIGTPLRPNATKVMMLGSGELGKEVVIELQRLGVEVIAVDRYENAPAQQVAHRAYTISMLDGAALRALVEKEKPDFIVPEVEAIATATLVELEQEGYNVVPTAKATQLTMNREGIRRLAAEELGLKTSPYRFVDNLEDFKQAVA...`
- **Output preview**: `The analysis of the specified protein sequence suggests its potential function as ATP binding, ligas...`
- **Extracted protein**: `MTTIGTPLRPNATKVMMLGSGELGKEVVIELQRLGVEVIAVDRYENAPAQQVAHRAYTISMLDGAALRALVEKEKPDFIVPEVEAIATATLVELEQEGYN...` (len=393)
- **Quality**: **valid_protein**

</details>

<details><summary>Record 1: backtick_protein → valid_protein</summary>

- **Instruction**: `Please examine the protein encoded by the amino acid sequence and describe its functional role, potential involvement in`
- **Input** (backtick_protein): ````
MPARRKYAVRDKWKLKKWYEVIAPPVFGNIVIGTTPADDPLKLIGRVMETTLYDITGDITQVHVRLYFQIIDVKENKAITRFKGHELSRDYIKSLIRRKSSKIQGIFNVVTKDGYHLRLTIIALTSYRCKTSQKRAIRKIMEEYVK...`
- **Output preview**: `Based on the given amino acid sequence, the protein appears to have a primary function of structural...`
- **Extracted protein**: `MPARRKYAVRDKWKLKKWYEVIAPPVFGNIVIGTTPADDPLKLIGRVMETTLYDITGDITQVHVRLYFQIIDVKENKAITRFKGHELSRDYIKSLIRRKS...` (len=212)
- **Quality**: **valid_protein**

</details>

<details><summary>Record 2: backtick_protein → valid_protein</summary>

- **Instruction**: `Please examine the protein encoded by the amino acid sequence and describe its functional role, potential involvement in`
- **Input** (backtick_protein): ````
MTTFVDRVELHVAAGNGGHGCASVHREKFKPLGGPDGGNGGRGGDVILTVDQSVTTLLDYHHSPHRKATNGKPGEGGNRSGKDGQDLVLPVPDGTVVLDGAGNVLADLVGHGTSYVAAQGGRGGLGNAALASARRKAPGFALLGEP...`
- **Output preview**: `Based on the given amino acid sequence, the protein appears to have a primary function of GTP bindin...`
- **Extracted protein**: `MTTFVDRVELHVAAGNGGHGCASVHREKFKPLGGPDGGNGGRGGDVILTVDQSVTTLLDYHHSPHRKATNGKPGEGGNRSGKDGQDLVLPVPDGTVVLDG...` (len=478)
- **Quality**: **valid_protein**

</details>

<details><summary>Record 3: backtick_protein → valid_protein</summary>

- **Instruction**: `Provide a detailed description of the protein with the following amino acid sequence, including its function, subcellula`
- **Input** (backtick_protein): ````
MSSITKRDKVIIGSRKSQLAMLQTEWVRDRIQELNPGIIVEIKTMDTTGDKVLDVSLSKIGDKGLFTKELEDMMLNGTIDLAVHSLKDIPTKLPDGLKLGAITKRYNTSDAFIANAKKHGKNCKLSELPQGAMIGSSSLRRVAQLK...`
- **Output preview**: `The protein with the amino acid sequence is expected to exhibit hydroxymethylbilane synthase activit...`
- **Extracted protein**: `MSSITKRDKVIIGSRKSQLAMLQTEWVRDRIQELNPGIIVEIKTMDTTGDKVLDVSLSKIGDKGLFTKELEDMMLNGTIDLAVHSLKDIPTKLPDGLKLG...` (len=325)
- **Quality**: **valid_protein**

</details>

<details><summary>Record 4: backtick_protein → valid_protein</summary>

- **Instruction**: `Please examine the protein encoded by the amino acid sequence and describe its functional role, potential involvement in`
- **Input** (backtick_protein): ````
MSIKLGYLFKLPSYKTSSISPEVIVDLEKFLNLCEGSCSQCNTPFCVDVGPRNAIIQLPNGADVLNLNPNFQAIFECCSKNSLTGVQIFGMYNDGTYELRSFCPVHGVNEDPANGSGAGSVGVFFALNNPSIISSDFAHLLFNQGK...`
- **Output preview**: `Based on the given amino acid sequence, the protein appears to have a primary function of isomerase ...`
- **Extracted protein**: `MSIKLGYLFKLPSYKTSSISPEVIVDLEKFLNLCEGSCSQCNTPFCVDVGPRNAIIQLPNGADVLNLNPNFQAIFECCSKNSLTGVQIFGMYNDGTYELR...` (len=184)
- **Quality**: **valid_protein**

</details>

### `sp_gene_prediction.json`

- **Records**: 263,061
- **Source**: `sp_`
- **Input type distribution**: {'backtick_protein': 262273, 'backtick_mixed': 788}
- **Extraction quality**: {'valid_protein': 263028, 'partial_protein': 33}
- **Protein lengths**: min=50, max=1000, mean=331, median=297

**Samples:**

<details><summary>Record 0: backtick_protein → valid_protein</summary>

- **Instruction**: `Predict the gene name associated with this protein sequence:`
- **Input** (backtick_protein): ````
MAFSAEDVLKEYDRRRRMEALLLSLYYPNDRKLLDYKEWSPPRVQVECPKAPVEWNNPPSEKGLIVGHFSGIKYKGEKAQASEVDVNKMCCWVSKFKDAMRRYQGIQTCKIPGKVLSDLDAKIKAYNLTVEGVEGFVRYSRVTKQH...`
- **Output preview**: `The gene encoding this protein is FV3-001R....`
- **Extracted protein**: `MAFSAEDVLKEYDRRRRMEALLLSLYYPNDRKLLDYKEWSPPRVQVECPKAPVEWNNPPSEKGLIVGHFSGIKYKGEKAQASEVDVNKMCCWVSKFKDAM...` (len=256)
- **Quality**: **valid_protein**

</details>

<details><summary>Record 1: backtick_protein → valid_protein</summary>

- **Instruction**: `Based on the amino acid sequence, predict the corresponding gene name:`
- **Input** (backtick_protein): ````
MASNTVSAQGGSNRPVRDFSNIQDVAQFLLFDPIWNEQPGSIVPWKMNREQALAERYPELQTSEPSEDYSGPVESLELLPLEIKLDIMQYLSWEQISWCKHPWLWTRWYKDNVVRVSAITFEDFQREYAFPEKIQEIHFTDTRAEE...`
- **Output preview**: `The gene encoding this protein is IIV3-002R....`
- **Extracted protein**: `MASNTVSAQGGSNRPVRDFSNIQDVAQFLLFDPIWNEQPGSIVPWKMNREQALAERYPELQTSEPSEDYSGPVESLELLPLEIKLDIMQYLSWEQISWCK...` (len=458)
- **Quality**: **valid_protein**

</details>

<details><summary>Record 2: backtick_protein → valid_protein</summary>

- **Instruction**: `Predict the gene name associated with this protein sequence:`
- **Input** (backtick_protein): ````
MNAKYDTDQGVGRMLFLGTIGLAVVVGGLMAYGYYYDGKTPSSGTSFHTASPSFSSRYRY
```...`
- **Output preview**: `The gene encoding this protein is FV3-004R....`
- **Extracted protein**: `MNAKYDTDQGVGRMLFLGTIGLAVVVGGLMAYGYYYDGKTPSSGTSFHTASPSFSSRYRY` (len=60)
- **Quality**: **valid_protein**

</details>

<details><summary>Record 3: backtick_protein → valid_protein</summary>

- **Instruction**: `Given the protein sequence below, identify the gene that encodes it:`
- **Input** (backtick_protein): ````
MRYTVLIALQGALLLLLLIDDGQGQSPYPYPGMPCNSSRQCGLGTCVHSRCAHCSSDGTLCSPEDPTMVWPCCPESSCQLVVGLPSLVNHYNCLPNQCTDSSQCPGGFGCMTRRSKCELCKADGEACNSPYLDWRKDKECCSGYCH...`
- **Output preview**: `The gene encoding this protein is IIV3-005L....`
- **Extracted protein**: `MRYTVLIALQGALLLLLLIDDGQGQSPYPYPGMPCNSSRQCGLGTCVHSRCAHCSSDGTLCSPEDPTMVWPCCPESSCQLVVGLPSLVNHYNCLPNQCTD...` (len=217)
- **Quality**: **valid_protein**

</details>

<details><summary>Record 4: backtick_protein → valid_protein</summary>

- **Instruction**: `Given the protein sequence below, identify the gene that encodes it:`
- **Input** (backtick_protein): ````
MEAKNITIDNTTYNFFKFYNINQPLTNLKYLNSERLCFSNAVMGKIVDDASTITITYHRVYFGISGPKPRQVADLGEYYDVNELLNYDTYTKTQEFAQKYNSLVKPTIDAKNWSGNELVLLVGNEWYCKTFGKAGSKNVFLYNMIP...`
- **Output preview**: `The gene encoding this protein is IIV3-007R....`
- **Extracted protein**: `MEAKNITIDNTTYNFFKFYNINQPLTNLKYLNSERLCFSNAVMGKIVDDASTITITYHRVYFGISGPKPRQVADLGEYYDVNELLNYDTYTKTQEFAQKY...` (len=447)
- **Quality**: **valid_protein**

</details>

### `sp_general_function.json`

- **Records**: 542,287
- **Source**: `sp_`
- **Input type distribution**: {'backtick_protein': 540376, 'backtick_mixed': 1911}
- **Extraction quality**: {'valid_protein': 542162, 'partial_protein': 124, 'not_protein': 1}
- **Protein lengths**: min=50, max=1000, mean=328, median=293

**Samples:**

<details><summary>Record 0: backtick_protein → valid_protein</summary>

- **Instruction**: `Analyze the protein with the following sequence and describe its properties:`
- **Input** (backtick_protein): ````
MAFSAEDVLKEYDRRRRMEALLLSLYYPNDRKLLDYKEWSPPRVQVECPKAPVEWNNPPSEKGLIVGHFSGIKYKGEKAQASEVDVNKMCCWVSKFKDAMRRYQGIQTCKIPGKVLSDLDAKIKAYNLTVEGVEGFVRYSRVTKQH...`
- **Output preview**: `This protein is Putative transcription factor 001R, found in Frog virus 3 (isolate Goorha)....`
- **Extracted protein**: `MAFSAEDVLKEYDRRRRMEALLLSLYYPNDRKLLDYKEWSPPRVQVECPKAPVEWNNPPSEKGLIVGHFSGIKYKGEKAQASEVDVNKMCCWVSKFKDAM...` (len=256)
- **Quality**: **valid_protein**

</details>

<details><summary>Record 1: backtick_protein → valid_protein</summary>

- **Instruction**: `Could you evaluate the protein with this amino acid sequence and present a summary of its features? The sequence is:`
- **Input** (backtick_protein): ````
MSIIGATRLQNDKSDTYSAGPCYAGGCSAFTPRGTCGKDWDLGEQTCASGFCTSQPLCARIKKTQVCGLRYSSKGKDPLVSAEWDSRGAPYVRCTYDADLIDTQAQVDQFVSMFGESPSLAERYCMRGVKNTAGELVSRVSSDADP...`
- **Output preview**: `This protein is Uncharacterized protein 002L, found in Frog virus 3 (isolate Goorha)....`
- **Extracted protein**: `MSIIGATRLQNDKSDTYSAGPCYAGGCSAFTPRGTCGKDWDLGEQTCASGFCTSQPLCARIKKTQVCGLRYSSKGKDPLVSAEWDSRGAPYVRCTYDADL...` (len=320)
- **Quality**: **valid_protein**

</details>

<details><summary>Record 2: backtick_protein → valid_protein</summary>

- **Instruction**: `Analyze the protein with the following sequence and describe its properties:`
- **Input** (backtick_protein): ````
MASNTVSAQGGSNRPVRDFSNIQDVAQFLLFDPIWNEQPGSIVPWKMNREQALAERYPELQTSEPSEDYSGPVESLELLPLEIKLDIMQYLSWEQISWCKHPWLWTRWYKDNVVRVSAITFEDFQREYAFPEKIQEIHFTDTRAEE...`
- **Output preview**: `This protein is Uncharacterized protein 002R, found in Invertebrate iridescent virus 3....`
- **Extracted protein**: `MASNTVSAQGGSNRPVRDFSNIQDVAQFLLFDPIWNEQPGSIVPWKMNREQALAERYPELQTSEPSEDYSGPVESLELLPLEIKLDIMQYLSWEQISWCK...` (len=458)
- **Quality**: **valid_protein**

</details>

<details><summary>Record 3: backtick_protein → valid_protein</summary>

- **Instruction**: `Inspect the protein with the subsequent sequence and offer a concise description of its properties:`
- **Input** (backtick_protein): ````
MYQAINPCPQSWYGSPQLEREIVCKMSGAPHYPNYYPVHPNALGGAWFDTSLNARSLTTTPSLTTCTPPSLAACTPPTSLGMVDSPPHINPPRRIGTLCFDFGSAKSPQRCECVASDRPSTTSNTAPDTYRLLITNSKTRKNNYGT...`
- **Output preview**: `This protein is Uncharacterized protein 003L, found in Invertebrate iridescent virus 3....`
- **Extracted protein**: `MYQAINPCPQSWYGSPQLEREIVCKMSGAPHYPNYYPVHPNALGGAWFDTSLNARSLTTTPSLTTCTPPSLAACTPPTSLGMVDSPPHINPPRRIGTLCF...` (len=156)
- **Quality**: **valid_protein**

</details>

<details><summary>Record 4: backtick_protein → valid_protein</summary>

- **Instruction**: `Assess the following protein sequence and provide a brief report on its primary characteristics:`
- **Input** (backtick_protein): ````
MARPLLGKTSSVRRRLESLSACSIFFFLRKFCQKMASLVFLNSPVYQMSNILLTERRQVDRAMGGSDDDGVMVVALSPSDFKTVLGSALLAVERDMVHVVPKYLQTPGILHDMLVLLTPIFGEALSVDMSGATDVMVQQIATAGFV...`
- **Output preview**: `This protein is Uncharacterized protein 3R, found in Frog virus 3 (isolate Goorha)....`
- **Extracted protein**: `MARPLLGKTSSVRRRLESLSACSIFFFLRKFCQKMASLVFLNSPVYQMSNILLTERRQVDRAMGGSDDDGVMVVALSPSDFKTVLGSALLAVERDMVHVV...` (len=438)
- **Quality**: **valid_protein**

</details>

### `sp_organism_prediction.json`

- **Records**: 271,498
- **Source**: `sp_`
- **Input type distribution**: {'backtick_protein': 270529, 'backtick_mixed': 969}
- **Extraction quality**: {'valid_protein': 271430, 'partial_protein': 68}
- **Protein lengths**: min=50, max=1000, mean=328, median=293

**Samples:**

<details><summary>Record 0: backtick_protein → valid_protein</summary>

- **Instruction**: `Which organism does this protein sequence come from? Analyze the sequence below:`
- **Input** (backtick_protein): ````
MASNTVSAQGGSNRPVRDFSNIQDVAQFLLFDPIWNEQPGSIVPWKMNREQALAERYPELQTSEPSEDYSGPVESLELLPLEIKLDIMQYLSWEQISWCKHPWLWTRWYKDNVVRVSAITFEDFQREYAFPEKIQEIHFTDTRAEE...`
- **Output preview**: `This protein originates from Invertebrate iridescent virus 3 (taxonomy ID: 345201)....`
- **Extracted protein**: `MASNTVSAQGGSNRPVRDFSNIQDVAQFLLFDPIWNEQPGSIVPWKMNREQALAERYPELQTSEPSEDYSGPVESLELLPLEIKLDIMQYLSWEQISWCK...` (len=458)
- **Quality**: **valid_protein**

</details>

<details><summary>Record 1: backtick_protein → valid_protein</summary>

- **Instruction**: `Predict the organism of origin for the protein represented by this sequence:`
- **Input** (backtick_protein): ````
MNAKYDTDQGVGRMLFLGTIGLAVVVGGLMAYGYYYDGKTPSSGTSFHTASPSFSSRYRY
```...`
- **Output preview**: `This protein originates from Frog virus 3 (isolate Goorha) (taxonomy ID: 654924)....`
- **Extracted protein**: `MNAKYDTDQGVGRMLFLGTIGLAVVVGGLMAYGYYYDGKTPSSGTSFHTASPSFSSRYRY` (len=60)
- **Quality**: **valid_protein**

</details>

<details><summary>Record 2: backtick_protein → valid_protein</summary>

- **Instruction**: `Given the following protein sequence, identify the organism it originates from:`
- **Input** (backtick_protein): ````
MRYTVLIALQGALLLLLLIDDGQGQSPYPYPGMPCNSSRQCGLGTCVHSRCAHCSSDGTLCSPEDPTMVWPCCPESSCQLVVGLPSLVNHYNCLPNQCTDSSQCPGGFGCMTRRSKCELCKADGEACNSPYLDWRKDKECCSGYCH...`
- **Output preview**: `This protein originates from Invertebrate iridescent virus 3 (taxonomy ID: 345201)....`
- **Extracted protein**: `MRYTVLIALQGALLLLLLIDDGQGQSPYPYPGMPCNSSRQCGLGTCVHSRCAHCSSDGTLCSPEDPTMVWPCCPESSCQLVVGLPSLVNHYNCLPNQCTD...` (len=217)
- **Quality**: **valid_protein**

</details>

<details><summary>Record 3: backtick_protein → valid_protein</summary>

- **Instruction**: `Given the following protein sequence, identify the organism it originates from:`
- **Input** (backtick_protein): ````
MQNPLPEVMSPEHDKRTTTPMSKEANKFIRELDKKPGDLAVVSDFVKRNTGKRLPIGKRSNLYVRICDLSGTIYMGETFILESWEELYLPEPTKMEVLGTLESCCGIPPFPEWIVMVGEDQCVYAYGDEEILLFAYSVKQLVEEGI...`
- **Output preview**: `This protein originates from Frog virus 3 (isolate Goorha) (taxonomy ID: 654924)....`
- **Extracted protein**: `MQNPLPEVMSPEHDKRTTTPMSKEANKFIRELDKKPGDLAVVSDFVKRNTGKRLPIGKRSNLYVRICDLSGTIYMGETFILESWEELYLPEPTKMEVLGT...` (len=204)
- **Quality**: **valid_protein**

</details>

<details><summary>Record 4: backtick_protein → valid_protein</summary>

- **Instruction**: `Determine the source organism for the protein with the following amino acid sequence:`
- **Input** (backtick_protein): ````
MDSLNEVCYEQIKGTFYKGLFGDFPLIVDKKTGCFNATKLCVLGGKRFVDWNKTLRSKKLIQYYETRCDIKTESLLYEIKGDNNDEITKQITGTYLPKEFILDIASWISVEFYDKCNNIIINYFVNEYKTMDKKTLQSKINEVEEK...`
- **Output preview**: `This protein originates from Invertebrate iridescent virus 6 (taxonomy ID: 176652)....`
- **Extracted protein**: `MDSLNEVCYEQIKGTFYKGLFGDFPLIVDKKTGCFNATKLCVLGGKRFVDWNKTLRSKKLIQYYETRCDIKTESLLYEIKGDNNDEITKQITGTYLPKEF...` (len=352)
- **Quality**: **valid_protein**

</details>

### `wp_disease_association.json`

- **Records**: 1,763
- **Source**: `wp_`
- **Input type distribution**: {'backtick_protein': 1631, 'backtick_mixed': 132}
- **Extraction quality**: {'valid_protein': 1760, 'partial_protein': 3}
- **Protein lengths**: min=54, max=995, mean=307, median=265

**Samples:**

<details><summary>Record 0: backtick_protein → valid_protein</summary>

- **Instruction**: `Describe the clinical significance of this protein.`
- **Input** (backtick_protein): ````
MTATDNARQVTIIGAGLAGTLVARLLARNGWQVNLFERRPDPRIETGARGRSINLALAERGAHALRLAGLEREVLAEAVMMRGRMVHVPGTPPNLQPYGRDDSEVIWSINRDRLNRILLDGAEAAGASIHFNLGLDSVDFARQRLT...`
- **Output preview**: `Kynurenine 3-monooxygenase is an attractive drug target for several neurodegenerative and neuroinfla...`
- **Extracted protein**: `MTATDNARQVTIIGAGLAGTLVARLLARNGWQVNLFERRPDPRIETGARGRSINLALAERGAHALRLAGLEREVLAEAVMMRGRMVHVPGTPPNLQPYGR...` (len=461)
- **Quality**: **valid_protein**

</details>

<details><summary>Record 1: backtick_protein → valid_protein</summary>

- **Instruction**: `Describe the clinical significance of this protein.`
- **Input** (backtick_protein): ````
LKENPSGRPRSKEGDKGCGALVWVGEPVTLRTAETIAGKYGVWMRDPKPTHPYTQESTWRIDTVGTEIRQVFEYSQISQFEQGYPSKVHVLPRALESTGAVVYAGSLYFQGAESRTVVRYELDTETVKAEKEIPGAGYHGHFPYAW...`
- **Output preview**: `MYOC contains a signal sequence for secretion and is secreted into the aqueous humor of the eye by t...`
- **Extracted protein**: `LKENPSGRPRSKEGDKGCGALVWVGEPVTLRTAETIAGKYGVWMRDPKPTHPYTQESTWRIDTVGTEIRQVFEYSQISQFEQGYPSKVHVLPRALESTGA...` (len=277)
- **Quality**: **valid_protein**

</details>

<details><summary>Record 2: backtick_mixed → valid_protein</summary>

- **Instruction**: `Are there any disease associations known for this protein? Explain.`
- **Input** (backtick_mixed): ````
KKHTGYVGLKNQGATCYXNSLLQTLFFTNQLRKAVYXXPTEGDDSSKSVPLALQRVFYELQHSDKPVGTKKLTKSFGWETLDSFXQHDVQELCRVLLDNVENKXKGTCVEGTIPKLFRGKXVSYIQCKEVDYRSDRREDYYDIQLS...`
- **Output preview**: `Loss-of-function mutations of USP7 are associated with neurodevelopmental disorder, called Hao-Fount...`
- **Extracted protein**: `KKHTGYVGLKNQGATCYXNSLLQTLFFTNQLRKAVYXXPTEGDDSSKSVPLALQRVFYELQHSDKPVGTKKLTKSFGWETLDSFXQHDVQELCRVLLDNV...` (len=353)
- **Quality**: **valid_protein**

</details>

<details><summary>Record 3: backtick_protein → valid_protein</summary>

- **Instruction**: `What is the role of this protein in human disease?`
- **Input** (backtick_protein): ````
KKHTGYVGLKNQGATCYMNSLLQTLFFTNQLRKAVYMMPTEGDDSSKSVPLALQRVFYELQHSDKPVGTKKLTKSFGWETLDSFMQHDVQELCRVLLDNVENKMKGTCVEGTIPKLFRGKMVSYIQCKEVDYRSDRREDYYDIQLS...`
- **Output preview**: `Loss of a single UBC allele has no apparent phenotype, while homozygous deletion of UBC gene leads t...`
- **Extracted protein**: `KKHTGYVGLKNQGATCYMNSLLQTLFFTNQLRKAVYMMPTEGDDSSKSVPLALQRVFYELQHSDKPVGTKKLTKSFGWETLDSFMQHDVQELCRVLLDNV...` (len=353)
- **Quality**: **valid_protein**

</details>

<details><summary>Record 4: backtick_protein → valid_protein</summary>

- **Instruction**: `Are there any disease associations known for this protein? Explain.`
- **Input** (backtick_protein): ````
APTSSSTKKTQLQLEHLLLDLQMILNGINNCKNPKLTRMLTFKFYMPKKATELKHLQCLEEELKPLEEVLNLAQSKNFHLRPRDLISNINVIVLELKGSETTFMCEYADETATIVEFLNRWITFCQSIISTLT
```...`
- **Output preview**: `While the causes of itchiness are poorly understood, some evidence indicates that IL-2 is involved i...`
- **Extracted protein**: `APTSSSTKKTQLQLEHLLLDLQMILNGINNCKNPKLTRMLTFKFYMPKKATELKHLQCLEEELKPLEEVLNLAQSKNFHLRPRDLISNINVIVLELKGSE...` (len=133)
- **Quality**: **valid_protein**

</details>

### `wp_protein_function.json`

- **Records**: 3,507
- **Source**: `wp_`
- **Input type distribution**: {'backtick_protein': 3182, 'backtick_mixed': 325}
- **Extraction quality**: {'valid_protein': 3495, 'partial_protein': 12}
- **Protein lengths**: min=50, max=995, mean=310, median=271

**Samples:**

<details><summary>Record 0: backtick_protein → valid_protein</summary>

- **Instruction**: `What is the function of this protein?`
- **Input** (backtick_protein): ````
MTATDNARQVTIIGAGLAGTLVARLLARNGWQVNLFERRPDPRIETGARGRSINLALAERGAHALRLAGLEREVLAEAVMMRGRMVHVPGTPPNLQPYGRDDSEVIWSINRDRLNRILLDGAEAAGASIHFNLGLDSVDFARQRLT...`
- **Output preview**: `Kynurenine 3-monooxygenase catalyzes the conversion of -kynurenine to 3-hydroxy--kynurenine, an impo...`
- **Extracted protein**: `MTATDNARQVTIIGAGLAGTLVARLLARNGWQVNLFERRPDPRIETGARGRSINLALAERGAHALRLAGLEREVLAEAVMMRGRMVHVPGTPPNLQPYGR...` (len=461)
- **Quality**: **valid_protein**

</details>

<details><summary>Record 1: backtick_protein → valid_protein</summary>

- **Instruction**: `Describe the biological function of the protein with the following sequence:`
- **Input** (backtick_protein): ````
LRLLDHRALVCSQPGLNCTVKNSTCLDDSWIHPRNLTPSSPKDLQIQLHFAHTQQGDLFPVAHIEWTLQTDASILYLEGAELSVLQLNTNERLCVRFEFLSKLRHHHRRWRFTFSHFVVDPDQEYEVTVHHLPKPIPDGDPNHQSK...`
- **Output preview**: `Numerous immune regulatory functions have been reported for the IL-17 family of cytokines, presumabl...`
- **Extracted protein**: `LRLLDHRALVCSQPGLNCTVKNSTCLDDSWIHPRNLTPSSPKDLQIQLHFAHTQQGDLFPVAHIEWTLQTDASILYLEGAELSVLQLNTNERLCVRFEFL...` (len=311)
- **Quality**: **valid_protein**

</details>

<details><summary>Record 2: backtick_mixed → valid_protein</summary>

- **Instruction**: `Analyze the following protein sequence and describe its molecular function:`
- **Input** (backtick_mixed): ````
GNNRALINDKLASLQYNPKTVXVFNGTSISNIDLPAEERFDDSTYIVXTREKCSYEADFDIAVPSAYEDVTYPGALLVASNDLLDGKPQELAVDKDRVNITVDLPGATDISFKVVPTFANVRAGINDILSKWFDSHGGEWSLPANF...`
- **Output preview**: `When an insect ingests these proteins, they are activated by proteolytic cleavage. The N-terminus is...`
- **Extracted protein**: `GNNRALINDKLASLQYNPKTVXVFNGTSISNIDLPAEERFDDSTYIVXTREKCSYEADFDIAVPSAYEDVTYPGALLVASNDLLDGKPQELAVDKDRVNI...` (len=474)
- **Quality**: **valid_protein**

</details>

<details><summary>Record 3: backtick_protein → valid_protein</summary>

- **Instruction**: `What is the function of this protein?`
- **Input** (backtick_protein): ````
LKENPSGRPRSKEGDKGCGALVWVGEPVTLRTAETIAGKYGVWMRDPKPTHPYTQESTWRIDTVGTEIRQVFEYSQISQFEQGYPSKVHVLPRALESTGAVVYAGSLYFQGAESRTVVRYELDTETVKAEKEIPGAGYHGHFPYAW...`
- **Output preview**: `MYOC encodes the protein myocilin. The precise function of myocilin is unknown, but it is normally s...`
- **Extracted protein**: `LKENPSGRPRSKEGDKGCGALVWVGEPVTLRTAETIAGKYGVWMRDPKPTHPYTQESTWRIDTVGTEIRQVFEYSQISQFEQGYPSKVHVLPRALESTGA...` (len=277)
- **Quality**: **valid_protein**

</details>

<details><summary>Record 4: backtick_protein → valid_protein</summary>

- **Instruction**: `Analyze the following protein sequence and describe its molecular function:`
- **Input** (backtick_protein): ````
KKHTGYVGLKNQGATCYMNSLLQTLFFTNQLRKAVYMMPTEGDDSSKSVPLALQRVFYELQHSDKPVGTKKLTKSFGWETLDSFMQHDVQELCRVLLDNVENKMKGTCVEGTIPKLFRGKMVSYIQCKEVDYRSDRREDYYDIQLS...`
- **Output preview**: `The diversity of polyubiquitin-C means that ubiquitylation contributes to the regulation of many cel...`
- **Extracted protein**: `KKHTGYVGLKNQGATCYMNSLLQTLFFTNQLRKAVYMMPTEGDDSSKSVPLALQRVFYELQHSDKPVGTKKLTKSFGWETLDSFMQHDVQELCRVLLDNV...` (len=353)
- **Quality**: **valid_protein**

</details>

### `wp_protein_overview.json`

- **Records**: 7,166
- **Source**: `wp_`
- **Input type distribution**: {'backtick_protein': 6429, 'backtick_mixed': 737}
- **Extraction quality**: {'valid_protein': 7146, 'partial_protein': 19, 'not_protein': 1}
- **Protein lengths**: min=50, max=998, mean=308, median=271

**Samples:**

<details><summary>Record 0: backtick_protein → valid_protein</summary>

- **Instruction**: `Describe this protein's biological role and key properties.`
- **Input** (backtick_protein): ````
MTATDNARQVTIIGAGLAGTLVARLLARNGWQVNLFERRPDPRIETGARGRSINLALAERGAHALRLAGLEREVLAEAVMMRGRMVHVPGTPPNLQPYGRDDSEVIWSINRDRLNRILLDGAEAAGASIHFNLGLDSVDFARQRLT...`
- **Output preview**: `In enzymology, a kynurenine 3-monooxygenase () is an enzyme that catalyzes the chemical reaction

:-...`
- **Extracted protein**: `MTATDNARQVTIIGAGLAGTLVARLLARNGWQVNLFERRPDPRIETGARGRSINLALAERGAHALRLAGLEREVLAEAVMMRGRMVHVPGTPPNLQPYGR...` (len=461)
- **Quality**: **valid_protein**

</details>

<details><summary>Record 1: backtick_protein → valid_protein</summary>

- **Instruction**: `Provide an overview of this protein, including its function and biological significance.`
- **Input** (backtick_protein): ````
LRLLDHRALVCSQPGLNCTVKNSTCLDDSWIHPRNLTPSSPKDLQIQLHFAHTQQGDLFPVAHIEWTLQTDASILYLEGAELSVLQLNTNERLCVRFEFLSKLRHHHRRWRFTFSHFVVDPDQEYEVTVHHLPKPIPDGDPNHQSK...`
- **Output preview**: `Interleukin 17 family (IL17 family) is a family of pro-inflammatory cystine knot cytokines. They are...`
- **Extracted protein**: `LRLLDHRALVCSQPGLNCTVKNSTCLDDSWIHPRNLTPSSPKDLQIQLHFAHTQQGDLFPVAHIEWTLQTDASILYLEGAELSVLQLNTNERLCVRFEFL...` (len=311)
- **Quality**: **valid_protein**

</details>

<details><summary>Record 2: backtick_protein → valid_protein</summary>

- **Instruction**: `Give a comprehensive overview of this protein's properties and biological importance.`
- **Input** (backtick_protein): ````
RHMQEILDAILSGDAASADYAALALPESYRAVTLHKGEERMFDGLASRDKDPRKSLHLDDVPLPELGPGEALVAVMASSVNYNTVWSSIFEPVSTFGFLERYGRLSPLTARHDLPYHVLGSDLAGVVLRTGAGVNAWKPGDEVVAH...`
- **Output preview**: `Dino Moras, born on 23 November 1944, is a French biochemist, research director at the CNRS and co-d...`
- **Extracted protein**: `RHMQEILDAILSGDAASADYAALALPESYRAVTLHKGEERMFDGLASRDKDPRKSLHLDDVPLPELGPGEALVAVMASSVNYNTVWSSIFEPVSTFGFLE...` (len=445)
- **Quality**: **valid_protein**

</details>

<details><summary>Record 3: backtick_mixed → valid_protein</summary>

- **Instruction**: `Describe this protein's biological role and key properties.`
- **Input** (backtick_mixed): ````
GNNRALINDKLASLQYNPKTVXVFNGTSISNIDLPAEERFDDSTYIVXTREKCSYEADFDIAVPSAYEDVTYPGALLVASNDLLDGKPQELAVDKDRVNITVDLPGATDISFKVVPTFANVRAGINDILSKWFDSHGGEWSLPANF...`
- **Output preview**: `Delta endotoxins (δ-endotoxins) are a family of pore-forming toxins produced by Bacillus thuringiens...`
- **Extracted protein**: `GNNRALINDKLASLQYNPKTVXVFNGTSISNIDLPAEERFDDSTYIVXTREKCSYEADFDIAVPSAYEDVTYPGALLVASNDLLDGKPQELAVDKDRVNI...` (len=474)
- **Quality**: **valid_protein**

</details>

<details><summary>Record 4: backtick_protein → valid_protein</summary>

- **Instruction**: `Describe this protein's biological role and key properties.`
- **Input** (backtick_protein): ````
GASKLRAVLEKLKLSRDDISTAAGMVKGVVDHLLLRLKCDSAFRGVGLLNTGSYYEHVKISAPNEFDVMFKLEVPRIQLEEYSNTRAYYFVKFKRNPKENPLSQFLEGEILSASKMLSKFRKIIKEEINDIKDTDVIMKRKRGGSP...`
- **Output preview**: `Hemoglobin (haemoglobin, Hb or Hgb) is a protein containing iron that facilitates the transportation...`
- **Extracted protein**: `GASKLRAVLEKLKLSRDDISTAAGMVKGVVDHLLLRLKCDSAFRGVGLLNTGSYYEHVKISAPNEFDVMFKLEVPRIQLEEYSNTRAYYFVKFKRNPKEN...` (len=362)
- **Quality**: **valid_protein**

</details>

### `wp_protein_structure.json`

- **Records**: 2,930
- **Source**: `wp_`
- **Input type distribution**: {'backtick_protein': 2640, 'backtick_mixed': 290}
- **Extraction quality**: {'valid_protein': 2920, 'partial_protein': 10}
- **Protein lengths**: min=50, max=993, mean=307, median=262

**Samples:**

<details><summary>Record 0: backtick_protein → valid_protein</summary>

- **Instruction**: `Provide details about this protein's three-dimensional structure.`
- **Input** (backtick_protein): ````
MTATDNARQVTIIGAGLAGTLVARLLARNGWQVNLFERRPDPRIETGARGRSINLALAERGAHALRLAGLEREVLAEAVMMRGRMVHVPGTPPNLQPYGRDDSEVIWSINRDRLNRILLDGAEAAGASIHFNLGLDSVDFARQRLT...`
- **Output preview**: `Kynurenine 3-monooxygenase is a dimer containing asymmetric subunits...`
- **Extracted protein**: `MTATDNARQVTIIGAGLAGTLVARLLARNGWQVNLFERRPDPRIETGARGRSINLALAERGAHALRLAGLEREVLAEAVMMRGRMVHVPGTPPNLQPYGR...` (len=461)
- **Quality**: **valid_protein**

</details>

<details><summary>Record 1: backtick_protein → valid_protein</summary>

- **Instruction**: `Describe the structure of this protein.`
- **Input** (backtick_protein): ````
LRLLDHRALVCSQPGLNCTVKNSTCLDDSWIHPRNLTPSSPKDLQIQLHFAHTQQGDLFPVAHIEWTLQTDASILYLEGAELSVLQLNTNERLCVRFEFLSKLRHHHRRWRFTFSHFVVDPDQEYEVTVHHLPKPIPDGDPNHQSK...`
- **Output preview**: `IL-17(A) is a 155-amino acid protein that is a disulfide-linked, homodimeric, secreted glycoprotein ...`
- **Extracted protein**: `LRLLDHRALVCSQPGLNCTVKNSTCLDDSWIHPRNLTPSSPKDLQIQLHFAHTQQGDLFPVAHIEWTLQTDASILYLEGAELSVLQLNTNERLCVRFEFL...` (len=311)
- **Quality**: **valid_protein**

</details>

<details><summary>Record 2: backtick_mixed → valid_protein</summary>

- **Instruction**: `Characterize the structural properties of this protein.`
- **Input** (backtick_mixed): ````
GNNRALINDKLASLQYNPKTVXVFNGTSISNIDLPAEERFDDSTYIVXTREKCSYEADFDIAVPSAYEDVTYPGALLVASNDLLDGKPQELAVDKDRVNITVDLPGATDISFKVVPTFANVRAGINDILSKWFDSHGGEWSLPANF...`
- **Output preview**: `The activated region of the delta toxin is composed of three distinct structural domains: an N-termi...`
- **Extracted protein**: `GNNRALINDKLASLQYNPKTVXVFNGTSISNIDLPAEERFDDSTYIVXTREKCSYEADFDIAVPSAYEDVTYPGALLVASNDLLDGKPQELAVDKDRVNI...` (len=474)
- **Quality**: **valid_protein**

</details>

<details><summary>Record 3: backtick_protein → valid_protein</summary>

- **Instruction**: `What are the structural features of the protein with the following sequence?`
- **Input** (backtick_protein): ````
LKENPSGRPRSKEGDKGCGALVWVGEPVTLRTAETIAGKYGVWMRDPKPTHPYTQESTWRIDTVGTEIRQVFEYSQISQFEQGYPSKVHVLPRALESTGAVVYAGSLYFQGAESRTVVRYELDTETVKAEKEIPGAGYHGHFPYAW...`
- **Output preview**: `The protein is made up of the two folding domains, the leucine zipper-like domain at the N-terminal ...`
- **Extracted protein**: `LKENPSGRPRSKEGDKGCGALVWVGEPVTLRTAETIAGKYGVWMRDPKPTHPYTQESTWRIDTVGTEIRQVFEYSQISQFEQGYPSKVHVLPRALESTGA...` (len=277)
- **Quality**: **valid_protein**

</details>

<details><summary>Record 4: backtick_protein → valid_protein</summary>

- **Instruction**: `What are the structural features of the protein with the following sequence?`
- **Input** (backtick_protein): ````
SMSYTWTGALITPCAAEESKLPINPLSNSLLRHHNMVYATTSRSASLRQKKVTFDRLQVLDDHYRDVLKEMKAKASTVKAKLLSIEEACKLTPPHSAKSKFGYGAKDVRNLSSRAVNHIRSVWEDLLEDTETPIDTTIMAKSEVFC...`
- **Output preview**: `thumb|upright=1.5|Structure of Hepatitis C Virus
The hepatitis C virus particle consists of a lipid ...`
- **Extracted protein**: `SMSYTWTGALITPCAAEESKLPINPLSNSLLRHHNMVYATTSRSASLRQKKVTFDRLQVLDDHYRDVLKEMKAKASTVKAKLLSIEEACKLTPPHSAKSK...` (len=570)
- **Quality**: **valid_protein**

</details>

## Recommendations

### Critical: `mol_protein_design.json` (195,975 records = 12.3% of dataset)

This is a **design task** where the input contains text requirements and the output contains the generated protein sequence. The current `_extract_protein_sequence()` runs on the `input` field only, so it returns garbage (the requirements text) for all 195,975 records.

**Impact on ESM-3 approach**: The encoder receives the requirements text as a "protein sequence", which will:
- ESM-3 tokenizer will tokenize it as unknown characters → garbage embeddings
- The model wastes 12.3% of training on meaningless encoder outputs
- May actively hurt projector training by introducing noise

**Options**:
1. **Exclude from ESM-3 training**: Only use `mol_protein_design` for the `text` approach (where it's fine — the LLM sees the full instruction+input→output as text). For ESM-3 approaches, filter out records where `_extract_protein_sequence()` fails.
2. **Extract from output**: For design tasks, extract the protein from the `output` field instead and swap the role — but this changes the task semantics for the encoder.
3. **Use both**: Extract protein from output, feed it to the encoder, and still train on the original instruction→output format. The encoder provides protein-aware context even though the task is generation.

**Recommended**: Option 1 — exclude `mol_protein_design.json` from ESM-3 training, keep it for text-only. This is the simplest and cleanest approach.

### Minor: ~239 `not_protein` + ~396 `partial_protein` across other files

Root cause: sequences containing runs of `X` (unknown/masked residues). The `X` character is not in the amino acid character set `ACDEFGHIKLMNPQRSTVWY`, so sequences like `MKVL...XXXXXXXXX...AGKT` have AA fraction < 0.9 and fail the line-by-line check.

When this happens, `_extract_protein_sequence()` falls through to its **fallback path** which returns `input_text.strip()` — the entire input INCLUDING the triple backtick wrapper. This means:
- Extracted "protein" starts with `` ``` `` followed by a newline
- ESM-3 receives a string like `` ```\nMKVL...XXX...\n``` `` which is NOT a valid protein

**Fix**: The extractor should:
1. Strip triple-backtick wrappers before analysis
2. Include `X` in the amino acid character set (it's the standard "any/unknown" residue)

Impact: ~635 records (0.05% of non-design data). Low urgency but easy to fix.

### Data Balance After Excluding Design

Without `mol_protein_design`:
- **Mol-Instructions**: 299,029 records (21.5%)
- **Swiss-Prot**: 1,076,846 records (77.4%)
- **Wikipedia Protein**: 15,366 records (1.1%)
- **Total**: 1,391,241 records

The `sampling_temperature=0.5` already handles this imbalance via upsampling.

### Protein Length Distribution

| Source | Min | Median | Mean | Max |
|--------|-----|--------|------|-----|
| mol_* (excl. design) | 10 | ~390 | ~410 | 768 |
| sp_* | 10 | ~320 | ~330 | varies |
| wp_* | 10 | ~308 | ~308 | varies |

Note: Mol-Instructions proteins are capped at 768 residues (dataset preprocessing limit).
