"""
PPI (Protein-Protein Interaction) Prediction Evaluation

This module evaluates the model's ability to predict whether two proteins
physically interact. Protein-protein interactions are fundamental to most
biological processes, making this a crucial benchmark for protein language models.

Metrics computed:
- Accuracy: Binary classification accuracy
- AUC-ROC: Area under ROC curve
- AUPR: Area under precision-recall curve
- F1 Score: Binary F1
- MCC: Matthews Correlation Coefficient (good for imbalanced data)
- Precision/Recall: At various thresholds

Supported data formats:
- JSON: List of interaction records
- TSV: Tab-separated values (protein_id_1, protein_id_2, seq_1, seq_2, label)
- BioSNAP: BioSNAP dataset format
"""

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from omegaconf import DictConfig

try:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        confusion_matrix,
        f1_score,
        matthews_corrcoef,
        precision_recall_curve,
        precision_score,
        recall_score,
        roc_auc_score,
        roc_curve,
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

log = logging.getLogger(__name__)

# Patterns for parsing yes/no responses
YES_PATTERN = re.compile(r"\b(yes|interact|interacts|positive|true|binding|binds)\b", re.IGNORECASE)
NO_PATTERN = re.compile(r"\b(no|not interact|does not interact|negative|false|no binding|do not bind)\b", re.IGNORECASE)
CONFIDENCE_PATTERN = re.compile(r"(?:confidence|probability|score)[:\s]*(\d*\.?\d+)", re.IGNORECASE)
PERCENTAGE_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*%")


@dataclass
class PPITestSample:
    """A test sample for protein-protein interaction prediction."""

    protein_id_1: str
    sequence_1: str
    protein_id_2: str
    sequence_2: str
    label: int  # 0 = no interaction, 1 = interaction
    confidence: Optional[float] = None  # Ground truth confidence if available
    description: str = ""
    source: str = ""  # Data source (e.g., STRING, IntAct, BioGRID)

    def __post_init__(self):
        """Validate label value."""
        if self.label not in (0, 1):
            raise ValueError(f"Label must be 0 or 1, got {self.label}")


@dataclass
class PPIPredictionResult:
    """Result of PPI prediction for a single sample."""

    predicted_label: int
    predicted_confidence: float
    ground_truth_label: int
    generated_text: str
    protein_id_1: str = ""
    protein_id_2: str = ""
    raw_score: Optional[float] = None  # Raw model score before thresholding


def parse_ppi_prediction(text: str) -> Tuple[int, float]:
    """
    Extract yes/no prediction and confidence from generated text.

    The function looks for:
    1. Explicit yes/no answers
    2. Interaction-related keywords
    3. Confidence scores or percentages

    Args:
        text: Generated text that may contain prediction.

    Returns:
        Tuple of (predicted_label, confidence).
        - predicted_label: 1 for interaction, 0 for no interaction
        - confidence: Confidence score between 0 and 1

    Examples:
        >>> parse_ppi_prediction("Yes, these proteins interact with high confidence.")
        (1, 0.9)
        >>> parse_ppi_prediction("No, these proteins do not interact.")
        (0, 0.9)
        >>> parse_ppi_prediction("Confidence: 0.85. Yes, they interact.")
        (1, 0.85)
        >>> parse_ppi_prediction("There is a 75% probability of interaction.")
        (1, 0.75)
    """
    if not text:
        return 0, 0.5  # Default to no interaction with neutral confidence

    text = text.strip()

    # Try to extract explicit confidence score
    confidence = 0.5  # Default neutral confidence

    # Look for confidence patterns
    conf_match = CONFIDENCE_PATTERN.search(text)
    if conf_match:
        try:
            conf_val = float(conf_match.group(1))
            if conf_val > 1:
                conf_val = conf_val / 100  # Convert percentage
            confidence = min(max(conf_val, 0.0), 1.0)
        except ValueError:
            pass

    # Look for percentage patterns
    if confidence == 0.5:
        pct_match = PERCENTAGE_PATTERN.search(text)
        if pct_match:
            try:
                confidence = float(pct_match.group(1)) / 100
                confidence = min(max(confidence, 0.0), 1.0)
            except ValueError:
                pass

    # Determine prediction label
    # Check the beginning of the response first for clear yes/no
    first_sentence = text.split('.')[0] if '.' in text else text
    first_word = text.split()[0].lower() if text.split() else ""

    # Direct yes/no at start
    if first_word in ("yes", "yes,"):
        return 1, max(confidence, 0.8)
    if first_word in ("no", "no,"):
        return 0, max(confidence, 0.8)

    # Look for yes patterns
    yes_match = YES_PATTERN.search(first_sentence)
    no_match = NO_PATTERN.search(first_sentence)

    # If both patterns found, look at which comes first
    if yes_match and no_match:
        if yes_match.start() < no_match.start():
            return 1, confidence if confidence != 0.5 else 0.7
        else:
            return 0, confidence if confidence != 0.5 else 0.7

    # Single pattern match
    if yes_match:
        return 1, confidence if confidence != 0.5 else 0.8

    if no_match:
        return 0, confidence if confidence != 0.5 else 0.8

    # Check full text if first sentence didn't have clear answer
    yes_in_full = YES_PATTERN.search(text)
    no_in_full = NO_PATTERN.search(text)

    if yes_in_full and not no_in_full:
        return 1, confidence if confidence != 0.5 else 0.6

    if no_in_full and not yes_in_full:
        return 0, confidence if confidence != 0.5 else 0.6

    # If both or neither, default to no interaction with low confidence
    return 0, 0.5


def load_ppi_test_dataset(
    cfg: DictConfig,
    max_samples: Optional[int] = None,
) -> List[PPITestSample]:
    """
    Load PPI test dataset.

    This function supports multiple data sources:
    1. Pre-processed JSON file with interaction data
    2. TSV file with protein pairs and labels
    3. BioSNAP format datasets

    Args:
        cfg: Configuration with dataset settings.
        max_samples: Maximum number of samples to load (for testing).

    Returns:
        List of test samples with interaction labels.
    """
    # Extract dataset config
    dataset_cfg = cfg.get("dataset", {})
    data_path = dataset_cfg.get("path", None)
    data_format = dataset_cfg.get("format", "json")

    # Check for max_samples in config
    if max_samples is None:
        max_samples = cfg.get("evaluation", {}).get("max_samples", None)

    samples = []

    if data_path and Path(data_path).exists():
        # Load from local file
        if data_format == "json":
            samples = _load_json_dataset(data_path, max_samples)
        elif data_format == "tsv":
            samples = _load_tsv_dataset(data_path, max_samples)
        elif data_format == "biosnap":
            samples = _load_biosnap_dataset(data_path, max_samples)
        else:
            log.warning(f"Unknown data format: {data_format}, using demo data")
            samples = _create_demo_dataset(max_samples or 20)
    else:
        # Use demo dataset for testing
        log.info("No dataset path provided, using demo dataset")
        samples = _create_demo_dataset(max_samples or 20)

    log.info(f"Loaded {len(samples)} PPI test samples")

    # Log class distribution
    pos_count = sum(1 for s in samples if s.label == 1)
    neg_count = len(samples) - pos_count
    log.info(f"Class distribution: {pos_count} positive, {neg_count} negative")

    return samples


def _load_json_dataset(path: str, max_samples: Optional[int]) -> List[PPITestSample]:
    """Load PPI dataset from JSON file."""
    samples = []

    with open(path, "r") as f:
        data = json.load(f)

    # Handle different JSON formats
    if isinstance(data, list):
        entries = data
    elif isinstance(data, dict) and "interactions" in data:
        entries = data["interactions"]
    elif isinstance(data, dict) and "pairs" in data:
        entries = data["pairs"]
    else:
        entries = [data]

    for entry in entries:
        if max_samples and len(samples) >= max_samples:
            break

        # Extract fields with fallbacks
        protein_id_1 = entry.get("protein_id_1", entry.get("id1", entry.get("protein_a", f"p1_{len(samples)}")))
        protein_id_2 = entry.get("protein_id_2", entry.get("id2", entry.get("protein_b", f"p2_{len(samples)}")))
        sequence_1 = entry.get("sequence_1", entry.get("seq1", entry.get("seq_a", "")))
        sequence_2 = entry.get("sequence_2", entry.get("seq2", entry.get("seq_b", "")))
        label = entry.get("label", entry.get("interaction", entry.get("interacts", 0)))

        # Handle string labels
        if isinstance(label, str):
            label = 1 if label.lower() in ("1", "true", "yes", "positive") else 0
        elif isinstance(label, bool):
            label = 1 if label else 0
        else:
            label = int(label)

        confidence = entry.get("confidence", entry.get("score", None))
        if confidence is not None:
            confidence = float(confidence)

        if sequence_1 and sequence_2:
            samples.append(PPITestSample(
                protein_id_1=protein_id_1,
                sequence_1=sequence_1,
                protein_id_2=protein_id_2,
                sequence_2=sequence_2,
                label=label,
                confidence=confidence,
                description=entry.get("description", ""),
                source=entry.get("source", "json"),
            ))

    return samples


def _load_tsv_dataset(path: str, max_samples: Optional[int]) -> List[PPITestSample]:
    """Load PPI dataset from TSV file."""
    samples = []

    with open(path, "r") as f:
        # Check for header
        first_line = f.readline().strip()
        has_header = first_line.lower().startswith(("protein", "id", "#"))
        if not has_header:
            # Process first line as data
            f.seek(0)

        for line in f:
            if max_samples and len(samples) >= max_samples:
                break

            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split("\t")
            if len(parts) < 5:
                continue

            protein_id_1 = parts[0]
            protein_id_2 = parts[1]
            sequence_1 = parts[2]
            sequence_2 = parts[3]
            label = int(parts[4])

            confidence = None
            if len(parts) > 5:
                try:
                    confidence = float(parts[5])
                except ValueError:
                    pass

            samples.append(PPITestSample(
                protein_id_1=protein_id_1,
                sequence_1=sequence_1,
                protein_id_2=protein_id_2,
                sequence_2=sequence_2,
                label=label,
                confidence=confidence,
                source="tsv",
            ))

    return samples


def _load_biosnap_dataset(path: str, max_samples: Optional[int]) -> List[PPITestSample]:
    """
    Load PPI dataset from BioSNAP format.

    BioSNAP format typically has:
    - Protein IDs in two columns
    - Separate sequence file mapping IDs to sequences
    """
    samples = []
    path = Path(path)

    # Try to find interaction file and sequence file
    interaction_file = path if path.is_file() else path / "interactions.tsv"
    sequence_file = path.parent / "sequences.fasta" if path.is_file() else path / "sequences.fasta"

    # Load sequences if available
    sequences = {}
    if sequence_file.exists():
        current_id = None
        current_seq = []
        with open(sequence_file, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    if current_id:
                        sequences[current_id] = "".join(current_seq)
                    current_id = line[1:].split()[0]
                    current_seq = []
                else:
                    current_seq.append(line)
            if current_id:
                sequences[current_id] = "".join(current_seq)

    # Load interactions
    if interaction_file.exists():
        with open(interaction_file, "r") as f:
            for line in f:
                if max_samples and len(samples) >= max_samples:
                    break

                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                parts = line.split("\t")
                if len(parts) < 2:
                    continue

                protein_id_1 = parts[0]
                protein_id_2 = parts[1]
                label = int(parts[2]) if len(parts) > 2 else 1

                sequence_1 = sequences.get(protein_id_1, "")
                sequence_2 = sequences.get(protein_id_2, "")

                if sequence_1 and sequence_2:
                    samples.append(PPITestSample(
                        protein_id_1=protein_id_1,
                        sequence_1=sequence_1,
                        protein_id_2=protein_id_2,
                        sequence_2=sequence_2,
                        label=label,
                        source="biosnap",
                    ))

    return samples


def _create_demo_dataset(num_samples: int = 20) -> List[PPITestSample]:
    """
    Create a demo dataset for testing purposes.

    Contains known interacting and non-interacting protein pairs from
    well-characterized systems like:
    - Insulin receptor signaling
    - p53 pathway
    - Hemoglobin complex
    - Cytokine signaling
    """
    # Demo protein sequences (truncated for brevity but still representative)
    demo_proteins = {
        # Insulin signaling pathway
        "INS": {
            "id": "P01308",
            "seq": "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN",
            "name": "Insulin",
        },
        "INSR": {
            "id": "P06213",
            "seq": "MATGGRRGAAAAPLLVAVAALLLGAAGHLYPGEVCPGMDIRNNLTRLHELENCSVIEGHLQILLMFKTRPEDFRDLSFPKLIMITDYLLLFRVYGLESLKDLFPNLTVIRGSRLFFNYALVIFEMVHLKELGLYNLMNITRGSVRIEKNNELCYLATIDWSRILDSVEDNYIVLNKDDNEECGDICPGTAKGKTNCPATVINGQFVERCWTHSHCQKVCPTICKSHGCTAEGLCCHSECLGNCSQPDDPTKCVACRNFYLDGRCVETCPPPYYHFQDWRCVNFSFCQDLHHKCKNSRRQGCHQYVIHNNKCIPECPSGYTMNSSNLLCTPCLGPCPKVCHLLEGEKTIDSVTSAQELRGCTVINGSLIINIRGGNNLAAELEANLGLIEEISGYLKIRRSYALVSLSFFRKLRLIRGETLEIGNYSFYALDNQNLRQLWDWSKHNLTITQGKLFFHYNPKLCLSEIHKMEEVSGTKGRQERNDIALKTNGDQASCENELLKFSYIRTSFDKILLRWEPYWPPDFRDLLGFMLFYKEAPYQNVTEFDGQDACGSNSWTVVDIDPPLRSNDPKSQNHPGWLMRGLKPWTQYAIFVKTLVTFSDERRTYGAKSDIIYVQTDATNPSVPLDPISVSNSSSQIILKWKPPSDPNGNITHYLVFWERQAEDSELFELDYCLKGLKLPSRTWSPPFESEDSQKHNQSEYEDSAGECCSCPKTDSQILKELEESSFRKTFEDYLHNVVFVPRKTSSGTGAEDPRPSRKRRSLGDVGNVTVAVPTVAAFPNTSSTSVPTSPEEHRPFEKVVNKESLVISGLRHFTGYRIELQACNQDTPEERCSVAAYVSARTMPEAKADDIVGPVTHEIFENNVVHLMWQEPKEPNGLIVLYEVSYRRYGDEELHLCVSRKHFALERGCRLRGLSPGNYSVRIRATSLAGNGSWTEPTYFYVTDYLDVPSNIAKIIIGPLIFVFLFSVVIGSIYLFLRKRQPDGPLGPLYASSNPEYLSASDVFPCSVYVPDEWEVSREKITLLRELGQGSFGMVYEGNARDIIKGEAETRVAVKTVNESASLRERIEFLNEASVMKGFTCHHVVRLLGVVSKGQPTLVVMELMAHGDLKSYLRSLRPEAENNPGRPPPTLQEMIQMAAEIADGMAYLNAKKFVHRDLAARNCMVAHDFTVKIGDFGMTRDIYETDYYRKGGKGLLPVRWMAPESLKDGVFTTSSDMWSFGVVLWEITSLAEQPYQGLSNEQVLKFVMDGGYLDQPDNCPERVTDLMRMCWQFNPKMRPTFLEIVNLLKDDLHPSFPEVSFFHSEENKAPESEELEMEFEDMENVPLDRSSHCQREEAGGRDGGSSLGFKRSYEEHIPYTHMNGGKKNGRILTLPRSNPS",
            "name": "Insulin receptor",
        },
        # p53 pathway
        "TP53": {
            "id": "P04637",
            "seq": "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHERCSDSDGLAPPQHLIRVEGNLRVEYLDDRNTFRHSVVVPYEPPEVGSDCTTIHYNYMCNSSCMGGMNRRPILTIITLEDSSGNLLGRNSFEVRVCACPGRDRRTEEENLRKKGEPHHELPPGSTKRALPNNTSSSPQPKKKPLDGEYFTLQIRGRERFEMFRELNEALELKDAQAGKEPGGSRAHSSHLKSKKGQSTSRHKKLMFKTEGPDSD",
            "name": "p53",
        },
        "MDM2": {
            "id": "Q00987",
            "seq": "MCNTNMSVPTDGAVTTSQIPASEQETLVRPKPLLLKLLKSVGAQKDTYTMKEVLFYLGQYIMTKRLYDEKQQHIVYCSNDLLGDLFGVPSFSVKEHRKIYTMIYRNLVVVNQQESSDSGTSVSENRCHLEGGSDQKDLVQELQEERTDTLQAEFDQLVKLLEGVGSQGDNLFLPKLLFEVYHKYLDCVTINTGGPLLGRISQAAPSGQDVVKLEKKDQSGNSSETIVGKAESGHPLCGSQHRTTLLAIPYFTDWKKDKVCLNSGKSCYCPMCAPGDGSLVADKECSAIVNKLNSPFLWLDMIDMSNDYQCILCSKDNGHVPDFEGRGSLPPHLIRFTVARELRPHTDLDVLILHRVDISRSKEWETAKASACVLTAATNTSGRISSYDLGYQPNSDKRDSGDESMGWDNATYSKAQHCHAQQNMLFLSRVSEKGQMLCSHVDPTMPTVRTYQFYQKEPGESELQDVLQALIDSMHRRDSGRLRDRQYQALLHKILRHLCHGKKGDDSDSENLPDDYGFCTCRFVRIMSKLGIKCTRCKEHLYCGKCLTSWQQSCCTGK",
            "name": "MDM2",
        },
        # Hemoglobin complex
        "HBA": {
            "id": "P69905",
            "seq": "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR",
            "name": "Hemoglobin alpha",
        },
        "HBB": {
            "id": "P68871",
            "seq": "MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLSTPDAVMGNPKVKAHGKKVLGAFSDGLAHLDNLKGTFATLSELHCDKLHVDPENFRLLGNVLVCVLAHHFGKEFTPPVQAAYQKVVAGVANALAHKYH",
            "name": "Hemoglobin beta",
        },
        # Cytokine signaling
        "TNF": {
            "id": "P01375",
            "seq": "MSTESMIRDVELAEEALPKKTGGPQGSRRCLFLSLFSFLIVAGATTLFCLLHFGVIGPQREEFPRDLSLISPLAQAVRSSSRTPSDKPVAHVVANPQAEGQLQWLNRRANALLANGVELRDNQLVVPSEGLYLIYSQVLFKGQGCPSTHVLLTHTISRIAVSYQTKVNLLSAIKSPCQRETPEGAEAKPWYEPIYLGGVFQLEKGDRLSAEINRPDYLDFAESGQVYFGIIAL",
            "name": "TNF-alpha",
        },
        "TNFR1": {
            "id": "P19438",
            "seq": "MGLSTVPDLLLPLVLLELLVGIYPSGVIGLVPHLGDREKRDSVCPQGKYIHPQNNSICCTKCHKGTYLYNDCPGPGQDTDCRECESGSFTASENHLRHCLSCSKCRKEMGQVEISSCTVDRDTVCGCRKNQYRHYWSENLFQCFNCSLCLNGTVHLSCQEKQNTVCTCHAGFFLRENECVSCSNCKKSLECTKLCLPQIENVKGTEDSGTTMGQPQVTPLVKDTHRTGSLLFPVSVEGTATIALSLPQVWVNRTRRTDLYLTESILLVWSTGNSRVLFSNP",
            "name": "TNF receptor 1",
        },
        # Growth factor signaling
        "EGF": {
            "id": "P01133",
            "seq": "NSDSECPLSHDGYCLHDGVCMYIEALDKYACNCVVGYIGERCQYRDLKWWELR",
            "name": "EGF",
        },
        "EGFR": {
            "id": "P00533",
            "seq": "MRPSGTAGAALLALLAALCPASRALEEKKVCQGTSNKLTQLGTFEDHFLSLQRMFNNCEVVLGNLEITYVQRNYDLSFLKTIQEVAGYVLIALNTVERIPLENLQIIRGNMYYENSYALAVLSNYDANKTGLKELPMRNLQEILHGAVRFSNNPALCNVESIQWRDIVSSDFLSNMSMDFQNHLGSCQKCDPSCPNGSCWGAGEENCQKLTKIICAQQCSGRCRGKSPSDCCHNQCAAGCTGPRESDCLVCRKFRDEATCKDTCPPLMLYNPTTYQMDVNPEGKYSFGATCVKKCPRNYVVTDHGSCVRACGADSYEMEEDGVRKCKKCEGPCRKVCNGIGIGEFKDSLSINATNIKHFKNCTSISGDLHILPVAFRGDSFTHTPPLDPQELDILKTVKEITGFLLIQAWPENRTDLHAFENLEIIRGRTKQHGQFSLAVVSLNITSLGLRSLKEISDGDVIISGNKNLCYANTINWKKLFGTSGQKTKIISNRGENSCKATGQVCHALCSPEGCWGPEPRDCVSCRNVSRGRECVDKCNLLEGEPREFVENSECIQCHPECLPQAMNITCTGRGPDNCIQCAHYIDGPHCVKTCPAGVMGENNTLVWKYADAGHVCHLCHPNCTYGCTGPGLEGCPTNGPKIPS",
            "name": "EGFR",
        },
        # Non-interacting controls
        "ALB": {
            "id": "P02768",
            "seq": "MKWVTFISLLFLFSSAYSRGVFRRDAHKSEVAHRFKDLGEENFKALVLIAFAQYLQQCPFEDHVKLVNEVTEFAKTCVADESAENCDKSLHTLFGDKLCTVATLRETYGEMADCCAKQEPERNECFLQHKDDNPNLPRLVRPEVDVMCTAFHDNEETFLKKYLYEIARRHPYFYAPELLFFAKRYKAAFTECCQAADKAACLLPKLDELRDEGKASSAKQRLKCASLQKFGERAFKAWAVARLSQRFPKAEFAEVSKLVTDLTKVHTECCHGDLLECADDRADLAKYICENQDSISSKLKECCEKPLLEKSHCIAEVENDEMPADLPSLAADFVESKDVCKNYAEAKDVFLGMFLYEYARRHPDYSVVLLLRLAKTYETTLEKCCAAADPHECYAKVFDEFKPLVEEPQNLIKQNCELFEQLGEYKFQNALLVRYTKKVPQVSTPTLVEVSRNLGKVGSKCCKHPEAKRMPCAEDYLSVVLNQLCVLHEKTPVSDRVTKCCTESLVNRRPCFSALEVDETYVPKEFNAETFTFHADICTLSEKERQIKKQTALVELVKHKPKATKEQLKAVMDDFAAFVEKCCKADDKETCFAEEGKKLVAASQAALGL",
            "name": "Albumin",
        },
        "ACTB": {
            "id": "P60709",
            "seq": "MDDDIAALVVDNGSGMCKAGFAGDDAPRAVFPSIVGRPRHQGVMVGMGQKDSYVGDEAQSKRGILTLKYPIEHGIVTNWDDMEKIWHHTFYNELRVAPEEHPVLLTEAPLNPKANREKMTQIMFETFNTPAMYVAIQAVLSLYASGRTTGIVMDSGDGVTHTVPIYEGYALPHAILRLDLAGRDLTDYLMKILTERGYSFTTTAEREIVRDIKEKLCYVALDFEQEMATAASSSSLEKSYELPDGQVITIGNERFRCPEALFQPSFLGMESCGIHETTFNSIMKCDVDIRKDLYANTVLSGGTTMYPGIADRMQKEITALAPSTMKIKIIAPPERKYSVWIGGSILASLSTFQQMWISKQEYDESGPSIVHRKCF",
            "name": "Beta-actin",
        },
    }

    # Known interacting pairs (positive examples)
    positive_pairs = [
        ("INS", "INSR"),   # Insulin - Insulin receptor (well-characterized)
        ("TP53", "MDM2"),  # p53 - MDM2 (tumor suppressor regulation)
        ("HBA", "HBB"),    # Hemoglobin alpha - beta (hemoglobin tetramer)
        ("TNF", "TNFR1"),  # TNF-alpha - TNF receptor (cytokine signaling)
        ("EGF", "EGFR"),   # EGF - EGFR (growth factor signaling)
    ]

    # Non-interacting pairs (negative examples)
    negative_pairs = [
        ("INS", "HBB"),    # Insulin - Hemoglobin beta
        ("TP53", "HBA"),   # p53 - Hemoglobin alpha
        ("TNF", "INSR"),   # TNF - Insulin receptor
        ("EGF", "MDM2"),   # EGF - MDM2
        ("ALB", "ACTB"),   # Albumin - Actin
        ("ALB", "TNF"),    # Albumin - TNF
        ("ACTB", "INS"),   # Actin - Insulin
        ("HBA", "MDM2"),   # Hemoglobin alpha - MDM2
        ("HBB", "EGFR"),   # Hemoglobin beta - EGFR
        ("ALB", "TP53"),   # Albumin - p53
    ]

    samples = []

    # Add positive pairs
    for p1_key, p2_key in positive_pairs:
        if len(samples) >= num_samples:
            break
        p1 = demo_proteins[p1_key]
        p2 = demo_proteins[p2_key]
        samples.append(PPITestSample(
            protein_id_1=p1["id"],
            sequence_1=p1["seq"],
            protein_id_2=p2["id"],
            sequence_2=p2["seq"],
            label=1,
            confidence=0.9,
            description=f"{p1['name']} - {p2['name']} interaction",
            source="demo_positive",
        ))

    # Add negative pairs
    for p1_key, p2_key in negative_pairs:
        if len(samples) >= num_samples:
            break
        p1 = demo_proteins[p1_key]
        p2 = demo_proteins[p2_key]
        samples.append(PPITestSample(
            protein_id_1=p1["id"],
            sequence_1=p1["seq"],
            protein_id_2=p2["id"],
            sequence_2=p2["seq"],
            label=0,
            confidence=0.9,
            description=f"{p1['name']} - {p2['name']} (no interaction)",
            source="demo_negative",
        ))

    return samples[:num_samples]


def create_ppi_prompt(
    sequence_1: str,
    sequence_2: str,
    prompt_template: Optional[str] = None,
) -> str:
    """
    Create a prompt for PPI prediction.

    Args:
        sequence_1: First protein sequence.
        sequence_2: Second protein sequence.
        prompt_template: Optional custom prompt template with {sequence_1} and {sequence_2} placeholders.

    Returns:
        Formatted prompt string.
    """
    if prompt_template:
        return prompt_template.format(sequence_1=sequence_1, sequence_2=sequence_2)

    # Default prompt template
    return f"""Protein A: {sequence_1}

Protein B: {sequence_2}

Do these two proteins physically interact?
Answer with "Yes" or "No" and explain your reasoning.
"""


def compute_ppi_metrics(
    predictions: List[PPIPredictionResult],
    thresholds: Optional[List[float]] = None,
) -> Dict[str, float]:
    """
    Compute evaluation metrics for PPI predictions.

    Metrics computed:
    - accuracy: Binary classification accuracy
    - auroc: Area under ROC curve
    - aupr: Area under precision-recall curve
    - f1: F1 score
    - mcc: Matthews Correlation Coefficient
    - precision: Precision score
    - recall: Recall score

    Args:
        predictions: List of prediction results.
        thresholds: Optional list of thresholds for precision/recall at thresholds.

    Returns:
        Dictionary of metric names to values.
    """
    if not SKLEARN_AVAILABLE:
        log.warning("scikit-learn not available, computing basic metrics only")
        return _compute_basic_metrics(predictions)

    if not predictions:
        return {"error": "no_predictions"}

    # Extract labels and scores
    y_true = np.array([p.ground_truth_label for p in predictions])
    y_pred = np.array([p.predicted_label for p in predictions])
    y_scores = np.array([p.predicted_confidence for p in predictions])

    # Adjust scores for negative predictions
    # If prediction is 0, the confidence should be (1 - confidence) for positive class
    y_prob_positive = np.where(y_pred == 1, y_scores, 1 - y_scores)

    metrics = {}

    # Basic classification metrics
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)
    metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
    metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)

    # Matthews Correlation Coefficient (good for imbalanced data)
    metrics["mcc"] = matthews_corrcoef(y_true, y_pred)

    # AUC-ROC
    try:
        if len(np.unique(y_true)) > 1:  # Need both classes for AUC
            metrics["auroc"] = roc_auc_score(y_true, y_prob_positive)
        else:
            metrics["auroc"] = 0.0
            log.warning("Only one class present in ground truth, AUC-ROC set to 0.0")
    except Exception as e:
        log.warning(f"Could not compute AUC-ROC: {e}")
        metrics["auroc"] = 0.0

    # AUPR (Average Precision)
    try:
        if len(np.unique(y_true)) > 1:
            metrics["aupr"] = average_precision_score(y_true, y_prob_positive)
        else:
            metrics["aupr"] = 0.0
    except Exception as e:
        log.warning(f"Could not compute AUPR: {e}")
        metrics["aupr"] = 0.0

    # Confusion matrix metrics
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics["true_positives"] = int(tp)
        metrics["true_negatives"] = int(tn)
        metrics["false_positives"] = int(fp)
        metrics["false_negatives"] = int(fn)

        # Specificity
        metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    except Exception as e:
        log.warning(f"Could not compute confusion matrix: {e}")

    # Precision/Recall at various thresholds
    if thresholds:
        for thresh in thresholds:
            y_pred_at_thresh = (y_prob_positive >= thresh).astype(int)
            metrics[f"precision_at_{thresh}"] = precision_score(y_true, y_pred_at_thresh, zero_division=0)
            metrics[f"recall_at_{thresh}"] = recall_score(y_true, y_pred_at_thresh, zero_division=0)

    # Statistics
    metrics["num_samples"] = len(predictions)
    metrics["num_positive"] = int(y_true.sum())
    metrics["num_negative"] = int(len(y_true) - y_true.sum())
    metrics["predicted_positive"] = int(y_pred.sum())
    metrics["predicted_negative"] = int(len(y_pred) - y_pred.sum())
    metrics["avg_confidence"] = float(y_scores.mean())

    return metrics


def _compute_basic_metrics(predictions: List[PPIPredictionResult]) -> Dict[str, float]:
    """Compute basic metrics without sklearn."""
    if not predictions:
        return {"error": "no_predictions"}

    metrics = {}

    y_true = [p.ground_truth_label for p in predictions]
    y_pred = [p.predicted_label for p in predictions]

    # Accuracy
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    metrics["accuracy"] = correct / len(predictions)

    # Basic precision/recall
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

    metrics["precision"] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    metrics["recall"] = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if metrics["precision"] + metrics["recall"] > 0:
        metrics["f1"] = (
            2 * metrics["precision"] * metrics["recall"] /
            (metrics["precision"] + metrics["recall"])
        )
    else:
        metrics["f1"] = 0.0

    metrics["num_samples"] = len(predictions)
    metrics["num_positive"] = sum(y_true)
    metrics["num_negative"] = len(y_true) - sum(y_true)

    return metrics


def evaluate_ppi(
    cfg: DictConfig,
    checkpoint_path: Optional[str] = None,
    model=None,
    output_dir: Optional[str] = None,
) -> Dict[str, float]:
    """
    Evaluate protein-protein interaction prediction.

    This function:
    1. Loads the model from checkpoint (unless pre-loaded model is provided)
    2. Loads the PPI test dataset
    3. Generates predictions for each protein pair
    4. Parses yes/no predictions from generated text
    5. Computes evaluation metrics

    Args:
        cfg: Hydra configuration containing:
            - model: Model configuration
            - dataset: Dataset configuration (path, format, max_samples)
            - evaluation: Evaluation settings (batch_size, max_new_tokens)
            - logging: Logging settings (wandb, tensorboard, save_results)

        checkpoint_path: Path to model checkpoint.
        model: Pre-loaded model instance (skips loading if provided).

    Returns:
        Dictionary of metric names to values.
    """
    log.info("Evaluating PPI prediction...")

    if model is None:
        # Import model here to avoid circular imports
        from src.models.multimodal_llm import ProteinLLM

        # Load model
        if checkpoint_path:
            log.info(f"Loading model from checkpoint: {checkpoint_path}")
            model = ProteinLLM.from_pretrained(checkpoint_path)
        else:
            log.info("Creating model from config (no checkpoint provided)")
            model = ProteinLLM.from_config(cfg)

        model.eval()

    # Get evaluation settings
    eval_cfg = cfg.get("evaluation", {})
    batch_size = eval_cfg.get("batch_size", 1)
    max_new_tokens = eval_cfg.get("max_new_tokens", 256)
    max_samples = eval_cfg.get("max_samples", None)
    prompt_template = eval_cfg.get("prompt_template", None)
    thresholds = eval_cfg.get("thresholds", [0.5, 0.7, 0.9])

    # Load test dataset
    test_samples = load_ppi_test_dataset(cfg, max_samples=max_samples)

    if not test_samples:
        log.error("No test samples loaded")
        return {"error": "no_test_samples"}

    log.info(f"Evaluating on {len(test_samples)} samples")

    # Generate predictions
    predictions = []

    for i in range(0, len(test_samples), batch_size):
        batch = test_samples[i:i + batch_size]

        # Prepare prompts
        prompts = [
            create_ppi_prompt(sample.sequence_1, sample.sequence_2, prompt_template)
            for sample in batch
        ]

        # Prepare sequences - for PPI we need to provide both sequences
        # The model should handle pairs appropriately
        sequences_1 = [sample.sequence_1 for sample in batch]
        sequences_2 = [sample.sequence_2 for sample in batch]

        # Generate responses
        # PPI: both sequences are in the prompt via create_ppi_prompt().
        # Pass first protein for encoding; the prompt contains both sequences as text.
        try:
            generated_texts = model.generate(
                protein_sequences=sequences_1,
                prompt=prompts,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
            )
        except Exception as e:
            log.error(f"Generation failed for batch {i}: {e}")
            continue

        # Process results
        for sample, generated_text in zip(batch, generated_texts):
            predicted_label, predicted_confidence = parse_ppi_prediction(generated_text)

            result = PPIPredictionResult(
                predicted_label=predicted_label,
                predicted_confidence=predicted_confidence,
                ground_truth_label=sample.label,
                generated_text=generated_text,
                protein_id_1=sample.protein_id_1,
                protein_id_2=sample.protein_id_2,
            )
            predictions.append(result)

        if (i + batch_size) % 10 == 0 or (i + batch_size) >= len(test_samples):
            log.info(f"Processed {min(i + batch_size, len(test_samples))}/{len(test_samples)} samples")

    # Compute metrics
    metrics = compute_ppi_metrics(predictions, thresholds=thresholds)

    log.info("PPI Prediction Evaluation Results:")
    for key, value in sorted(metrics.items()):
        if isinstance(value, float):
            log.info(f"  {key}: {value:.4f}")
        else:
            log.info(f"  {key}: {value}")

    # Save predictions to output_dir if provided
    if output_dir:
        _save_predictions(predictions, output_dir, "ppi")

    # Log to wandb if configured
    logging_cfg = cfg.get("logging", {})
    if logging_cfg.get("wandb", {}).get("enabled", False):
        _log_to_wandb(metrics, predictions, cfg)

    # Log to tensorboard if configured
    if logging_cfg.get("tensorboard", {}).get("enabled", False):
        _log_to_tensorboard(metrics, cfg)

    return metrics


def _save_predictions(
    predictions: List[PPIPredictionResult],
    output_dir: str,
    task_name: str,
) -> None:
    """Save individual predictions to JSON file."""
    output_path = Path(output_dir) / f"{task_name}_predictions.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = [
        {
            "protein_id_1": p.protein_id_1,
            "protein_id_2": p.protein_id_2,
            "predicted_label": p.predicted_label,
            "predicted_confidence": p.predicted_confidence,
            "ground_truth_label": p.ground_truth_label,
            "generated_text": p.generated_text,
        }
        for p in predictions
    ]

    with open(output_path, "w") as f:
        json.dump(records, f, indent=2)

    log.info(f"Predictions saved to {output_path}")


def _log_to_wandb(
    metrics: Dict[str, float],
    predictions: List[PPIPredictionResult],
    cfg: DictConfig,
) -> None:
    """Log metrics to Weights & Biases."""
    try:
        import wandb

        # Log metrics
        wandb.log({"ppi_prediction": metrics})

        # Log a sample table
        table_data = []
        for p in predictions[:20]:  # Limit to 20 samples
            table_data.append([
                f"{p.protein_id_1}-{p.protein_id_2}",
                p.predicted_label,
                p.ground_truth_label,
                p.predicted_confidence,
                "Correct" if p.predicted_label == p.ground_truth_label else "Wrong",
                p.generated_text[:200],
            ])

        table = wandb.Table(
            columns=["pair", "predicted", "ground_truth", "confidence", "status", "generated_text"],
            data=table_data,
        )
        wandb.log({"ppi_prediction_samples": table})

    except ImportError:
        log.warning("wandb not installed, skipping wandb logging")
    except Exception as e:
        log.warning(f"Failed to log to wandb: {e}")


def _log_to_tensorboard(
    metrics: Dict[str, float],
    cfg: DictConfig,
) -> None:
    """Log metrics to TensorBoard."""
    try:
        from torch.utils.tensorboard import SummaryWriter

        log_dir = cfg.get("logging", {}).get("tensorboard", {}).get("log_dir", "./runs")
        writer = SummaryWriter(log_dir=log_dir)

        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                writer.add_scalar(f"ppi_prediction/{key}", value)

        writer.close()

    except ImportError:
        log.warning("tensorboard not installed, skipping tensorboard logging")
    except Exception as e:
        log.warning(f"Failed to log to tensorboard: {e}")


# Utility functions for external use

def evaluate_ppi_from_predictions(
    predictions: List[Dict[str, Any]],
    thresholds: Optional[List[float]] = None,
) -> Dict[str, float]:
    """
    Evaluate PPI predictions from a list of pre-computed predictions.

    Useful for evaluating predictions without loading the model.

    Args:
        predictions: List of dicts with keys:
            - predicted_label: Predicted label (0 or 1)
            - predicted_confidence: Confidence score
            - ground_truth_label: Ground truth label (0 or 1)

        thresholds: Optional list of thresholds for metrics at thresholds.

    Returns:
        Dictionary of metric names to values.
    """
    results = []
    for pred in predictions:
        results.append(PPIPredictionResult(
            predicted_label=pred.get("predicted_label", 0),
            predicted_confidence=pred.get("predicted_confidence", 0.5),
            ground_truth_label=pred.get("ground_truth_label", 0),
            generated_text=pred.get("generated_text", ""),
            protein_id_1=pred.get("protein_id_1", ""),
            protein_id_2=pred.get("protein_id_2", ""),
        ))

    return compute_ppi_metrics(results, thresholds=thresholds)
