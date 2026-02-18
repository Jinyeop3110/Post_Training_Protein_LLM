"""
GO (Gene Ontology) Term Prediction Evaluation

This module evaluates the model's ability to predict GO terms for proteins.
GO terms are organized into three categories (namespaces):
- MF (Molecular Function): Activities at the molecular level
- BP (Biological Process): Larger cellular/organism processes
- CC (Cellular Component): Locations within the cell

Metrics computed:
- Accuracy: Exact match of predicted GO terms
- F1 Score (macro/micro): For multi-label classification
- AUPR: Area under precision-recall curve (if confidence scores available)
- Precision/Recall: Per-category (MF, BP, CC)
"""

import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from omegaconf import DictConfig, OmegaConf

try:
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_recall_curve,
        precision_score,
        recall_score,
        average_precision_score,
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

log = logging.getLogger(__name__)

# GO term pattern: GO:XXXXXXX (7 digits)
GO_TERM_PATTERN = re.compile(r"GO:\d{7}")

# GO namespace prefixes for categorization
GO_NAMESPACE_PREFIXES = {
    "MF": "GO:0003",  # Molecular Function typically starts with GO:0003xxx
    "BP": "GO:0008",  # Biological Process typically starts with GO:0008xxx
    "CC": "GO:0005",  # Cellular Component typically starts with GO:0005xxx
}


@dataclass
class GOTestSample:
    """A test sample for GO term prediction."""

    protein_id: str
    sequence: str
    go_terms: Set[str]
    go_terms_mf: Set[str] = field(default_factory=set)
    go_terms_bp: Set[str] = field(default_factory=set)
    go_terms_cc: Set[str] = field(default_factory=set)
    description: str = ""

    def __post_init__(self):
        """Categorize GO terms into namespaces if not provided."""
        if not self.go_terms_mf and not self.go_terms_bp and not self.go_terms_cc:
            self.go_terms_mf, self.go_terms_bp, self.go_terms_cc = categorize_go_terms(
                self.go_terms
            )


@dataclass
class GOPredictionResult:
    """Result of GO term prediction for a single sample."""

    protein_id: str
    predicted_terms: Set[str]
    ground_truth_terms: Set[str]
    generated_text: str
    predicted_mf: Set[str] = field(default_factory=set)
    predicted_bp: Set[str] = field(default_factory=set)
    predicted_cc: Set[str] = field(default_factory=set)
    ground_truth_mf: Set[str] = field(default_factory=set)
    ground_truth_bp: Set[str] = field(default_factory=set)
    ground_truth_cc: Set[str] = field(default_factory=set)


def parse_go_terms(text: str) -> List[str]:
    """
    Extract GO terms from generated text.

    GO terms follow the format GO:XXXXXXX where X is a digit.

    Args:
        text: Generated text that may contain GO terms.

    Returns:
        List of unique GO terms found in the text.

    Examples:
        >>> parse_go_terms("The protein has GO:0003674 and GO:0008150 functions.")
        ['GO:0003674', 'GO:0008150']
        >>> parse_go_terms("No GO terms here")
        []
        >>> parse_go_terms("GO:0005515, GO:0005515, GO:0016020")
        ['GO:0005515', 'GO:0016020']
    """
    if not text:
        return []

    # Find all GO terms
    matches = GO_TERM_PATTERN.findall(text)

    # Return unique terms while preserving order
    seen = set()
    unique_terms = []
    for term in matches:
        if term not in seen:
            seen.add(term)
            unique_terms.append(term)

    return unique_terms


def categorize_go_terms(
    go_terms: Union[List[str], Set[str]]
) -> Tuple[Set[str], Set[str], Set[str]]:
    """
    Categorize GO terms into MF, BP, and CC namespaces.

    Note: This is a simplified categorization based on GO term prefixes.
    For accurate categorization, use the full GO ontology.

    Args:
        go_terms: Collection of GO terms.

    Returns:
        Tuple of (MF terms, BP terms, CC terms).
    """
    mf_terms = set()
    bp_terms = set()
    cc_terms = set()

    for term in go_terms:
        # Simplified categorization based on common prefixes
        # In production, this should use the actual GO ontology
        if term.startswith("GO:0003") or term.startswith("GO:0004") or term.startswith("GO:0016"):
            mf_terms.add(term)
        elif term.startswith("GO:0006") or term.startswith("GO:0007") or term.startswith("GO:0008"):
            bp_terms.add(term)
        elif term.startswith("GO:0005") or term.startswith("GO:0009") or term.startswith("GO:0012"):
            cc_terms.add(term)
        else:
            # Default to MF for unknown prefixes
            mf_terms.add(term)

    return mf_terms, bp_terms, cc_terms


def compute_go_metrics(
    predictions: List[GOPredictionResult],
    include_per_category: bool = True,
) -> Dict[str, float]:
    """
    Compute evaluation metrics for GO term predictions.

    Args:
        predictions: List of prediction results.
        include_per_category: Whether to include per-category (MF/BP/CC) metrics.

    Returns:
        Dictionary of metric names to values.
    """
    if not SKLEARN_AVAILABLE:
        log.warning("scikit-learn not available, computing basic metrics only")
        return _compute_basic_metrics(predictions)

    if not predictions:
        return {"error": "no_predictions"}

    metrics = {}

    # Collect all unique GO terms
    all_terms = set()
    for pred in predictions:
        all_terms.update(pred.predicted_terms)
        all_terms.update(pred.ground_truth_terms)

    if not all_terms:
        return {"error": "no_go_terms_found"}

    # Convert to sorted list for consistent indexing
    term_list = sorted(all_terms)
    term_to_idx = {term: idx for idx, term in enumerate(term_list)}

    # Build binary matrices
    n_samples = len(predictions)
    n_terms = len(term_list)

    y_true = np.zeros((n_samples, n_terms), dtype=np.int32)
    y_pred = np.zeros((n_samples, n_terms), dtype=np.int32)

    for i, pred in enumerate(predictions):
        for term in pred.ground_truth_terms:
            if term in term_to_idx:
                y_true[i, term_to_idx[term]] = 1
        for term in pred.predicted_terms:
            if term in term_to_idx:
                y_pred[i, term_to_idx[term]] = 1

    # Compute overall metrics
    metrics["accuracy"] = _compute_exact_match_accuracy(predictions)
    metrics["f1_micro"] = f1_score(y_true, y_pred, average="micro", zero_division=0)
    metrics["f1_macro"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["f1_samples"] = f1_score(y_true, y_pred, average="samples", zero_division=0)
    metrics["precision_micro"] = precision_score(y_true, y_pred, average="micro", zero_division=0)
    metrics["recall_micro"] = recall_score(y_true, y_pred, average="micro", zero_division=0)
    metrics["precision_macro"] = precision_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["recall_macro"] = recall_score(y_true, y_pred, average="macro", zero_division=0)

    # Compute AUPR (Average Precision)
    # For multi-label, compute per-label and average
    try:
        # Only compute for labels that appear in ground truth
        label_mask = y_true.sum(axis=0) > 0
        if label_mask.any():
            aupr_values = []
            for j in range(n_terms):
                if label_mask[j]:
                    ap = average_precision_score(y_true[:, j], y_pred[:, j])
                    if not np.isnan(ap):
                        aupr_values.append(ap)
            if aupr_values:
                metrics["aupr"] = np.mean(aupr_values)
            else:
                metrics["aupr"] = 0.0
        else:
            metrics["aupr"] = 0.0
    except Exception as e:
        log.warning(f"Could not compute AUPR: {e}")
        metrics["aupr"] = 0.0

    # Per-category metrics
    if include_per_category:
        for category, get_terms in [
            ("mf", lambda p: (p.predicted_mf, p.ground_truth_mf)),
            ("bp", lambda p: (p.predicted_bp, p.ground_truth_bp)),
            ("cc", lambda p: (p.predicted_cc, p.ground_truth_cc)),
        ]:
            cat_metrics = _compute_category_metrics(predictions, get_terms)
            for key, value in cat_metrics.items():
                metrics[f"{category}_{key}"] = value

    # Additional statistics
    metrics["num_samples"] = n_samples
    metrics["num_unique_terms"] = n_terms
    metrics["avg_predicted_terms"] = np.mean([len(p.predicted_terms) for p in predictions])
    metrics["avg_ground_truth_terms"] = np.mean([len(p.ground_truth_terms) for p in predictions])

    return metrics


def _compute_basic_metrics(predictions: List[GOPredictionResult]) -> Dict[str, float]:
    """Compute basic metrics without sklearn."""
    if not predictions:
        return {"error": "no_predictions"}

    metrics = {}

    # Exact match accuracy
    metrics["accuracy"] = _compute_exact_match_accuracy(predictions)

    # Simple precision/recall
    total_true_positives = 0
    total_predicted = 0
    total_ground_truth = 0

    for pred in predictions:
        true_positives = len(pred.predicted_terms & pred.ground_truth_terms)
        total_true_positives += true_positives
        total_predicted += len(pred.predicted_terms)
        total_ground_truth += len(pred.ground_truth_terms)

    metrics["precision_micro"] = (
        total_true_positives / total_predicted if total_predicted > 0 else 0.0
    )
    metrics["recall_micro"] = (
        total_true_positives / total_ground_truth if total_ground_truth > 0 else 0.0
    )

    if metrics["precision_micro"] + metrics["recall_micro"] > 0:
        metrics["f1_micro"] = (
            2 * metrics["precision_micro"] * metrics["recall_micro"] /
            (metrics["precision_micro"] + metrics["recall_micro"])
        )
    else:
        metrics["f1_micro"] = 0.0

    metrics["num_samples"] = len(predictions)
    metrics["avg_predicted_terms"] = total_predicted / len(predictions)
    metrics["avg_ground_truth_terms"] = total_ground_truth / len(predictions)

    return metrics


def _compute_exact_match_accuracy(predictions: List[GOPredictionResult]) -> float:
    """Compute exact match accuracy (all terms must match)."""
    if not predictions:
        return 0.0

    exact_matches = sum(
        1 for p in predictions
        if p.predicted_terms == p.ground_truth_terms
    )
    return exact_matches / len(predictions)


def _compute_category_metrics(
    predictions: List[GOPredictionResult],
    get_terms_fn,
) -> Dict[str, float]:
    """Compute metrics for a specific GO category."""
    metrics = {}

    total_true_positives = 0
    total_predicted = 0
    total_ground_truth = 0

    for pred in predictions:
        predicted, ground_truth = get_terms_fn(pred)
        true_positives = len(predicted & ground_truth)
        total_true_positives += true_positives
        total_predicted += len(predicted)
        total_ground_truth += len(ground_truth)

    precision = total_true_positives / total_predicted if total_predicted > 0 else 0.0
    recall = total_true_positives / total_ground_truth if total_ground_truth > 0 else 0.0

    metrics["precision"] = precision
    metrics["recall"] = recall
    metrics["f1"] = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0 else 0.0
    )

    return metrics


def load_go_test_dataset(
    cfg: DictConfig,
    max_samples: Optional[int] = None,
) -> List[GOTestSample]:
    """
    Load GO term test dataset.

    This function supports multiple data sources:
    1. Pre-processed JSON file with GO annotations
    2. Swiss-Prot FASTA with GO terms in headers
    3. HuggingFace datasets with GO annotations

    Args:
        cfg: Configuration with dataset settings.
        max_samples: Maximum number of samples to load (for testing).

    Returns:
        List of test samples with GO annotations.
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
        elif data_format == "fasta":
            samples = _load_fasta_dataset(data_path, max_samples)
        else:
            log.warning(f"Unknown data format: {data_format}, using demo data")
            samples = _create_demo_dataset(max_samples or 10)
    else:
        # Use demo dataset for testing
        log.info("No dataset path provided, using demo dataset")
        samples = _create_demo_dataset(max_samples or 10)

    log.info(f"Loaded {len(samples)} GO test samples")
    return samples


def _load_json_dataset(path: str, max_samples: Optional[int]) -> List[GOTestSample]:
    """Load GO dataset from JSON file."""
    samples = []

    with open(path, "r") as f:
        data = json.load(f)

    # Handle different JSON formats
    if isinstance(data, list):
        entries = data
    elif isinstance(data, dict) and "proteins" in data:
        entries = data["proteins"]
    else:
        entries = [data]

    for entry in entries:
        if max_samples and len(samples) >= max_samples:
            break

        # Extract fields with fallbacks
        protein_id = entry.get("id", entry.get("protein_id", f"protein_{len(samples)}"))
        sequence = entry.get("sequence", entry.get("seq", ""))

        # Get GO terms (support multiple field names)
        go_terms_raw = entry.get("go_terms", entry.get("GO", entry.get("annotations", [])))
        if isinstance(go_terms_raw, str):
            go_terms = set(parse_go_terms(go_terms_raw))
        elif isinstance(go_terms_raw, list):
            go_terms = set(go_terms_raw)
        else:
            go_terms = set()

        if sequence and go_terms:
            samples.append(GOTestSample(
                protein_id=protein_id,
                sequence=sequence,
                go_terms=go_terms,
                description=entry.get("description", ""),
            ))

    return samples


def _load_fasta_dataset(path: str, max_samples: Optional[int]) -> List[GOTestSample]:
    """Load GO dataset from FASTA file with GO annotations in headers."""
    samples = []

    current_id = None
    current_seq = []
    current_go_terms = set()

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                # Save previous entry
                if current_id and current_seq and current_go_terms:
                    samples.append(GOTestSample(
                        protein_id=current_id,
                        sequence="".join(current_seq),
                        go_terms=current_go_terms,
                    ))
                    if max_samples and len(samples) >= max_samples:
                        break

                # Parse new header
                header = line[1:]
                parts = header.split("|")
                current_id = parts[1] if len(parts) > 1 else parts[0].split()[0]
                current_go_terms = set(parse_go_terms(header))
                current_seq = []
            else:
                current_seq.append(line)

        # Save last entry
        if current_id and current_seq and current_go_terms:
            if not max_samples or len(samples) < max_samples:
                samples.append(GOTestSample(
                    protein_id=current_id,
                    sequence="".join(current_seq),
                    go_terms=current_go_terms,
                ))

    return samples


def _create_demo_dataset(num_samples: int = 10) -> List[GOTestSample]:
    """Create a demo dataset for testing purposes."""
    # Representative protein sequences with known GO terms
    demo_data = [
        {
            "id": "P00533",  # EGFR
            "sequence": "MRPSGTAGAALLALLAALCPASRALEEKKVCQGTSNKLTQLGTFEDHFLSLQRMFNNCEVVLGNLEITYVQRNYDLSFLKTIQEVAGYVLIALNTVERIPLENLQIIRGNMYYENSYALAVLSNYDANKTGLKELPMRNLQEILHGAVRFSNNPALCNVESIQWRDIVSSDFLSNMSMDFQNHLGSCQKCDPSCPNGSCWGAGEENCQKLTKIICAQQCSGRCRGKSPSDCCHNQCAAGCTGPRESDCLVCRKFRDEATCKDTCPPLMLYNPTTYQMDVNPEGKYSFGATCVKKCPRNYVVTDHGSCVRACGADSYEMEEDGVRKCKKCEGPCRKVCNGIGIGEFKDSLSINATNIKHFKNCTSISGDLHILPVAFRGDSFTHTPPLDPQELDILKTVKEITGFLLIQAWPENRTDLHAFENLEIIRGRTKQHGQFSLAVVSLNITSLGLRSLKEISDGDVIISGNKNLCYANTINWKKLFGTSGQKTKIISNRGENSCKATGQVCHALCSPEGCWGPEPRDCVSCRNVSRGRECVDKCNLLEGEPREFVENSECIQCHPECLPQAMNITCTGRGPDNCIQCAHYIDGPHCVKTCPAGVMGENNTLVWKYADAGHVCHLCHPNCTYGCTGPGLEGCPTNGPKIPSIATGMVGALLLLLVVALGIGLFMRRRHIVRKRTLRRLLQERELVEPLTPSGEAPNQALLRILKETEFKKIKVLGSGAFGTVYKGLWIPEGEKVKIPVAIKELREATSPKANKEILDEAYVMASVDNPHVCRLLGICLTSTVQLITQLMPFGCLLDYVREHKDNIGSQYLLNWCVQIAKGMNYLEDRRLVHRDLAARNVLVKTPQHVKITDFGLAKLLGAEEKEYHAEGGKVPIKWMALESILHRIYTHQSDVWSYGVTVWELMTFGSKPYDGIPASEISSILEKGERLPQPPICTIDVYMIMVKCWMIDADSRPKFRELIIEFSKMARDPQRYLVIQGDERMHLPSPTDSNFYRALMDEEDMDDVVDADEYLIPQQGFFSSPSTSRTPLLSSLSATSNNSTVACIDRNGLQSCPIKEDSFLQRYSSDPTGALTEDSIDDTFLPVPEYINQSVPKRPAGSVQNPVYHNQPLNPAPSRDPHYQDPHSTAVGNPEYLNTVQPTCVNSTFDSPAHWAQKGSHQISLDNPDYQQDFFPKEAKPNGIFKGSTAENAEYLRVAPQSSEFIGA",
            "go_terms": {"GO:0004713", "GO:0005524", "GO:0016740", "GO:0046872", "GO:0005886"},
        },
        {
            "id": "P04637",  # TP53
            "sequence": "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHERCSDSDGLAPPQHLIRVEGNLRVEYLDDRNTFRHSVVVPYEPPEVGSDCTTIHYNYMCNSSCMGGMNRRPILTIITLEDSSGNLLGRNSFEVRVCACPGRDRRTEEENLRKKGEPHHELPPGSTKRALPNNTSSSPQPKKKPLDGEYFTLQIRGRERFEMFRELNEALELKDAQAGKEPGGSRAHSSHLKSKKGQSTSRHKKLMFKTEGPDSD",
            "go_terms": {"GO:0003677", "GO:0006355", "GO:0005634", "GO:0006915", "GO:0007049"},
        },
        {
            "id": "P01308",  # Insulin
            "sequence": "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN",
            "go_terms": {"GO:0005179", "GO:0008286", "GO:0005576", "GO:0042593", "GO:0010906"},
        },
        {
            "id": "P68871",  # Hemoglobin beta
            "sequence": "MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLSTPDAVMGNPKVKAHGKKVLGAFSDGLAHLDNLKGTFATLSELHCDKLHVDPENFRLLGNVLVCVLAHHFGKEFTPPVQAAYQKVVAGVANALAHKYH",
            "go_terms": {"GO:0005344", "GO:0019825", "GO:0005833", "GO:0015671", "GO:0005506"},
        },
        {
            "id": "P02768",  # Albumin
            "sequence": "MKWVTFISLLFLFSSAYSRGVFRRDAHKSEVAHRFKDLGEENFKALVLIAFAQYLQQCPFEDHVKLVNEVTEFAKTCVADESAENCDKSLHTLFGDKLCTVATLRETYGEMADCCAKQEPERNECFLQHKDDNPNLPRLVRPEVDVMCTAFHDNEETFLKKYLYEIARRHPYFYAPELLFFAKRYKAAFTECCQAADKAACLLPKLDELRDEGKASSAKQRLKCASLQKFGERAFKAWAVARLSQRFPKAEFAEVSKLVTDLTKVHTECCHGDLLECADDRADLAKYICENQDSISSKLKECCEKPLLEKSHCIAEVENDEMPADLPSLAADFVESKDVCKNYAEAKDVFLGMFLYEYARRHPDYSVVLLLRLAKTYETTLEKCCAAADPHECYAKVFDEFKPLVEEPQNLIKQNCELFEQLGEYKFQNALLVRYTKKVPQVSTPTLVEVSRNLGKVGSKCCKHPEAKRMPCAEDYLSVVLNQLCVLHEKTPVSDRVTKCCTESLVNRRPCFSALEVDETYVPKEFNAETFTFHADICTLSEKERQIKKQTALVELVKHKPKATKEQLKAVMDDFAAFVEKCCKADDKETCFAEEGKKLVAASQAALGL",
            "go_terms": {"GO:0005215", "GO:0008289", "GO:0005576", "GO:0019825", "GO:0005615"},
        },
        {
            "id": "P01375",  # TNF-alpha
            "sequence": "MSTESMIRDVELAEEALPKKTGGPQGSRRCLFLSLFSFLIVAGATTLFCLLHFGVIGPQREEFPRDLSLISPLAQAVRSSSRTPSDKPVAHVVANPQAEGQLQWLNRRANALLANGVELRDNQLVVPSEGLYLIYSQVLFKGQGCPSTHVLLTHTISRIAVSYQTKVNLLSAIKSPCQRETPEGAEAKPWYEPIYLGGVFQLEKGDRLSAEINRPDYLDFAESGQVYFGIIAL",
            "go_terms": {"GO:0005164", "GO:0006955", "GO:0005615", "GO:0006915", "GO:0007165"},
        },
        {
            "id": "P21802",  # FGFR2
            "sequence": "MVSWGRFICLVVVTMATLSLARPSFSLVEDTTLEPEEPPTKYQISQPEVYVAAPGESLEVRCLLKDAAVISWTKDGVHLGPNNRTVLIGEYLQIKGATPRDSGLYACTASRTVDSETWYFMVNVTDAISSGDDEDDTDGAEDFVSENSNNKRAPYWTNTEKMEKRLHAVPAANTVKFRCPAGGNPMPTMRWLKNGKEFKQEHRIGGYKVRNQHWSLIMESVVPSDKGNYTCVVENEYGSINHTYHLDVVERSPHRPILQAGLPANASTVVGGDVEFVCKVYSDAQPHIQWIKHVEKNGSKYGPDGLPYLKVLKAAGVNTTDKEIEVLYIRNVTFEDAGEYTCLAGNSIGFSHHSAWLVVLPAEEELVEADEAGSVYAGILSYGVGFFLFILVVAAVTLCRLRSPPKKGLGSPTVHKISRFPLKRQVSLESNASMSSNTPLVRITTRLSSTADTPMLAGVSEYELPEDPRWELPRDRLVLGKPLGEGCFGQVVMAEAVGIDKDKPKEAVTVAVKMLKDDATEKDLSDLVSEMEMMKMIGKHKNIINLLGACTQDGPLYVIVEYASKGNLREYLRARRPPGMEYSYDINRVPEEQMTFKDLVSCTYQLARGMEYLASQKCIHRDLAARNVLVTENNVMKIADFGLARDINNIDYYKKTTNGRLPVKWMAPEALFDRVYTHQSDVWSFGVLLWEIFTLGGSPYPGIPVEELFKLLKEGHRMDKPANCTNELYMMMRDCWHAVPSQRPTFKQLVEDLDRILTLTTNEEYLDLSQPLEQYSPSYPDTRSSCSSGDDSVFSPDPMPYEPCLPQYPHINGSVKT",
            "go_terms": {"GO:0004713", "GO:0005524", "GO:0016740", "GO:0005886", "GO:0007169"},
        },
        {
            "id": "P06401",  # Progesterone receptor
            "sequence": "MTELKAKGPRAPHVAGGPPSPEVGSPLLCRPAAGPFPGSQTSDTLPEVSAIPISLDGLLFPRPCQGQDPSEDSETTGKAGVKRSQASRFKQQKKGQHMQMVQPPSHHVMQKSAQQQPPSKGHPDQFSLPQGHCQMPPGLPQPRAPAGPMPQQHLSTQVKCPEDTSLGTLNRRSPSPTSSSTSSSTTGPPPQHLSTSYGQLLTPQFQQSSQPQTLCSPGSQTPHSSSPQSSSFAHNLQQFQQIPQCSPSSASSVTSSGNFYNHFSGQQQGFHHNQQHQQQYSPCTSPYSSGFQANVQGFYQQFHQQGQQNQVNHHGFATFSQSPAQFSQQNSPQHQQQQQSQQQHQQASHHPQQHQQQQLQIFNFPPQPPAAPSHIRQEQHSSVPPELSLTASENSGGKPAAVTCQPFPRPSQAMGVGSLHPSVHSQNMQPMGGHNMMPSTVNVFPIGPLPMGPGGTPQVNPGVPGVMGMHSPLGPSGMPQCGQVQGVPLNLSLYVQPSITTPGSQQGLSNCRPAQQQSSQPQHLPVQGPQVLSPHQQSQDSLGYAGQHHFQQPQQQQQQYRSRLEATRQSGSLSFEFSSYAGSTGGIQGGGMPLHNHQQQQGQQYHGAHFTSPQSSNSQQGMGQQQQHMPMYPGQAMQTGFGGPLPPAPPPQPHVQQQQQQQQQQQQPQQYQGISQQQQQQPQQQQHHQSGQQPQQQVIQQQQQQQQQQQSHGQQQQAAGQGQHGQQQQQQQGQPLGMVPQQQQQQQQQQQETLQQQQQQRQHHLRRQQGEQQGYGIQQQQQQQQQHIQGQQPPAQQAQGQQPQVQQQQQQHLGMGKDSMSQHQQAQTQGHMQQQQQDGQPSHQDFSRQHSSMQHQQQTHQGHQQSHHHHHHHHLRQSHLVPRGPQTPTQSSSPPLHMQPIIHIHHHQTSPPHFGQHQGAQQPQQQQQPASQQISQSLQQQQQPQQQQQQSQQWQRQSQQQAPQMVQQSQHHAAAMGQQQQQQQQQQQQQQQQQQQQQQQQQQQQ",
            "go_terms": {"GO:0003677", "GO:0003707", "GO:0005634", "GO:0006355", "GO:0008270"},
        },
        {
            "id": "P00441",  # SOD1
            "sequence": "MATKAVCVLKGDGPVQGIINFEQKESNGPVKVWGSIKGLTEGLHGFHVHEFGDNTAGCTSAGPHFNPLSRKHGGPKDEERHVGDLGNVTADKDGVADVSIEDSVISLSGDHCIIGRTLVVHEKADDLGKGGNEESTKTGNAGSRLACGVIGIAQ",
            "go_terms": {"GO:0004784", "GO:0046872", "GO:0005737", "GO:0006801", "GO:0019430"},
        },
        {
            "id": "P01889",  # HLA-B
            "sequence": "MLVMAPRTVLLLLSAALALTETWAGSHSMRYFYTSVSRPGRGEPRFISVGYVDDTQFVRFDSDAASPREEPRAPWIEQEGPEYWDRNTQIFKTNTQTYRESLRNLRGYYNQSEAGSHTLQSMYGCDVGPDGRLLRGHDQYAYDGKDYIALNEDLRSWTAADTAAQITQRKWEAARVAEQLRAYLEGLCVEWLRRYLENGKETLQRADPPKTHVTHHPVSDHEATLRCWALGFYPAEITLTWQRDGEDQTQDTELVETRPAGDRTFQKWAAVVVPSGEEQRYTCHVQHEGLPKPLTLRWELSSQPTIPIVGIIAGLVLLGAVITGAVVAAVMWRRKSSDRKGGSYTQAASSDSAQGSDVSLTACKV",
            "go_terms": {"GO:0042612", "GO:0005886", "GO:0019882", "GO:0002474", "GO:0005615"},
        },
    ]

    samples = []
    for entry in demo_data[:num_samples]:
        samples.append(GOTestSample(
            protein_id=entry["id"],
            sequence=entry["sequence"],
            go_terms=set(entry["go_terms"]),
        ))

    return samples


def create_go_prompt(
    sequence: str,
    prompt_template: Optional[str] = None,
) -> str:
    """
    Create a prompt for GO term prediction.

    Args:
        sequence: Protein sequence.
        prompt_template: Optional custom prompt template with {sequence} placeholder.

    Returns:
        Formatted prompt string.
    """
    if prompt_template:
        return prompt_template.format(sequence=sequence)

    # Default prompt template
    return f"""Protein sequence: {sequence}

What are the Gene Ontology (GO) terms for this protein?
List the molecular functions, biological processes, and cellular components.
Format each GO term as GO:XXXXXXX (e.g., GO:0003674).
"""


def evaluate_go(
    cfg: DictConfig,
    checkpoint_path: Optional[str] = None,
) -> Dict[str, float]:
    """
    Evaluate GO term prediction.

    This function:
    1. Loads the model from checkpoint
    2. Loads the GO term test dataset
    3. Generates predictions for each protein
    4. Parses GO terms from generated text
    5. Computes evaluation metrics

    Args:
        cfg: Hydra configuration containing:
            - model: Model configuration
            - dataset: Dataset configuration (path, format, max_samples)
            - evaluation: Evaluation settings (batch_size, max_new_tokens)
            - logging: Logging settings (wandb, tensorboard, save_results)

        checkpoint_path: Path to model checkpoint.

    Returns:
        Dictionary of metric names to values.
    """
    log.info("Evaluating GO term prediction...")

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

    # Load test dataset
    test_samples = load_go_test_dataset(cfg, max_samples=max_samples)

    if not test_samples:
        log.error("No test samples loaded")
        return {"error": "no_test_samples"}

    log.info(f"Evaluating on {len(test_samples)} samples")

    # Generate predictions
    predictions = []

    for i in range(0, len(test_samples), batch_size):
        batch = test_samples[i:i + batch_size]

        # Prepare prompts
        sequences = [sample.sequence for sample in batch]
        prompts = [create_go_prompt(seq, prompt_template) for seq in sequences]

        # Generate responses
        try:
            generated_texts = model.generate(
                protein_sequences=sequences,
                prompt=prompts,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Deterministic for evaluation
                temperature=1.0,
            )
        except Exception as e:
            log.error(f"Generation failed for batch {i}: {e}")
            continue

        # Process results
        for sample, generated_text in zip(batch, generated_texts):
            predicted_terms = set(parse_go_terms(generated_text))
            predicted_mf, predicted_bp, predicted_cc = categorize_go_terms(predicted_terms)

            result = GOPredictionResult(
                protein_id=sample.protein_id,
                predicted_terms=predicted_terms,
                ground_truth_terms=sample.go_terms,
                generated_text=generated_text,
                predicted_mf=predicted_mf,
                predicted_bp=predicted_bp,
                predicted_cc=predicted_cc,
                ground_truth_mf=sample.go_terms_mf,
                ground_truth_bp=sample.go_terms_bp,
                ground_truth_cc=sample.go_terms_cc,
            )
            predictions.append(result)

        if (i + batch_size) % 10 == 0 or (i + batch_size) >= len(test_samples):
            log.info(f"Processed {min(i + batch_size, len(test_samples))}/{len(test_samples)} samples")

    # Compute metrics
    metrics = compute_go_metrics(predictions)

    log.info("GO Prediction Evaluation Results:")
    for key, value in sorted(metrics.items()):
        if isinstance(value, float):
            log.info(f"  {key}: {value:.4f}")
        else:
            log.info(f"  {key}: {value}")

    # Save results if configured
    logging_cfg = cfg.get("logging", {})
    if logging_cfg.get("save_results", False):
        _save_results(predictions, metrics, cfg)

    # Log to wandb if configured
    if logging_cfg.get("wandb", {}).get("enabled", False):
        _log_to_wandb(metrics, predictions, cfg)

    # Log to tensorboard if configured
    if logging_cfg.get("tensorboard", {}).get("enabled", False):
        _log_to_tensorboard(metrics, cfg)

    return metrics


def _save_results(
    predictions: List[GOPredictionResult],
    metrics: Dict[str, float],
    cfg: DictConfig,
) -> None:
    """Save evaluation results to JSON file."""
    output_dir = cfg.get("logging", {}).get("output_dir", "./outputs")
    output_path = Path(output_dir) / "go_prediction_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = {
        "metrics": metrics,
        "predictions": [
            {
                "protein_id": p.protein_id,
                "predicted_terms": list(p.predicted_terms),
                "ground_truth_terms": list(p.ground_truth_terms),
                "generated_text": p.generated_text,
            }
            for p in predictions
        ],
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    log.info(f"Results saved to {output_path}")


def _log_to_wandb(
    metrics: Dict[str, float],
    predictions: List[GOPredictionResult],
    cfg: DictConfig,
) -> None:
    """Log metrics to Weights & Biases."""
    try:
        import wandb

        # Log metrics
        wandb.log({"go_prediction": metrics})

        # Log a sample table
        table_data = []
        for p in predictions[:20]:  # Limit to 20 samples
            table_data.append([
                p.protein_id,
                ", ".join(sorted(p.predicted_terms)),
                ", ".join(sorted(p.ground_truth_terms)),
                len(p.predicted_terms & p.ground_truth_terms),
                p.generated_text[:200],
            ])

        table = wandb.Table(
            columns=["protein_id", "predicted", "ground_truth", "correct", "generated_text"],
            data=table_data,
        )
        wandb.log({"go_prediction_samples": table})

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
                writer.add_scalar(f"go_prediction/{key}", value)

        writer.close()

    except ImportError:
        log.warning("tensorboard not installed, skipping tensorboard logging")
    except Exception as e:
        log.warning(f"Failed to log to tensorboard: {e}")


# Utility functions for external use

def evaluate_go_from_predictions(
    predictions: List[Dict[str, Any]],
) -> Dict[str, float]:
    """
    Evaluate GO predictions from a list of pre-computed predictions.

    Useful for evaluating predictions without loading the model.

    Args:
        predictions: List of dicts with keys:
            - protein_id: Protein identifier
            - predicted_terms: List of predicted GO terms
            - ground_truth_terms: List of ground truth GO terms

    Returns:
        Dictionary of metric names to values.
    """
    results = []
    for pred in predictions:
        predicted = set(pred.get("predicted_terms", []))
        ground_truth = set(pred.get("ground_truth_terms", []))

        predicted_mf, predicted_bp, predicted_cc = categorize_go_terms(predicted)
        gt_mf, gt_bp, gt_cc = categorize_go_terms(ground_truth)

        results.append(GOPredictionResult(
            protein_id=pred.get("protein_id", "unknown"),
            predicted_terms=predicted,
            ground_truth_terms=ground_truth,
            generated_text=pred.get("generated_text", ""),
            predicted_mf=predicted_mf,
            predicted_bp=predicted_bp,
            predicted_cc=predicted_cc,
            ground_truth_mf=gt_mf,
            ground_truth_bp=gt_bp,
            ground_truth_cc=gt_cc,
        ))

    return compute_go_metrics(results)
