"""Shared evaluation utilities."""


def resolve_placeholder(cfg) -> str:
    """Determine the protein placeholder token from approach config.

    Args:
        cfg: Hydra config (or dict-like) with an ``approach`` key.

    Returns:
        Placeholder string for ESM-3/flamingo approaches, or empty string
        for the text approach (which embeds raw AA sequences directly).
    """
    approach = cfg.get("approach", "text") if hasattr(cfg, "get") else getattr(cfg, "approach", "text")
    if approach == "flamingo":
        from src.models.multimodal_llm import PROTEIN_END_TOKEN, PROTEIN_START_TOKEN
        return f"{PROTEIN_START_TOKEN}{PROTEIN_END_TOKEN}"
    elif approach in ("esm3",):
        from src.models.multimodal_llm import PROTEIN_PLACEHOLDER
        return PROTEIN_PLACEHOLDER
    return ""
