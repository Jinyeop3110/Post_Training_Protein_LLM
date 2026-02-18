# Test Writer Agent

Generate tests for the protein-LLM project.

## Test Standards

### Framework
- Use pytest
- Tests in `tests/` mirror `src/` structure
- Use fixtures for common setup

### Naming
- Test files: `test_<module>.py`
- Test functions: `test_<function>_<scenario>`

### Coverage Requirements
- All public functions tested
- Edge cases: empty sequences, max length, invalid inputs
- Integration tests for dataloaders

## Test Templates

### Unit Test
```python
import pytest
from src.models.protein_encoder import ESM2Encoder

class TestESM2Encoder:
    @pytest.fixture
    def encoder(self):
        return ESM2Encoder(model_name="esm2_t33_650M_UR50D")

    def test_encode_single_sequence(self, encoder):
        seq = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
        embedding = encoder.encode(seq)
        assert embedding.shape[-1] == 1280  # ESM-2 650M dim

    def test_encode_empty_raises(self, encoder):
        with pytest.raises(ValueError):
            encoder.encode("")
```

### Dataloader Test
```python
def test_pdb_dataloader_batch_structure():
    from src.data import get_pdb_dataloader
    dl = get_pdb_dataloader("./data/pdb_2021aug02_sample", batch_size=2)
    batch = next(iter(dl))

    assert "sequence" in batch
    assert "coords" in batch
    assert len(batch["sequence"]) == 2
```

## Generated Tests Location
`tests/` directory, mirroring `src/` structure
