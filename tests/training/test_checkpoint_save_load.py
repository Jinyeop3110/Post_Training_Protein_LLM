"""Tests for ProteinLLMTrainer checkpoint save/load logic.

Covers:
A. Multimodal weights save/load roundtrip (pooling.pt, projector.pt)
B. Multimodal optimizer state save/load (mm_optimizer.pt)
C. _save_checkpoint creates pooling.pt and projector.pt
D. _load_from_checkpoint loads multimodal weights
E. Efficient save path does NOT create pytorch_model_fsdp.bin
"""

import os
from unittest.mock import MagicMock, PropertyMock, patch

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Helpers — lightweight stand-ins for pooling/projector modules
# ---------------------------------------------------------------------------

class _FakePooling(nn.Module):
    """Minimal attention-pooling-like module."""

    def __init__(self, in_dim: int = 1536, n_queries: int = 32):
        super().__init__()
        self.query = nn.Parameter(torch.randn(n_queries, in_dim))
        self.linear = nn.Linear(in_dim, in_dim)

    def forward(self, x):
        return self.linear(x)


class _FakeProjector(nn.Module):
    """Minimal MLP projector."""

    def __init__(self, in_dim: int = 1536, hidden: int = 5120, out_dim: int = 2560):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class _FakeProteinLLM:
    """Lightweight stand-in for the real ProteinLLM (no actual LLM needed)."""

    def __init__(self, pooling=None, projector=None):
        self.pooling = pooling
        self.projector = projector
        self.gated_xattn_blocks = None
        self.llm = None


# =========================================================================
# A. Multimodal weights save/load roundtrip
# =========================================================================


class TestMultimodalWeightsRoundtrip:
    """Save pooling/projector state_dicts to .pt files and reload them."""

    def test_pooling_save_load(self, tmp_path):
        """Pooling state_dict survives a save/load cycle."""
        pooling = _FakePooling()
        save_path = tmp_path / "pooling.pt"

        # Save
        torch.save(pooling.state_dict(), save_path)

        # Reload into a fresh module
        pooling2 = _FakePooling()
        pooling2.load_state_dict(
            torch.load(save_path, map_location="cpu", weights_only=True)
        )

        for k in pooling.state_dict():
            assert torch.equal(pooling.state_dict()[k], pooling2.state_dict()[k]), (
                f"Mismatch in pooling key {k}"
            )

    def test_projector_save_load(self, tmp_path):
        """Projector state_dict survives a save/load cycle."""
        projector = _FakeProjector()
        save_path = tmp_path / "projector.pt"

        torch.save(projector.state_dict(), save_path)

        projector2 = _FakeProjector()
        projector2.load_state_dict(
            torch.load(save_path, map_location="cpu", weights_only=True)
        )

        for k in projector.state_dict():
            assert torch.equal(
                projector.state_dict()[k], projector2.state_dict()[k]
            ), f"Mismatch in projector key {k}"

    def test_load_into_different_instance(self, tmp_path):
        """Loading produces weights identical to originals, not the fresh init."""
        projector = _FakeProjector()
        projector2 = _FakeProjector()

        # They should differ before loading (random init)
        any_diff = False
        for k in projector.state_dict():
            if not torch.equal(projector.state_dict()[k], projector2.state_dict()[k]):
                any_diff = True
                break
        assert any_diff, "Two randomly-initialized modules should differ"

        torch.save(projector.state_dict(), tmp_path / "projector.pt")
        projector2.load_state_dict(
            torch.load(tmp_path / "projector.pt", map_location="cpu", weights_only=True)
        )

        for k in projector.state_dict():
            assert torch.equal(
                projector.state_dict()[k], projector2.state_dict()[k]
            )


# =========================================================================
# B. Multimodal optimizer state save/load (mm_optimizer.pt format)
# =========================================================================


class TestMultimodalOptimizerState:
    """Verify the mm_optimizer.pt format expected by the trainer."""

    def test_mm_optimizer_format(self, tmp_path):
        """mm_optimizer.pt must have 'config' and 'states' keys."""
        projector = _FakeProjector()
        optimizer = torch.optim.AdamW(projector.parameters(), lr=1e-3)

        # Run a fake backward so optimizer state is populated
        loss = projector(torch.randn(2, 1536)).sum()
        loss.backward()
        optimizer.step()

        # Simulate what _save_optimizer_and_scheduler does for mm_optimizer
        mm_group = optimizer.param_groups[0]
        mm_save = {
            "config": {k: v for k, v in mm_group.items() if k != "params"},
            "states": [optimizer.state[p] for p in mm_group["params"]],
        }
        save_path = tmp_path / "mm_optimizer.pt"
        torch.save(mm_save, save_path)

        # Reload and verify structure
        loaded = torch.load(save_path, map_location="cpu", weights_only=True)
        assert "config" in loaded
        assert "states" in loaded
        assert isinstance(loaded["config"], dict)
        assert "lr" in loaded["config"]
        assert isinstance(loaded["states"], list)
        assert len(loaded["states"]) == len(list(projector.parameters()))

    def test_mm_optimizer_states_restore(self, tmp_path):
        """Optimizer step/exp_avg/exp_avg_sq survive save/load."""
        projector = _FakeProjector()
        optimizer = torch.optim.AdamW(projector.parameters(), lr=1e-3)

        loss = projector(torch.randn(2, 1536)).sum()
        loss.backward()
        optimizer.step()

        # Capture original states
        original_states = {}
        for p in optimizer.param_groups[0]["params"]:
            original_states[id(p)] = {
                k: v.clone() if isinstance(v, torch.Tensor) else v
                for k, v in optimizer.state[p].items()
            }

        # Save in the trainer's format
        mm_group = optimizer.param_groups[0]
        mm_save = {
            "config": {k: v for k, v in mm_group.items() if k != "params"},
            "states": [optimizer.state[p] for p in mm_group["params"]],
        }
        torch.save(mm_save, tmp_path / "mm_optimizer.pt")

        # Simulate load: create fresh optimizer, load states back
        projector2 = _FakeProjector()
        projector2.load_state_dict(projector.state_dict())
        optimizer2 = torch.optim.AdamW(projector2.parameters(), lr=1e-3)

        loaded = torch.load(
            tmp_path / "mm_optimizer.pt", map_location="cpu", weights_only=True
        )

        # Apply loaded state (mirrors _load_optimizer_and_scheduler logic)
        for p, state_data in zip(
            optimizer2.param_groups[0]["params"], loaded["states"]
        ):
            optimizer2.state[p] = {
                k: v.to(p.device) if isinstance(v, torch.Tensor) else v
                for k, v in state_data.items()
            }

        # Verify step, exp_avg, exp_avg_sq match
        for p_orig, p_new in zip(
            optimizer.param_groups[0]["params"],
            optimizer2.param_groups[0]["params"],
        ):
            state_orig = optimizer.state[p_orig]
            state_new = optimizer2.state[p_new]
            assert state_orig["step"] == state_new["step"]
            assert torch.equal(state_orig["exp_avg"], state_new["exp_avg"])
            assert torch.equal(state_orig["exp_avg_sq"], state_new["exp_avg_sq"])


# =========================================================================
# C. _save_checkpoint saves pooling.pt and projector.pt
# =========================================================================


class TestSaveCheckpoint:
    """Test that _save_checkpoint creates the expected .pt files."""

    def _make_trainer(self, output_dir, pooling=None, projector=None):
        """Build a minimally-mocked ProteinLLMTrainer for save_checkpoint tests."""
        from src.training.sft_trainer import ProteinLLMTrainer

        protein_llm = _FakeProteinLLM(pooling=pooling, projector=projector)

        # Mock the base Trainer pieces that _save_checkpoint / __init__ needs
        mock_args = MagicMock()
        mock_args.should_save = True
        mock_args.output_dir = str(output_dir)
        mock_args.local_rank = -1
        mock_args.world_size = 1
        mock_args.n_gpu = 0
        mock_args.device = torch.device("cpu")
        mock_args.should_log = True
        mock_args.report_to = []
        mock_args.logging_dir = str(output_dir / "logs")
        mock_args.learning_rate = 1e-4

        mock_state = MagicMock()
        mock_state.global_step = 100

        # We need to construct the trainer without calling HF Trainer.__init__
        # because that requires valid TrainingArguments, model, etc.
        # Instead, we create the instance manually and set the required attributes.
        trainer = object.__new__(ProteinLLMTrainer)
        trainer.protein_llm = protein_llm
        trainer.args = mock_args
        trainer.state = mock_state

        return trainer

    def test_saves_pooling_and_projector(self, tmp_path):
        """_save_checkpoint must create pooling.pt and projector.pt."""
        pooling = _FakePooling()
        projector = _FakeProjector()
        trainer = self._make_trainer(tmp_path, pooling=pooling, projector=projector)

        # Create the checkpoint directory that super()._save_checkpoint would
        ckpt_dir = tmp_path / "checkpoint-100"
        ckpt_dir.mkdir(parents=True)

        # Patch super()._save_checkpoint to be a no-op (we only test our additions)
        with patch.object(
            type(trainer).__bases__[0], "_save_checkpoint", return_value=None
        ):
            trainer._save_checkpoint(model=MagicMock(), trial=None)

        assert (ckpt_dir / "pooling.pt").exists(), "pooling.pt not created"
        assert (ckpt_dir / "projector.pt").exists(), "projector.pt not created"

    def test_saves_only_projector_when_no_pooling(self, tmp_path):
        """When pooling is None (perceiver), only projector.pt is created."""
        projector = _FakeProjector()
        trainer = self._make_trainer(tmp_path, pooling=None, projector=projector)

        ckpt_dir = tmp_path / "checkpoint-100"
        ckpt_dir.mkdir(parents=True)

        with patch.object(
            type(trainer).__bases__[0], "_save_checkpoint", return_value=None
        ):
            trainer._save_checkpoint(model=MagicMock(), trial=None)

        assert not (ckpt_dir / "pooling.pt").exists()
        assert (ckpt_dir / "projector.pt").exists()

    def test_no_save_when_not_should_save(self, tmp_path):
        """When args.should_save is False, no .pt files are created."""
        pooling = _FakePooling()
        projector = _FakeProjector()
        trainer = self._make_trainer(tmp_path, pooling=pooling, projector=projector)
        trainer.args.should_save = False

        ckpt_dir = tmp_path / "checkpoint-100"
        ckpt_dir.mkdir(parents=True)

        with patch.object(
            type(trainer).__bases__[0], "_save_checkpoint", return_value=None
        ):
            trainer._save_checkpoint(model=MagicMock(), trial=None)

        assert not (ckpt_dir / "pooling.pt").exists()
        assert not (ckpt_dir / "projector.pt").exists()

    def test_saved_weights_match_originals(self, tmp_path):
        """The .pt files must contain the exact state_dicts from the modules."""
        pooling = _FakePooling()
        projector = _FakeProjector()
        trainer = self._make_trainer(tmp_path, pooling=pooling, projector=projector)

        ckpt_dir = tmp_path / "checkpoint-100"
        ckpt_dir.mkdir(parents=True)

        with patch.object(
            type(trainer).__bases__[0], "_save_checkpoint", return_value=None
        ):
            trainer._save_checkpoint(model=MagicMock(), trial=None)

        loaded_pooling = torch.load(
            ckpt_dir / "pooling.pt", map_location="cpu", weights_only=True
        )
        loaded_projector = torch.load(
            ckpt_dir / "projector.pt", map_location="cpu", weights_only=True
        )

        for k, v in pooling.state_dict().items():
            assert torch.equal(v, loaded_pooling[k]), f"pooling mismatch: {k}"
        for k, v in projector.state_dict().items():
            assert torch.equal(v, loaded_projector[k]), f"projector mismatch: {k}"


# =========================================================================
# D. _load_from_checkpoint loads multimodal weights
# =========================================================================


class TestLoadFromCheckpoint:
    """Test that _load_from_checkpoint restores pooling/projector from .pt files."""

    def _make_trainer_for_load(self, pooling=None, projector=None):
        """Build a minimally-mocked trainer for _load_from_checkpoint tests."""
        from src.training.sft_trainer import ProteinLLMTrainer

        protein_llm = _FakeProteinLLM(pooling=pooling, projector=projector)

        mock_args = MagicMock()
        mock_args.should_save = True
        mock_args.local_rank = -1
        mock_args.world_size = 1
        mock_args.n_gpu = 0
        mock_args.device = torch.device("cpu")
        mock_args.learning_rate = 1e-4

        mock_model = MagicMock()

        trainer = object.__new__(ProteinLLMTrainer)
        trainer.protein_llm = protein_llm
        trainer.args = mock_args
        trainer.model = mock_model

        return trainer

    def test_loads_pooling_and_projector(self, tmp_path):
        """When pooling.pt and projector.pt exist, they are loaded into protein_llm."""
        # Save "source" weights
        src_pooling = _FakePooling()
        src_projector = _FakeProjector()
        torch.save(src_pooling.state_dict(), tmp_path / "pooling.pt")
        torch.save(src_projector.state_dict(), tmp_path / "projector.pt")

        # Create trainer with freshly initialized (different) modules
        dst_pooling = _FakePooling()
        dst_projector = _FakeProjector()
        trainer = self._make_trainer_for_load(
            pooling=dst_pooling, projector=dst_projector
        )

        # The method checks for pytorch_model_fsdp.bin; it must NOT exist
        # to take the new minimal path. Also need is_fsdp_enabled to be True
        # and adapter_model.safetensors to exist for the adapter path.
        # For this test we mock the FSDP adapter loading and focus on
        # the multimodal weight loading section.
        #
        # If pytorch_model_fsdp.bin exists OR fsdp is disabled, it falls through
        # to super(). We want to test the new path, so:
        # - No pytorch_model_fsdp.bin (already true for tmp_path)
        # - is_fsdp_enabled = True
        # - mock the adapter loading part
        type(trainer).is_fsdp_enabled = PropertyMock(return_value=True)

        # Create a dummy adapter_model.safetensors so the method doesn't skip
        # We mock safetensors.torch.load_file to return empty dict
        # and mock the FSDP state dict context manager
        dummy_adapter = tmp_path / "adapter_model.safetensors"
        dummy_adapter.touch()

        mock_fsdp = MagicMock()
        mock_fsdp.state_dict_type = MagicMock()
        mock_fsdp.state_dict_type.return_value.__enter__ = MagicMock()
        mock_fsdp.state_dict_type.return_value.__exit__ = MagicMock(return_value=False)

        with patch("safetensors.torch.load_file", return_value={}), \
             patch("torch.distributed.fsdp.FullyShardedDataParallel.state_dict_type") as mock_ctx, \
             patch("os.environ.get", return_value="0"):
            mock_ctx.return_value.__enter__ = MagicMock()
            mock_ctx.return_value.__exit__ = MagicMock(return_value=False)

            # model.state_dict returns empty, load_state_dict is no-op
            trainer.model.state_dict.return_value = {}
            trainer.model.load_state_dict = MagicMock()

            trainer._load_from_checkpoint(str(tmp_path))

        # After loading, dst modules should match src weights
        for k in src_pooling.state_dict():
            assert torch.equal(
                dst_pooling.state_dict()[k], src_pooling.state_dict()[k]
            ), f"Pooling key {k} not restored"

        for k in src_projector.state_dict():
            assert torch.equal(
                dst_projector.state_dict()[k], src_projector.state_dict()[k]
            ), f"Projector key {k} not restored"

    def test_skips_missing_pt_files(self, tmp_path):
        """If pooling.pt/projector.pt do not exist, loading does not crash."""
        dst_pooling = _FakePooling()
        dst_projector = _FakeProjector()

        # Snapshot the init weights so we can verify they are untouched
        orig_pooling_state = {
            k: v.clone() for k, v in dst_pooling.state_dict().items()
        }
        orig_projector_state = {
            k: v.clone() for k, v in dst_projector.state_dict().items()
        }

        trainer = self._make_trainer_for_load(
            pooling=dst_pooling, projector=dst_projector
        )
        type(trainer).is_fsdp_enabled = PropertyMock(return_value=True)

        # Create adapter file but no pooling/projector .pt files
        (tmp_path / "adapter_model.safetensors").touch()

        with patch("safetensors.torch.load_file", return_value={}), \
             patch("torch.distributed.fsdp.FullyShardedDataParallel.state_dict_type") as mock_ctx, \
             patch("os.environ.get", return_value="0"):
            mock_ctx.return_value.__enter__ = MagicMock()
            mock_ctx.return_value.__exit__ = MagicMock(return_value=False)
            trainer.model.state_dict.return_value = {}
            trainer.model.load_state_dict = MagicMock()

            # Should not raise
            trainer._load_from_checkpoint(str(tmp_path))

        # Weights unchanged (no .pt files to load)
        for k in dst_pooling.state_dict():
            assert torch.equal(dst_pooling.state_dict()[k], orig_pooling_state[k])
        for k in dst_projector.state_dict():
            assert torch.equal(dst_projector.state_dict()[k], orig_projector_state[k])

    def test_falls_through_to_super_when_fsdp_bin_exists(self, tmp_path):
        """If pytorch_model_fsdp.bin exists, super() is called (old format)."""
        (tmp_path / "pytorch_model_fsdp.bin").touch()

        trainer = self._make_trainer_for_load(
            pooling=_FakePooling(), projector=_FakeProjector()
        )
        type(trainer).is_fsdp_enabled = PropertyMock(return_value=True)

        with patch.object(
            type(trainer).__bases__[0],
            "_load_from_checkpoint",
            return_value=None,
        ) as mock_super:
            trainer._load_from_checkpoint(str(tmp_path))

        mock_super.assert_called_once()

    def test_falls_through_to_super_when_fsdp_disabled(self, tmp_path):
        """If FSDP is not enabled, super() is called regardless."""
        trainer = self._make_trainer_for_load(
            pooling=_FakePooling(), projector=_FakeProjector()
        )
        type(trainer).is_fsdp_enabled = PropertyMock(return_value=False)

        with patch.object(
            type(trainer).__bases__[0],
            "_load_from_checkpoint",
            return_value=None,
        ) as mock_super:
            trainer._load_from_checkpoint(str(tmp_path))

        mock_super.assert_called_once()


# =========================================================================
# E. No pytorch_model_fsdp.bin in new checkpoints
# =========================================================================


class TestNoPytorchModelFsdpBin:
    """Verify the efficient save path does not create pytorch_model_fsdp.bin."""

    def test_save_checkpoint_does_not_create_fsdp_bin(self, tmp_path):
        """_save_checkpoint should NOT produce pytorch_model_fsdp.bin."""
        from src.training.sft_trainer import ProteinLLMTrainer

        pooling = _FakePooling()
        projector = _FakeProjector()
        protein_llm = _FakeProteinLLM(pooling=pooling, projector=projector)

        mock_args = MagicMock()
        mock_args.should_save = True
        mock_args.output_dir = str(tmp_path)
        mock_args.local_rank = -1
        mock_args.world_size = 1
        mock_args.n_gpu = 0
        mock_args.device = torch.device("cpu")
        mock_args.should_log = True
        mock_args.report_to = []
        mock_args.logging_dir = str(tmp_path / "logs")
        mock_args.learning_rate = 1e-4

        mock_state = MagicMock()
        mock_state.global_step = 200

        trainer = object.__new__(ProteinLLMTrainer)
        trainer.protein_llm = protein_llm
        trainer.args = mock_args
        trainer.state = mock_state

        ckpt_dir = tmp_path / "checkpoint-200"
        ckpt_dir.mkdir(parents=True)

        with patch.object(
            type(trainer).__bases__[0], "_save_checkpoint", return_value=None
        ):
            trainer._save_checkpoint(model=MagicMock(), trial=None)

        assert not (ckpt_dir / "pytorch_model_fsdp.bin").exists(), (
            "pytorch_model_fsdp.bin should NOT exist in efficient checkpoints"
        )

    def test_save_optimizer_does_not_create_fsdp_bin(self, tmp_path):
        """_save_optimizer_and_scheduler should not produce pytorch_model_fsdp.bin.

        The key optimization: we call save_fsdp_optimizer directly and skip
        save_fsdp_model. Verify by checking the mock calls.
        """
        from src.training.sft_trainer import ProteinLLMTrainer

        projector = _FakeProjector()
        optimizer = torch.optim.AdamW(projector.parameters(), lr=1e-3)

        # Do a step so optimizer state exists
        loss = projector(torch.randn(2, 1536)).sum()
        loss.backward()
        optimizer.step()

        trainer = object.__new__(ProteinLLMTrainer)
        trainer._has_mm_param_group = True
        trainer.optimizer = optimizer
        trainer.lr_scheduler = MagicMock()
        trainer.lr_scheduler.state_dict.return_value = {"step": 1}
        trainer.model = MagicMock()
        trainer.accelerator = MagicMock()

        type(trainer).is_fsdp_enabled = PropertyMock(return_value=True)

        output_dir = str(tmp_path / "checkpoint-200")
        os.makedirs(output_dir, exist_ok=True)

        with patch(
            "accelerate.utils.save_fsdp_optimizer"
        ) as mock_save_fsdp_opt, \
             patch.dict(os.environ, {"RANK": "0"}):
            trainer._save_optimizer_and_scheduler(output_dir)

        # save_fsdp_optimizer was called (optimizer save)
        mock_save_fsdp_opt.assert_called_once()

        # mm_optimizer.pt was written (multimodal state)
        assert os.path.exists(os.path.join(output_dir, "mm_optimizer.pt"))

        # scheduler.pt was written
        assert os.path.exists(os.path.join(output_dir, "scheduler.pt"))

        # pytorch_model_fsdp.bin was NOT written
        assert not os.path.exists(
            os.path.join(output_dir, "pytorch_model_fsdp.bin")
        )

    def test_non_fsdp_falls_through_to_super(self, tmp_path):
        """When FSDP is disabled, _save_optimizer_and_scheduler delegates to super()."""
        from src.training.sft_trainer import ProteinLLMTrainer

        trainer = object.__new__(ProteinLLMTrainer)
        trainer._has_mm_param_group = False
        type(trainer).is_fsdp_enabled = PropertyMock(return_value=False)

        with patch.object(
            type(trainer).__bases__[0],
            "_save_optimizer_and_scheduler",
            return_value=None,
        ) as mock_super:
            trainer._save_optimizer_and_scheduler(str(tmp_path))

        mock_super.assert_called_once()


# =========================================================================
# Integration: full save -> load roundtrip
# =========================================================================


class TestSaveLoadIntegration:
    """End-to-end roundtrip: save multimodal checkpoint then load it back."""

    def test_save_and_load_roundtrip(self, tmp_path):
        """Save pooling+projector via _save_checkpoint, reload via _load_from_checkpoint."""
        from src.training.sft_trainer import ProteinLLMTrainer

        # --- Save phase ---
        pooling_orig = _FakePooling()
        projector_orig = _FakeProjector()
        protein_llm_save = _FakeProteinLLM(
            pooling=pooling_orig, projector=projector_orig
        )

        save_trainer = object.__new__(ProteinLLMTrainer)
        save_trainer.protein_llm = protein_llm_save

        mock_args = MagicMock()
        mock_args.should_save = True
        mock_args.output_dir = str(tmp_path)
        mock_args.learning_rate = 1e-4
        save_trainer.args = mock_args

        mock_state = MagicMock()
        mock_state.global_step = 50
        save_trainer.state = mock_state

        ckpt_dir = tmp_path / "checkpoint-50"
        ckpt_dir.mkdir(parents=True)

        with patch.object(
            type(save_trainer).__bases__[0], "_save_checkpoint", return_value=None
        ):
            save_trainer._save_checkpoint(model=MagicMock(), trial=None)

        assert (ckpt_dir / "pooling.pt").exists()
        assert (ckpt_dir / "projector.pt").exists()

        # --- Load phase ---
        pooling_new = _FakePooling()
        projector_new = _FakeProjector()
        protein_llm_load = _FakeProteinLLM(
            pooling=pooling_new, projector=projector_new
        )

        load_trainer = object.__new__(ProteinLLMTrainer)
        load_trainer.protein_llm = protein_llm_load
        load_trainer.args = mock_args
        load_trainer.model = MagicMock()
        type(load_trainer).is_fsdp_enabled = PropertyMock(return_value=True)

        # Create dummy adapter file
        (ckpt_dir / "adapter_model.safetensors").touch()

        with patch("safetensors.torch.load_file", return_value={}), \
             patch("torch.distributed.fsdp.FullyShardedDataParallel.state_dict_type") as mock_ctx, \
             patch("os.environ.get", return_value="0"):
            mock_ctx.return_value.__enter__ = MagicMock()
            mock_ctx.return_value.__exit__ = MagicMock(return_value=False)
            load_trainer.model.state_dict.return_value = {}
            load_trainer.model.load_state_dict = MagicMock()

            load_trainer._load_from_checkpoint(str(ckpt_dir))

        # Verify roundtrip fidelity
        for k in pooling_orig.state_dict():
            assert torch.equal(
                pooling_new.state_dict()[k], pooling_orig.state_dict()[k]
            ), f"Pooling key {k} not restored correctly in roundtrip"

        for k in projector_orig.state_dict():
            assert torch.equal(
                projector_new.state_dict()[k], projector_orig.state_dict()[k]
            ), f"Projector key {k} not restored correctly in roundtrip"
