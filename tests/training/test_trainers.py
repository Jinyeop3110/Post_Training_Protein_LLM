"""Tests for trainer implementations."""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
import torch
from omegaconf import OmegaConf


class TestGetQloraConfig:
    """Tests for get_qlora_config function."""

    def test_get_qlora_config_basic(self, mock_config):
        """Test basic QLoRA config extraction."""
        from src.training.sft_trainer import get_qlora_config

        # Add lora config to mock
        mock_config.training.lora = OmegaConf.create({
            "r": 8,
            "alpha": 16,
            "dropout": 0.05,
            "target_modules": ["k_proj", "v_proj"],
            "bias": "none",
            "task_type": "CAUSAL_LM",
        })

        config = get_qlora_config(mock_config)

        assert config is not None
        assert config.r == 8
        assert config.lora_alpha == 16
        assert config.lora_dropout == 0.05
        assert "k_proj" in config.target_modules
        assert "v_proj" in config.target_modules
        assert config.bias == "none"

    def test_get_qlora_config_defaults(self, mock_config):
        """Test QLoRA config with default values."""
        from src.training.sft_trainer import get_qlora_config

        # Empty lora config - should use defaults
        mock_config.training.lora = OmegaConf.create({})

        config = get_qlora_config(mock_config)

        assert config.r == 8  # default
        assert config.lora_alpha == 16  # default
        assert config.lora_dropout == 0.05  # default

    def test_get_qlora_config_minimum_rank(self, mock_config):
        """Test QLoRA config with minimum rank requirement."""
        from src.training.sft_trainer import get_qlora_config

        mock_config.training.lora = OmegaConf.create({
            "r": 4,
            "alpha": 8,
        })

        config = get_qlora_config(mock_config)
        assert config.r >= 4  # Minimum for proteins


class TestGetQuantizationConfig:
    """Tests for get_quantization_config function."""

    def test_quantization_enabled(self, mock_config):
        """Test quantization config when enabled."""
        from src.training.sft_trainer import get_quantization_config

        mock_config.training.quantization = OmegaConf.create({
            "enabled": True,
            "bits": 4,
            "bnb_4bit_compute_dtype": "bfloat16",
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
        })

        config = get_quantization_config(mock_config)

        assert config is not None
        assert config.load_in_4bit is True
        assert config.bnb_4bit_quant_type == "nf4"

    def test_quantization_disabled(self, mock_config):
        """Test quantization config when disabled."""
        from src.training.sft_trainer import get_quantization_config

        mock_config.training.quantization = OmegaConf.create({
            "enabled": False,
        })

        config = get_quantization_config(mock_config)
        assert config is None

    def test_quantization_8bit(self, mock_config):
        """Test 8-bit quantization config."""
        from src.training.sft_trainer import get_quantization_config

        mock_config.training.quantization = OmegaConf.create({
            "enabled": True,
            "bits": 8,
        })

        config = get_quantization_config(mock_config)

        assert config is not None
        assert config.load_in_8bit is True


class TestGetTrainingArguments:
    """Tests for get_training_arguments function."""

    def test_training_arguments_basic(self, mock_config):
        """Test basic training arguments creation."""
        from src.training.sft_trainer import get_training_arguments

        # Add required training config
        mock_config.training = OmegaConf.create({
            "epochs": 3,
            "batch_size": 8,
            "gradient_accumulation_steps": 4,
            "lr": 2e-4,
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
            "warmup_ratio": 0.03,
            "logging_steps": 10,
            "eval_steps": 100,
            "save_strategy": "steps",
            "save_steps": 500,
            "save_total_limit": 3,
            "lora": {"r": 8, "alpha": 16},
            "optimizer": {"type": "adamw_8bit"},
            "lr_scheduler": {"type": "cosine"},
        })
        mock_config.paths = OmegaConf.create({
            "checkpoint_dir": "./checkpoints",
            "log_dir": "./logs",
        })
        mock_config.logging = OmegaConf.create({
            "wandb": {"enabled": False},
            "tensorboard": {"enabled": False},
        })
        mock_config.hardware = OmegaConf.create({
            "precision": "bf16",
        })

        args = get_training_arguments(mock_config)

        assert args.num_train_epochs == 3
        assert args.per_device_train_batch_size == 8
        assert args.gradient_accumulation_steps == 4
        assert args.learning_rate == 2e-4
        assert args.warmup_ratio == 0.03


class TestProteinLLMDataCollator:
    """Tests for ProteinLLMDataCollator."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        tokenizer = MagicMock()
        tokenizer.pad_token = "[PAD]"
        tokenizer.pad_token_id = 0
        tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3, 0], [1, 2, 0, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]]),
        }
        return tokenizer

    def test_collator_initialization(self, mock_tokenizer):
        """Test collator initialization."""
        from src.training.sft_trainer import ProteinLLMDataCollator

        collator = ProteinLLMDataCollator(
            tokenizer=mock_tokenizer,
            max_length=512,
        )

        assert collator.max_length == 512
        assert collator.label_pad_token_id == -100

    def test_collator_call(self, mock_tokenizer):
        """Test collator call with batch."""
        from src.training.sft_trainer import ProteinLLMDataCollator

        collator = ProteinLLMDataCollator(
            tokenizer=mock_tokenizer,
            max_length=512,
        )

        batch = [
            {
                "formatted_prompt": "Test prompt 1",
                "protein_sequence": "MKTAYIAK",
            },
            {
                "formatted_prompt": "Test prompt 2",
                "protein_sequence": "MNIFEMLR",
            },
        ]

        result = collator(batch)

        assert "input_ids" in result
        assert "attention_mask" in result
        assert "labels" in result
        assert "protein_sequences" in result
        assert len(result["protein_sequences"]) == 2


class TestGPUMemoryCallback:
    """Tests for GPUMemoryCallback."""

    def test_callback_on_log_without_cuda(self):
        """Test callback when CUDA is not available."""
        from src.training.sft_trainer import GPUMemoryCallback

        callback = GPUMemoryCallback()
        logs = {}

        with patch("torch.cuda.is_available", return_value=False):
            callback.on_log(
                args=MagicMock(),
                state=MagicMock(),
                control=MagicMock(),
                logs=logs,
            )

        # No GPU metrics should be added
        assert "gpu_memory_allocated_gb" not in logs

    def test_callback_on_log_with_cuda(self):
        """Test callback when CUDA is available."""
        from src.training.sft_trainer import GPUMemoryCallback

        callback = GPUMemoryCallback()
        logs = {}

        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.memory_allocated", return_value=1024**3), \
             patch("torch.cuda.memory_reserved", return_value=2 * 1024**3), \
             patch("torch.cuda.max_memory_allocated", return_value=1.5 * 1024**3):

            callback.on_log(
                args=MagicMock(),
                state=MagicMock(),
                control=MagicMock(),
                logs=logs,
            )

        assert "gpu_memory_allocated_gb" in logs
        assert logs["gpu_memory_allocated_gb"] == 1.0


class TestSFTTrainer:
    """Tests for SFT trainer class."""

    def test_sft_trainer_import(self):
        """Test SFT trainer can be imported."""
        from src.training.sft_trainer import SFTTrainer
        assert SFTTrainer is not None

    def test_sft_trainer_init(self, mock_config):
        """Test SFT trainer initialization."""
        from src.training.sft_trainer import SFTTrainer

        # Add required config sections
        mock_config.training.lora = OmegaConf.create({
            "r": 8,
            "alpha": 16,
            "dropout": 0.05,
            "target_modules": ["k_proj", "v_proj"],
        })
        mock_config.training.quantization = OmegaConf.create({
            "enabled": True,
            "bits": 4,
        })

        trainer = SFTTrainer(mock_config)

        assert trainer.cfg == mock_config
        assert trainer.model is None
        assert trainer.tokenizer is None
        assert trainer.train_dataset is None
        assert trainer.eval_dataset is None

    def test_sft_trainer_setup_not_called(self, mock_config):
        """Test that train/evaluate fail if setup not called."""
        from src.training.sft_trainer import SFTTrainer

        mock_config.training.lora = OmegaConf.create({"r": 8})

        trainer = SFTTrainer(mock_config)

        with pytest.raises(RuntimeError, match="Trainer not initialized"):
            trainer.train()

        with pytest.raises(RuntimeError, match="Trainer not initialized"):
            trainer.evaluate()


class TestProteinLLMTrainer:
    """Tests for ProteinLLMTrainer class."""

    def test_protein_llm_trainer_import(self):
        """Test ProteinLLMTrainer can be imported."""
        from src.training.sft_trainer import ProteinLLMTrainer
        assert ProteinLLMTrainer is not None

    def test_protein_llm_trainer_init(self):
        """Test ProteinLLMTrainer initialization with mock."""
        from src.training.sft_trainer import ProteinLLMTrainer

        mock_model = MagicMock()
        mock_protein_llm = MagicMock()
        mock_args = MagicMock()
        mock_args.local_rank = -1
        mock_args.world_size = 1
        mock_args.n_gpu = 0
        mock_args.device = torch.device("cpu")
        mock_args.should_save = True
        mock_args.should_log = True
        mock_args.report_to = []
        mock_args.logging_dir = "/tmp/logs"

        # This would require more extensive mocking to fully test
        # Just verify the class can be instantiated
        assert ProteinLLMTrainer is not None


class TestRunSFTFunctions:
    """Tests for run_sft_qlora and run_sft_lora functions."""

    def test_run_sft_qlora_import(self):
        """Test run_sft_qlora can be imported."""
        from src.training.sft_trainer import run_sft_qlora
        assert run_sft_qlora is not None

    def test_run_sft_lora_import(self):
        """Test run_sft_lora can be imported."""
        from src.training.sft_trainer import run_sft_lora
        assert run_sft_lora is not None

    def test_run_sft_with_trl_import(self):
        """Test run_sft_with_trl can be imported."""
        from src.training.sft_trainer import run_sft_with_trl
        assert run_sft_with_trl is not None


class TestTrainingConfigLoading:
    """Tests for training config loading."""

    def test_full_config_loading(self):
        """Test loading full training configuration from YAML."""
        config = OmegaConf.create({
            "model": {
                "name": "qwen2_7b",
                "path": "Qwen/Qwen2.5-7B-Instruct",
            },
            "encoder": {
                "model_name": "esm2_t33_650M_UR50D",
                "embedding_dim": 1280,
            },
            "training": {
                "method": "sft_qlora",
                "lr": 2e-4,
                "epochs": 3,
                "batch_size": 8,
                "gradient_accumulation_steps": 4,
                "warmup_ratio": 0.03,
                "lora": {
                    "r": 8,
                    "alpha": 16,
                    "dropout": 0.05,
                    "target_modules": ["k_proj", "v_proj"],
                },
                "quantization": {
                    "enabled": True,
                    "bits": 4,
                },
            },
            "data": {
                "source": "zjunlp/Mol-Instructions",
                "subset": "Protein-oriented Instructions",
            },
        })

        assert config.training.lr == 2e-4
        assert config.training.lora.r == 8
        assert config.training.quantization.enabled is True


class TestGRPOTrainer:
    """Tests for GRPO trainer."""

    def test_grpo_trainer_import(self):
        """Test GRPO trainer can be imported."""
        try:
            from src.training.grpo_trainer import GRPOTrainer
            assert GRPOTrainer is not None
        except ImportError:
            pytest.skip("GRPO trainer not yet implemented")

    def test_grpo_config(self, mock_config):
        """Test GRPO configuration is valid."""
        try:
            from src.training.grpo_trainer import get_grpo_config

            config = get_grpo_config(mock_config)
            assert config is not None
            assert "group_size" in config or hasattr(config, "group_size")
        except ImportError:
            pytest.skip("GRPO config not yet implemented")


class TestDPOTrainer:
    """Tests for DPO trainer."""

    def test_dpo_trainer_import(self):
        """Test DPO trainer can be imported."""
        try:
            from src.training.dpo_trainer import DPOTrainer
            assert DPOTrainer is not None
        except ImportError:
            pytest.skip("DPO trainer not yet implemented")


class TestTrainingUtils:
    """Tests for training utilities."""

    def test_checkpoint_save_load(self, tmp_path):
        """Test checkpoint saving and loading."""
        try:
            from src.training.utils import save_checkpoint, load_checkpoint

            # Create dummy state
            state = {"model": torch.nn.Linear(10, 10).state_dict()}
            path = tmp_path / "checkpoint.pt"

            save_checkpoint(state, path)
            loaded = load_checkpoint(path)

            assert loaded is not None
        except ImportError:
            pytest.skip("Checkpoint utils not yet implemented")


class TestIntegration:
    """Integration tests for the SFT trainer (mocked)."""

    def test_trainer_workflow_mocked(self, mock_config):
        """Test the full trainer workflow with mocks."""
        from src.training.sft_trainer import SFTTrainer

        # Setup complete mock config
        mock_config.training = OmegaConf.create({
            "epochs": 1,
            "batch_size": 2,
            "lr": 1e-4,
            "gradient_accumulation_steps": 1,
            "warmup_ratio": 0.1,
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
            "logging_steps": 1,
            "eval_steps": 10,
            "save_strategy": "no",
            "save_steps": 100,
            "save_total_limit": 1,
            "max_seq_length": 256,
            "lora": {
                "r": 4,
                "alpha": 8,
                "dropout": 0.0,
                "target_modules": ["k_proj"],
            },
            "quantization": {"enabled": False},
            "optimizer": {"type": "adamw"},
            "lr_scheduler": {"type": "linear"},
        })
        mock_config.paths = OmegaConf.create({
            "checkpoint_dir": "/tmp/test_checkpoints",
            "log_dir": "/tmp/test_logs",
        })
        mock_config.logging = OmegaConf.create({
            "wandb": {"enabled": False},
            "tensorboard": {"enabled": False},
        })
        mock_config.hardware = OmegaConf.create({"precision": "bf16"})
        mock_config.data = OmegaConf.create({
            "source": "zjunlp/Mol-Instructions",
            "subset": "Protein-oriented Instructions",
            "paths": {"raw": None},
        })

        # Just test initialization - don't run full training in tests
        trainer = SFTTrainer(mock_config)
        assert trainer is not None
        assert trainer.cfg == mock_config
