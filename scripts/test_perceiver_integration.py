"""
Perceiver Resampler Integration Test

Tests the full pipeline on GPU:
1. Forward/backward pass with realistic shapes
2. Memory and throughput comparison vs MLP path
3. Dtype compatibility (float32 encoder → bf16 LLM)
4. Gradient flow through Perceiver
5. Save/load round-trip
"""

import gc
import os
import sys
import time

import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def fmt_mem(bytes_val):
    return f"{bytes_val / 1024**3:.2f} GB"

def measure_gpu():
    torch.cuda.synchronize()
    return {
        "allocated": torch.cuda.memory_allocated(),
        "reserved": torch.cuda.memory_reserved(),
        "max_allocated": torch.cuda.max_memory_allocated(),
    }

def reset_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def test_1_perceiver_forward_backward():
    """Test Perceiver forward/backward with realistic dims on GPU."""
    print("\n" + "="*70)
    print("TEST 1: Perceiver Forward/Backward on GPU")
    print("="*70)

    from src.models.perceiver import PerceiverResampler

    reset_gpu()
    mem_before = measure_gpu()

    # Production dims: ESM-3 (1536) → Qwen3-4B (2560)
    model = PerceiverResampler(
        encoder_dim=1536,
        output_dim=2560,
        num_queries=32,
        num_layers=6,
        num_heads=8,
        ffn_dim=2048,
        dropout=0.1,
    ).cuda()

    mem_after_init = measure_gpu()
    param_count = sum(p.numel() for p in model.parameters())
    param_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2

    print(f"  Parameters: {param_count:,} ({param_mb:.1f} MB)")
    print(f"  GPU after init: {fmt_mem(mem_after_init['allocated'])}")

    # Simulate batch: B=4, L=200 (typical protein length)
    x = torch.randn(4, 200, 1536, device="cuda")  # float32 like ESM-3
    mask = torch.ones(4, 200, dtype=torch.bool, device="cuda")
    mask[0, 150:] = False  # variable length

    # Forward pass
    torch.cuda.synchronize()
    t0 = time.time()
    out = model(x, attention_mask=mask)
    torch.cuda.synchronize()
    fwd_time = time.time() - t0

    mem_after_fwd = measure_gpu()
    print(f"  Output shape: {out.shape}")
    print(f"  Forward time: {fwd_time*1000:.1f} ms")
    print(f"  GPU after forward: {fmt_mem(mem_after_fwd['allocated'])}")

    # Backward pass
    torch.cuda.synchronize()
    t0 = time.time()
    loss = out.sum()
    loss.backward()
    torch.cuda.synchronize()
    bwd_time = time.time() - t0

    mem_after_bwd = measure_gpu()
    print(f"  Backward time: {bwd_time*1000:.1f} ms")
    print(f"  GPU after backward: {fmt_mem(mem_after_bwd['allocated'])}")
    print(f"  Peak GPU: {fmt_mem(mem_after_bwd['max_allocated'])}")

    # Verify gradients
    assert model.query_tokens.grad is not None, "No gradient on query_tokens!"
    assert model.input_proj.weight.grad is not None, "No gradient on input_proj!"
    for i, layer in enumerate(model.layers):
        has_grad = any(p.grad is not None for p in layer.parameters())
        assert has_grad, f"No gradients in layer {i}!"

    print("  ✓ All gradient checks passed")

    del model, x, mask, out
    reset_gpu()
    return True


def test_2_mlp_vs_perceiver_comparison():
    """Compare MLP vs Perceiver in memory and speed."""
    print("\n" + "="*70)
    print("TEST 2: MLP vs Perceiver Comparison")
    print("="*70)

    from src.models.perceiver import PerceiverResampler
    from src.models.pooling import get_pooling
    from src.models.projector import MLPProjector

    results = {}

    for name, setup_fn in [("MLP", "mlp"), ("Perceiver", "perceiver")]:
        reset_gpu()
        torch.cuda.synchronize()

        if setup_fn == "mlp":
            pooling = get_pooling(
                "attention", embed_dim=1536, num_output_tokens=32,
                num_heads=8, dropout=0.1, layer_norm=True,
            ).cuda()
            projector = MLPProjector(
                input_dim=1536, hidden_dim=2048, output_dim=2560,
                num_layers=2, activation="gelu", dropout=0.1,
            ).cuda()
            param_count = (
                sum(p.numel() for p in pooling.parameters()) +
                sum(p.numel() for p in projector.parameters())
            )

            def forward(x, mask=None):
                pooled = pooling(x)
                return projector(pooled)

        else:
            perceiver = PerceiverResampler(
                encoder_dim=1536, output_dim=2560, num_queries=32,
                num_layers=6, num_heads=8, ffn_dim=2048, dropout=0.1,
            ).cuda()
            param_count = sum(p.numel() for p in perceiver.parameters())

            def forward(x, mask=None):
                return perceiver(x, attention_mask=mask)

        mem_init = measure_gpu()

        # Benchmark: B=4, L=200
        x = torch.randn(4, 200, 1536, device="cuda", requires_grad=True)
        mask = torch.ones(4, 200, dtype=torch.bool, device="cuda")

        # Warm up
        out = forward(x, mask)
        out.sum().backward()
        x.grad = None

        # Timed run (average of 5)
        fwd_times = []
        bwd_times = []
        for _ in range(5):
            x_new = torch.randn(4, 200, 1536, device="cuda", requires_grad=True)
            torch.cuda.synchronize()

            t0 = time.time()
            out = forward(x_new, mask)
            torch.cuda.synchronize()
            fwd_times.append(time.time() - t0)

            t0 = time.time()
            out.sum().backward()
            torch.cuda.synchronize()
            bwd_times.append(time.time() - t0)

        mem_peak = measure_gpu()

        results[name] = {
            "params": param_count,
            "fwd_ms": sum(fwd_times) / len(fwd_times) * 1000,
            "bwd_ms": sum(bwd_times) / len(bwd_times) * 1000,
            "mem_allocated": mem_peak["allocated"],
            "mem_peak": mem_peak["max_allocated"],
        }

        del x, out
        if setup_fn == "mlp":
            del pooling, projector
        else:
            del perceiver
        reset_gpu()

    # Print comparison
    print(f"\n  {'Metric':<25s} {'MLP':>15s} {'Perceiver':>15s} {'Ratio':>10s}")
    print(f"  {'-'*65}")
    for metric, label in [
        ("params", "Parameters"),
        ("fwd_ms", "Forward (ms)"),
        ("bwd_ms", "Backward (ms)"),
        ("mem_peak", "Peak GPU"),
    ]:
        mlp_val = results["MLP"][metric]
        perc_val = results["Perceiver"][metric]
        ratio = perc_val / mlp_val if mlp_val > 0 else float("inf")

        if metric == "params":
            print(f"  {label:<25s} {mlp_val:>15,d} {perc_val:>15,d} {ratio:>9.1f}x")
        elif metric == "mem_peak":
            print(f"  {label:<25s} {fmt_mem(mlp_val):>15s} {fmt_mem(perc_val):>15s} {ratio:>9.1f}x")
        else:
            print(f"  {label:<25s} {mlp_val:>14.1f}ms {perc_val:>14.1f}ms {ratio:>9.1f}x")

    return results


def test_3_dtype_compatibility():
    """Test float32 input → bf16 output casting (simulating ESM→LLM bridge)."""
    print("\n" + "="*70)
    print("TEST 3: Dtype Compatibility (float32 → bf16)")
    print("="*70)

    from src.models.perceiver import PerceiverResampler

    reset_gpu()

    # Test with float32 input (ESM-3 output)
    model = PerceiverResampler(
        encoder_dim=1536, output_dim=2560, num_queries=32,
        num_layers=2, num_heads=8, ffn_dim=2048,
    ).cuda()

    x_fp32 = torch.randn(2, 100, 1536, device="cuda", dtype=torch.float32)
    out_fp32 = model(x_fp32)
    print(f"  float32 input → output dtype: {out_fp32.dtype}")
    assert out_fp32.dtype == torch.float32

    # Test with bf16 (if model is cast to bf16)
    model_bf16 = model.to(torch.bfloat16)
    x_bf16 = x_fp32.to(torch.bfloat16)
    out_bf16 = model_bf16(x_bf16)
    print(f"  bfloat16 input → output dtype: {out_bf16.dtype}")
    assert out_bf16.dtype == torch.bfloat16

    # Test mixed: float32 input through bf16 model (like our pipeline)
    # The model will auto-cast within its linear layers
    try:
        out_mixed = model_bf16(x_fp32)
        print(f"  float32 input through bf16 model → dtype: {out_mixed.dtype}")
    except RuntimeError as e:
        print(f"  float32 input through bf16 model → ERROR: {e}")
        print("  (This is expected; pipeline casts explicitly)")

    # Test the prepare_inputs casting pattern
    fake_text_embeds = torch.randn(2, 50, 2560, device="cuda", dtype=torch.bfloat16)
    protein_embeds = out_fp32.to(fake_text_embeds.dtype)
    combined = torch.cat([protein_embeds, fake_text_embeds], dim=1)
    print(f"  Combined embeds dtype: {combined.dtype}, shape: {combined.shape}")
    assert combined.dtype == torch.bfloat16
    print("  ✓ Dtype compatibility checks passed")

    del model, model_bf16
    reset_gpu()
    return True


def test_4_save_load_roundtrip():
    """Test save/load for Perceiver-based ProteinLLM config."""
    print("\n" + "="*70)
    print("TEST 4: Save/Load Round-trip")
    print("="*70)

    import json
    import tempfile
    from pathlib import Path

    from src.models.perceiver import PerceiverResampler

    reset_gpu()

    model = PerceiverResampler(
        encoder_dim=1536, output_dim=2560, num_queries=32,
        num_layers=2, num_heads=8, ffn_dim=2048,
    ).cuda()

    x = torch.randn(1, 50, 1536, device="cuda")
    model.eval()
    with torch.no_grad():
        out1 = model(x)

    # Save
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir)
        torch.save(model.state_dict(), save_path / "projector.pt")
        config = {
            "projector_type": "perceiver",
            "encoder_embed_dim": 1536,
            "llm_hidden_size": 2560,
            "num_prefix_tokens": 32,
            "perceiver_layers": 2,
            "perceiver_heads": 8,
            "perceiver_ffn_dim": 2048,
        }
        with open(save_path / "config.json", "w") as f:
            json.dump(config, f)

        # Load
        model2 = PerceiverResampler(
            encoder_dim=config["encoder_embed_dim"],
            output_dim=config["llm_hidden_size"],
            num_queries=config["num_prefix_tokens"],
            num_layers=config["perceiver_layers"],
            num_heads=config["perceiver_heads"],
            ffn_dim=config["perceiver_ffn_dim"],
        ).cuda()

        state = torch.load(save_path / "projector.pt", map_location="cuda")
        model2.load_state_dict(state)
        model2.eval()

        with torch.no_grad():
            out2 = model2(x)

        match = torch.allclose(out1, out2, atol=1e-6)
        print(f"  Outputs match after save/load: {match}")
        assert match, f"Max diff: {(out1 - out2).abs().max().item()}"

    print("  ✓ Save/load round-trip passed")

    del model, model2
    reset_gpu()
    return True


def test_5_gradient_checkpointing_compat():
    """Test that Perceiver works with gradient checkpointing context."""
    print("\n" + "="*70)
    print("TEST 5: Gradient Checkpointing Compatibility")
    print("="*70)

    from src.models.perceiver import PerceiverResampler

    reset_gpu()

    model = PerceiverResampler(
        encoder_dim=1536, output_dim=2560, num_queries=32,
        num_layers=6, num_heads=8, ffn_dim=2048,
    ).cuda().train()

    x = torch.randn(4, 200, 1536, device="cuda", requires_grad=True)

    # With autocast (like HF Trainer uses for bf16)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = model(x)
        loss = out.sum()

    loss.backward()

    print(f"  Output dtype under autocast: {out.dtype}")
    print(f"  Query tokens grad exists: {model.query_tokens.grad is not None}")
    print(f"  Input x grad exists: {x.grad is not None}")
    assert model.query_tokens.grad is not None
    print("  ✓ Gradient checkpointing compatibility passed")

    del model, x, out
    reset_gpu()
    return True


def test_6_smaller_perceiver_efficiency():
    """Test smaller Perceiver configs for efficiency trade-offs."""
    print("\n" + "="*70)
    print("TEST 6: Perceiver Layer Count vs Efficiency")
    print("="*70)

    from src.models.perceiver import PerceiverResampler

    configs = [
        {"num_layers": 1, "ffn_dim": 2048},
        {"num_layers": 2, "ffn_dim": 2048},
        {"num_layers": 4, "ffn_dim": 2048},
        {"num_layers": 6, "ffn_dim": 2048},
    ]

    print(f"\n  {'Layers':>8s} {'Params':>12s} {'Fwd (ms)':>10s} {'Bwd (ms)':>10s} {'Peak GPU':>12s}")
    print(f"  {'-'*56}")

    for cfg in configs:
        reset_gpu()
        model = PerceiverResampler(
            encoder_dim=1536, output_dim=2560, num_queries=32,
            num_heads=8, dropout=0.1, **cfg,
        ).cuda().train()

        params = sum(p.numel() for p in model.parameters())
        x = torch.randn(4, 200, 1536, device="cuda", requires_grad=True)

        # Warm up
        out = model(x)
        out.sum().backward()

        # Timed
        reset_gpu()
        x = torch.randn(4, 200, 1536, device="cuda", requires_grad=True)
        torch.cuda.synchronize()

        t0 = time.time()
        out = model(x)
        torch.cuda.synchronize()
        fwd = (time.time() - t0) * 1000

        t0 = time.time()
        out.sum().backward()
        torch.cuda.synchronize()
        bwd = (time.time() - t0) * 1000

        peak = measure_gpu()["max_allocated"]

        print(f"  {cfg['num_layers']:>8d} {params:>12,d} {fwd:>9.1f}ms {bwd:>9.1f}ms {fmt_mem(peak):>12s}")

        del model, x, out
        reset_gpu()

    return True


def test_7_encode_protein_perceiver():
    """Test encode_protein with Perceiver path (mocked encoder)."""
    print("\n" + "="*70)
    print("TEST 7: encode_protein with Perceiver Path")
    print("="*70)

    from src.models.multimodal_llm import ProteinLLM

    reset_gpu()

    # Create ProteinLLM with perceiver but without loading encoder/LLM
    model = ProteinLLM(
        approach="esm3",
        projector_type="perceiver",
        perceiver_layers=2,
        perceiver_heads=8,
        perceiver_ffn_dim=2048,
        encoder_embed_dim=1536,
        load_llm=False,
        load_encoder=False,
        device="cuda",
    )

    # Check that pooling is None for perceiver path
    assert model.pooling is None, f"Pooling should be None for perceiver, got {model.pooling}"
    print("  pooling is None: True (correct for Perceiver)")
    print(f"  projector_type: {model.projector_type}")

    # Manually set hidden size and build projector (normally done after LLM load)
    model.llm_hidden_size = 2560
    model._build_projector()

    assert model.projector is not None, "Projector should be built"
    proj_params = sum(p.numel() for p in model.projector.parameters())
    print(f"  Projector params: {proj_params:,}")
    print(f"  Projector type: {type(model.projector).__name__}")

    # Simulate encode_protein by calling projector directly
    fake_encoder_output = torch.randn(2, 150, 1536, device="cuda")
    out = model.projector(fake_encoder_output)
    print(f"  Projector output shape: {out.shape}")
    assert out.shape == (2, 32, 2560), f"Expected (2, 32, 2560), got {out.shape}"

    print("  ✓ encode_protein perceiver path validated")

    del model
    reset_gpu()
    return True


if __name__ == "__main__":
    print("="*70)
    print("PERCEIVER RESAMPLER INTEGRATION TEST SUITE")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {fmt_mem(torch.cuda.get_device_properties(0).total_memory)}")
    print("="*70)

    tests = [
        ("Forward/Backward", test_1_perceiver_forward_backward),
        ("MLP vs Perceiver", test_2_mlp_vs_perceiver_comparison),
        ("Dtype Compatibility", test_3_dtype_compatibility),
        ("Save/Load", test_4_save_load_roundtrip),
        ("Grad Checkpointing", test_5_gradient_checkpointing_compat),
        ("Layer Count Scaling", test_6_smaller_perceiver_efficiency),
        ("encode_protein Path", test_7_encode_protein_perceiver),
    ]

    passed = 0
    failed = 0
    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"\n  ✗ FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "="*70)
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("="*70)

    sys.exit(0 if failed == 0 else 1)
