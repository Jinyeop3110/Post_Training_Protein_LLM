---
name: code-reviewer
description: Code review for protein-LLM implementations
---

# Code Reviewer Agent

Review code changes for the protein-LLM project.

## Review Checklist

### Critical (Must Fix)
- [ ] ESM-3 encoder weights remain frozen (requires_grad=False)
- [ ] LoRA applied to all linear layers (q/k/v/o + gate/up/down), NOT just k/v
- [ ] Attention pooling used (not mean pooling) for MLP path
- [ ] Model configs use Instruct variants (e.g., Qwen3-4B-Instruct-2507)
- [ ] Training uses chat template format with system prompt
- [ ] No secrets or credentials in code
- [ ] CUDA operations are safe

### Type Safety
- [ ] All functions have type hints
- [ ] Return types are explicit
- [ ] Optional types used correctly

### Code Quality
- [ ] Google-style docstrings present
- [ ] No hardcoded paths (use config)
- [ ] Error handling appropriate
- [ ] Flash Attention enabled for H100

### Performance
- [ ] Gradient checkpointing where needed
- [ ] Memory-efficient operations
- [ ] Proper batch handling

## Review Commands
```bash
# Run linter
ruff check src/

# Run type checker
mypy src/

# Run tests
pytest tests/ -v

# Check coverage
pytest tests/ --cov=src --cov-report=html
```

## Review Format

```markdown
## Code Review: [filename]

### Summary
Brief description of changes

### Issues
- [ ] Critical: [description]
- [ ] Warning: [description]
- [ ] Suggestion: [description]

### Approved: Yes/No
```

## Common Issues

### Memory Leaks
```python
# Bad
for batch in dataloader:
    outputs.append(model(batch))  # Accumulates gradients

# Good
with torch.no_grad():
    for batch in dataloader:
        outputs.append(model(batch).cpu())
```

### Config Hardcoding
```python
# Bad
model = AutoModel.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")

# Good
model = AutoModel.from_pretrained(cfg.model.path)
```
