---
name: fix-issue
description: Fix a GitHub issue
disable-model-invocation: true
---
Fix issue: $ARGUMENTS

1. Use `gh issue view $ARGUMENTS` to understand the problem
2. Search codebase for relevant files
3. Implement fix with proper type hints and docstrings
4. Write/update tests if applicable
5. Run tests: `pytest src/ -v`
6. Create descriptive commit following conventional commits
