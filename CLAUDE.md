Development notes for this repository.

## Environment
conda env `mlmolprop`. Editable install. Never install mlmolprop-lab here.

## Rules
- Never import from mlmolprop_lab. Dependency direction is one-way.
- Never add the mlmolprop-lab repo as a git remote.
- Never commit data files; examples/data holds only public benchmarks.
- Before any commit, run: pip install -e ".[dev]" && pytest && ruff check .
- Tests: prefer property-based assertions over hardcoded expected values.

## Release checklist
1. Tests and ruff pass.
2. Bump version in pyproject.toml.
3. Bump version in CITATION.cff to match.
4. Commit, push.
5. Tag matching the version, push the tag.
6. Write a short GitHub Release note.
7. If a lab module was released, delete it from lab.
8. Update the lab pin and re-run lab tests.
