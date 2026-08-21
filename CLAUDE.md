Development notes for this repository.

## Environment
conda env `mlmolprop`. Editable install. Never install mlmolprop-lab,
mlmolprop-interpretation, or mlmolprop-pytorch here.

## Rules
- Never import from mlmolprop_lab, mlmolprop_interpretation, or
  mlmolprop_pytorch. Dependency direction is one-way: those three
  private repos depend on this one, never the reverse.
- Never add mlmolprop-lab, mlmolprop-interpretation, or mlmolprop-pytorch
  as a git remote here.
- Never commit data files; examples/data holds only public benchmarks.
- Before any commit, run: pip install -e ".[dev]" && pytest && ruff check .
- Tests: prefer property-based assertions over hardcoded expected values.
- Always ask before cutting a new release (i.e. before starting the Release
  checklist below). Plain commits/pushes of working-tree changes don't need
  to ask, but a version bump + tag does -- not every change is meant to ship
  immediately (e.g. a library addition not yet wired into any pipeline).

## Release checklist
1. Tests and ruff pass.
2. Bump version in pyproject.toml.
3. Bump version in CITATION.cff to match.
4. Commit, push.
5. Tag matching the version, push the tag.
6. Write a short GitHub Release note.
7. If a module from mlmolprop-lab, mlmolprop-interpretation, or
   mlmolprop-pytorch was released here, delete it from that repo.
8. For any of those three private repos you want on the new version,
   update its pin and re-run its tests (see that repo's own CLAUDE.md
   for the live-vs-pinned install workflow).
