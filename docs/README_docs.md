# DALES case generator docs

This folder contains Sphinx configuration for generating HTML documentation
for the DALES case generator, based on Python docstrings.

## Building the docs

From the project root (where `docs/` lives) and with `sphinx` installed, run:

```bash
cd docs
sphinx-build -b html . _build/html
```