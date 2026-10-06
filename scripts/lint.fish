#!/usr/bin/env fish
set -l root (path resolve (dirname (status filename))/..)
cd $root
python -m ruff check src tests
python -m mypy src/sts2_ai
