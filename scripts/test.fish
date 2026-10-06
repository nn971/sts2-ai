#!/usr/bin/env fish
set -l root (path resolve (dirname (status filename))/..)
cd $root
python -m pytest
