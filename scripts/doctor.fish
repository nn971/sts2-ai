#!/usr/bin/env fish
set -l root (path resolve (dirname (status filename))/..)
cd $root

set -l failed 0

echo "sts2-ai doctor"
echo "=============="

for cmd in git python
    if type -q $cmd
        echo "[ok] $cmd -> "(command -s $cmd)
    else
        echo "[missing] $cmd"
        set failed 1
    end
end

if test -d .git
    echo "[ok] Git repository initialized"
else
    echo "[info] Git repository has not been initialized yet"
end

if test -f .gitmodules
    echo "[ok] .gitmodules exists"
    git submodule status 2>/dev/null; or true
else
    echo "[info] emulator submodule not added yet"
end

if test -d emulator
    set -l emu_commit (git -C emulator rev-parse --short HEAD 2>/dev/null)
    if test -n "$emu_commit"
        echo "[ok] emulator checkout: $emu_commit"
    else
        echo "[warn] emulator/ exists but is not a readable Git checkout"
    end
end

python -c 'import sts2_ai; print("[ok] sts2_ai", sts2_ai.__version__)' 2>/dev/null
if test $status -ne 0
    echo "[info] Python package is not installed in the active environment"
end

exit $failed
