#!/usr/bin/env fish
set -l root (git rev-parse --show-toplevel 2>/dev/null)
if test $status -ne 0
    echo "error: run this inside an initialized sts2-ai Git repository" >&2
    exit 1
end

cd $root

if test -e emulator
    echo "error: emulator path already exists" >&2
    exit 1
end

set -l url ../sts2-emulator.git
if test (count $argv) -ge 1
    set url $argv[1]
end

echo "Adding sts2-emulator submodule from: $url"
git submodule add $url emulator; or exit $status
git submodule update --init --recursive; or exit $status

echo
echo "Submodule added. Review it, then commit with:"
echo "  git add .gitmodules emulator"
echo '  git commit -m "Add sts2-emulator submodule"'
