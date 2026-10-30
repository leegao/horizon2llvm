"""CLI wrapper around the C++ LLVM IR lifter (tools/lifter/main.cpp).

Maintains drop-in command-line compatibility with earlier scripts while delegating all NSO loading,
AArch64 -> LLVM IR lifting, and LLVM passes to the native ultratears-lifter binary.
"""
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'tools'))
from gen_game import build_lifter_binary  # noqa: E402


def main():
    lifter_bin = build_lifter_binary()
    sys.exit(subprocess.run([str(lifter_bin)] + sys.argv[1:]).returncode)


if __name__ == '__main__':
    main()
