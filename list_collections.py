#!/usr/bin/env python3
"""Backward compatible wrapper for the consolidated vdbbench CLI."""

import sys

from vdbbench.cli import main as cli_main


if __name__ == "__main__":
    exit_code = cli_main(["list", *sys.argv[1:]])
    sys.exit(exit_code)
