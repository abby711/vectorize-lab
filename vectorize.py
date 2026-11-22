#!/usr/bin/env python3
"""Runs the `vectorize` command without installing the package: ./vectorize.py in.png -o out.svg"""

import sys

from vectorlab.cli import main

if __name__ == "__main__":
    sys.exit(main())
