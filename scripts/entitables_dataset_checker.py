#!/usr/bin/env python
"""Launch the EntiTables implicit-query quality checker."""

from mm_joinability_dataset_checker import run_checker


if __name__ == "__main__":
    run_checker("EntiTables", "output_mm_joinability_v15", 7864)
