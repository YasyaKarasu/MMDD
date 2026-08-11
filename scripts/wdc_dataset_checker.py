#!/usr/bin/env python
"""Launch the WDC implicit-query quality checker."""

from mm_joinability_dataset_checker import run_checker


if __name__ == "__main__":
    run_checker("WDC", "output_wdc_200k_sampled_20260720", 7865)
