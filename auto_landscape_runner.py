#!/usr/bin/env python
"""Command-line entry point to launch AutoLand experiments."""
from __future__ import annotations

import argparse

from autoland import AutoLandscapePipeline, ExperimentSuite


def main():
    parser = argparse.ArgumentParser(description="AutoLand: automated loss landscape explorer")
    parser.add_argument("--config", required=True, help="Path to YAML configuration file")
    args = parser.parse_args()

    suite = ExperimentSuite.from_yaml(args.config)
    pipeline = AutoLandscapePipeline(suite)
    pipeline.run()


if __name__ == "__main__":
    main()
