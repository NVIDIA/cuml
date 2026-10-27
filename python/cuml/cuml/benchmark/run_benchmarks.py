# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the suite harness without importing cuML's GPU-dependent package.

Invoke this file directly for CPU-only benchmarking from a source checkout.
The benchmark package is imported under its standalone name, so relative
imports, resources, and spawned workers remain usable without initializing cuML.
"""

import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from benchmark.cli import main

    raise SystemExit(main())
