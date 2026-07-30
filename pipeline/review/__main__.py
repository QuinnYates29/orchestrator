"""Allows `python3 -m pipeline.review` for the deterministic checks alone."""
from __future__ import annotations

import sys

from pipeline.review.cli import main

sys.exit(main())
