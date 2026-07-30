"""Allows `python3 -m reviewbot` as well as `python3 -m reviewbot.cli`."""
from __future__ import annotations

import sys

from reviewbot.cli import main

sys.exit(main())
