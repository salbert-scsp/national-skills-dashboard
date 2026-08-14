"""
Puts the project root on sys.path for every test in this folder.

WHY THIS FILE EXISTS. The tests moved from the project root into verifications/, and the
modules they test (sortingalgorithmnew, main, json_store...) stayed at the root. pytest
adds the TEST file's directory to sys.path, not the project root, so without this the
imports resolve only when the runner happens to put the working directory on the path --
which `python3.11 -m pytest` does and a bare `pytest` does not. That difference showed up
immediately as nine collection errors from one invocation and a clean pass from the other.

Resolved from __file__ rather than from the working directory, so the suite runs the same
from anywhere:

    pytest                                    # from the project root
    python3.11 -m pytest verifications -q     # the documented form
    pytest /abs/path/to/verifications         # from another directory entirely
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Prepended, not appended: a stdlib or site-packages module sharing a name with one of
# ours would otherwise win, and the failure would look like a mysterious AttributeError
# rather than an import-order problem.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
