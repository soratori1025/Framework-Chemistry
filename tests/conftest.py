"""Shared test configuration.

A non-interactive matplotlib backend is selected before any test module is
imported.  Several tests exercise report and figure code that imports
``pyplot``; on headless machines (and in CI) the default Tk backend aborts with
``_tkinter.TclError: invalid command name "tcl_findLibrary"``.  Pinning ``Agg``
keeps the suite independent of the display environment.
"""

import os

os.environ.setdefault("MPLBACKEND", "Agg")
