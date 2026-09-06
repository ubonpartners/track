"""Evaluation engine (repo_cleanup.md stage 4a): matching -> metrics ->
report, and runner on top. Importing any of them installs the tqdm lock
that the threaded runner relies on (moved from the old track_test.py)."""
import threading

from tqdm.auto import tqdm

tqdm.set_lock(threading.RLock())
