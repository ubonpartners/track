"""Self-describing artefact metadata for trainer .pt files and runtime .bin files.

Both .pt checkpoints and exported .bin weights should be auditable: anyone
holding the file should be able to recover the command line, hyperparams,
git revision and host that produced it without consulting an external log.

Two surfaces:

  - PT side: `make_pt_meta(args, hparams, dataset_info)` returns a JSON-able
    dict to drop under `ckpt["_meta"]`. Trainers should write this whenever
    they save a checkpoint.

  - BIN side: `bin_trailer(meta)` returns bytes to APPEND to the runtime
    binary after the existing format. The C loaders call `fread` to a known
    layout and then `fclose` — they don't check EOF — so trailing bytes are
    silently ignored at runtime. Layout:

        u32  magic = 'META' (0x4154454D little-endian)
        u32  json_len
        u8[] json_payload  (UTF-8, padded to 4-byte boundary with 0x00)

To inspect a runtime .bin from the shell:

        python -m bench._artefact_meta --read /path/to/foo.bin
"""
from __future__ import annotations

import argparse
import datetime as _dt
import getpass
import json
import os
import socket
import struct
import subprocess
import sys
from typing import Any, Dict, Iterable, Optional


META_MAGIC = 0x4154454D   # 'META' little-endian
SCHEMA_VERSION = 1


def _git_rev(cwd: Optional[str] = None) -> Dict[str, Any]:
    """Return {commit, dirty, branch} for the repo containing `cwd` (or .)."""
    cwd = cwd or os.path.dirname(os.path.abspath(__file__))
    def _run(args: Iterable[str]) -> Optional[str]:
        try:
            out = subprocess.run(
                list(args), cwd=cwd, check=True,
                capture_output=True, text=True, timeout=5,
            ).stdout.strip()
            return out or None
        except (subprocess.CalledProcessError, FileNotFoundError,
                subprocess.TimeoutExpired):
            return None
    commit = _run(["git", "rev-parse", "HEAD"])
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    dirty_out = _run(["git", "status", "--porcelain"])
    return {
        "commit": commit,
        "branch": branch,
        "dirty": bool(dirty_out) if dirty_out is not None else None,
    }


def make_pt_meta(
    *,
    artefact_kind: str,
    args: Optional[argparse.Namespace] = None,
    hparams: Optional[Dict[str, Any]] = None,
    dataset_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a metadata dict.

    artefact_kind: short tag identifying what this is, e.g.
        "match_cost_two_tower" or "state_head_decoupled".
    args: argparse Namespace from the trainer; we capture ``vars(args)``.
    hparams: hyperparams the trainer derived (usually a subset of args
        plus a few computed bits like input dims).
    dataset_info: paths and counts for the corpus (also captured into .pt).
    """
    argv = list(sys.argv)
    return {
        "schema_version": SCHEMA_VERSION,
        "artefact_kind": artefact_kind,
        "produced_at_utc": _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "host": socket.gethostname(),
        "user": getpass.getuser(),
        "python": sys.version.split()[0],
        "argv": argv,
        "cwd": os.getcwd(),
        "git": _git_rev(),
        "args": vars(args) if args is not None else None,
        "hparams": hparams or {},
        "dataset": dataset_info or {},
    }


def _json_blob(meta: Dict[str, Any]) -> bytes:
    """Encode metadata as compact UTF-8 JSON, accepting non-jsonable junk."""
    return json.dumps(meta, sort_keys=True, default=str,
                      ensure_ascii=False).encode("utf-8")


def bin_trailer(meta: Dict[str, Any]) -> bytes:
    """Bytes to append to a runtime .bin so it is self-describing.

    Layout:  u32 META + u32 json_len + json + 0..3 zero pad bytes.
    """
    blob = _json_blob(meta)
    pad = (-len(blob)) & 3
    return struct.pack("<II", META_MAGIC, len(blob)) + blob + (b"\x00" * pad)


def write_meta_sidecar(meta: Dict[str, Any], path: str) -> None:
    """Drop a sibling .meta.json next to a runtime .bin for grep-friendliness."""
    sidecar = path + ".meta.json"
    with open(sidecar, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True, default=str)
    print(f"wrote {sidecar}")


def _read_trailer(path: str) -> Optional[Dict[str, Any]]:
    """Walk the file from EOF backwards looking for META magic. Returns None
    if no metadata trailer is present."""
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        sz = f.tell()
        # The trailer is at most ~4 KB in practice; scan the last 64 KB.
        scan = min(sz, 65536)
        f.seek(sz - scan, os.SEEK_SET)
        tail = f.read(scan)
    magic_le = struct.pack("<I", META_MAGIC)
    idx = tail.rfind(magic_le)
    if idx < 0:
        return None
    if idx + 8 > len(tail):
        return None
    json_len = struct.unpack_from("<I", tail, idx + 4)[0]
    start = idx + 8
    if start + json_len > len(tail):
        return None
    try:
        return json.loads(tail[start:start + json_len].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None


def _main():
    ap = argparse.ArgumentParser(prog="bench._artefact_meta")
    ap.add_argument("--read", required=True,
                    help="path to a .bin produced with a metadata trailer")
    args = ap.parse_args()
    meta = _read_trailer(args.read)
    if meta is None:
        sys.exit(f"no metadata trailer found in {args.read}")
    json.dump(meta, sys.stdout, indent=2, sort_keys=True, default=str)
    print()


if __name__ == "__main__":
    _main()
