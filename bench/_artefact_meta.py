"""Self-describing artefact metadata for trainer .pt files, runtime .bin
files, and .npz corpora.

Every artefact in the train pipeline should be reproducible from itself:
anyone holding the file should be able to recover the exact command
line, training args, hyperparams, git revision, host, and a free-form
comment without consulting an external log. The 2026-05-11 corpus-drift
investigation was a direct consequence of state_corpus_v18 lacking this
metadata — we could not recover *how* it was built (which extractor,
which pair-log source, which seed) and lost a day chasing the gap.
This module is the mandatory provenance carrier; callers must use it.

Three surfaces:

  - PT side: `make_pt_meta(...)` returns a JSON-able dict to drop under
    `ckpt["_meta"]`. Trainers write this when they save a checkpoint.

  - BIN side: `bin_trailer(meta)` returns bytes to APPEND to the runtime
    binary after the existing format. The C loaders call `fread` to a known
    layout and then `fclose` — they don't check EOF — so trailing bytes are
    silently ignored at runtime. Layout:

        u32  magic = 'META' (0x4154454D little-endian)
        u32  json_len
        u8[] json_payload  (UTF-8, padded to 4-byte boundary with 0x00)

  - NPZ side: `save_npz_with_meta(path, arrays, meta)` writes a .npz that
    contains both the data arrays and a `_meta` entry holding the same
    JSON dict (encoded as a 0-d numpy string). `read_npz_meta(path)` pulls
    it back out. `require_npz_meta(arr)` raises on missing meta — corpus
    loaders should call this to refuse unprovenanced inputs.

Inspect any artefact from the shell:

        python -m bench._artefact_meta --read /path/to/foo.{bin,pt,npz}
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
    comment: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a metadata dict.

    artefact_kind: short tag identifying what this is, e.g.
        "match_cost_two_tower" or "state_head_decoupled".
    args: argparse Namespace from the trainer; we capture ``vars(args)``.
    hparams: hyperparams the trainer derived (usually a subset of args
        plus a few computed bits like input dims).
    dataset_info: paths and counts for the corpus (also captured into .pt).
    comment: free-form human note. Surface a --comment CLI flag on every
        trainer / corpus-build entry-point and pass it through.
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
        "comment": comment,
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


# ---------- NPZ provenance --------------------------------------------------

NPZ_META_KEY = "_meta"


def save_npz_with_meta(path: str, meta: Dict[str, Any],
                       **arrays: Any) -> None:
    """Write `arrays` to `path` as a .npz alongside a `_meta` JSON blob.

    Refuses to write if `meta` doesn't carry the minimum reproducibility
    keys: argv, args, git, produced_at_utc, artefact_kind. Use
    `make_pt_meta(...)` to construct a valid dict.
    """
    import numpy as np
    _check_meta_complete(meta, where=f"save_npz_with_meta({path})")
    if NPZ_META_KEY in arrays:
        raise ValueError(
            f"reserved array name {NPZ_META_KEY!r} clashes with the "
            "metadata slot — rename your data array")
    payload = _json_blob(meta)
    arrays = dict(arrays)
    arrays[NPZ_META_KEY] = np.frombuffer(payload, dtype=np.uint8)
    np.savez(path, **arrays)


def read_npz_meta(path: str) -> Optional[Dict[str, Any]]:
    """Return the embedded metadata dict, or None if absent."""
    import numpy as np
    with np.load(path, allow_pickle=True) as z:
        if NPZ_META_KEY not in z.files:
            return None
        raw = z[NPZ_META_KEY]
    try:
        return json.loads(bytes(raw.tobytes()).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None


def require_npz_meta(path: str) -> Dict[str, Any]:
    """Like read_npz_meta but raises a clear error on missing/corrupt meta.

    Corpus-loading code should call this so unprovenanced .npz files are
    rejected at the boundary. Pre-existing corpora without meta need to be
    either regenerated or wrapped (see attach_npz_meta below)."""
    meta = read_npz_meta(path)
    if meta is None:
        raise ValueError(
            f"{path}: no _meta entry — refuse to load unprovenanced corpus. "
            f"Regenerate via the current build command, or migrate the file "
            f"with bench._artefact_meta --attach-meta <path> "
            f"--comment <note>.")
    return meta


def attach_npz_meta(path: str, meta: Dict[str, Any], *,
                    out_path: Optional[str] = None) -> str:
    """Rewrite `path` (or write to `out_path`) with `meta` added/overwritten.

    Used to retrofit legacy corpora once their build provenance has been
    reconstructed by hand — set the comment to record that the meta was
    reconstructed and not captured at build time."""
    import numpy as np
    _check_meta_complete(meta, where=f"attach_npz_meta({path})")
    out_path = out_path or path
    with np.load(path, allow_pickle=True) as z:
        kept = {k: z[k] for k in z.files if k != NPZ_META_KEY}
    save_npz_with_meta(out_path, meta, **kept)
    return out_path


def _check_meta_complete(meta: Dict[str, Any], *, where: str) -> None:
    required = ("argv", "args", "git", "produced_at_utc", "artefact_kind")
    missing = [k for k in required if meta.get(k) is None]
    if missing:
        raise ValueError(
            f"{where}: metadata missing required keys {missing!r}. Build "
            f"via make_pt_meta(...) so argv / git / args are filled.")


# ---------- inspect CLI -----------------------------------------------------

def _inspect_pt(path: str) -> Optional[Dict[str, Any]]:
    try:
        import torch
    except ImportError:
        return None
    try:
        ck = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    return ck.get("_meta") if isinstance(ck, dict) else None


def _main():
    ap = argparse.ArgumentParser(prog="bench._artefact_meta")
    ap.add_argument("--read", help="path to a .bin / .pt / .npz artefact")
    ap.add_argument("--attach-meta",
                    help="legacy-retrofit: path to a .npz lacking _meta to "
                         "rewrite with metadata. Requires --comment.")
    ap.add_argument("--comment", default=None,
                    help="free-form note carried in the attached metadata")
    args = ap.parse_args()
    if args.attach_meta:
        if not args.comment:
            sys.exit("--attach-meta requires --comment so the retrofit is auditable")
        meta = make_pt_meta(
            artefact_kind="legacy_corpus_retrofit",
            comment=args.comment,
        )
        # We can't recover argv/args of the original build; populate the
        # slots that _check_meta_complete needs with placeholders that make
        # it obvious the provenance is reconstructed.
        meta["args"] = meta.get("args") or {"_legacy_retrofit": True}
        meta["argv"] = meta.get("argv") or ["legacy_retrofit"]
        out = attach_npz_meta(args.attach_meta, meta)
        print(f"attached meta to {out}")
        return
    if not args.read:
        sys.exit("--read or --attach-meta is required")
    path = args.read
    meta = None
    if path.endswith(".npz"):
        meta = read_npz_meta(path)
    elif path.endswith(".pt"):
        meta = _inspect_pt(path)
    else:
        meta = _read_trailer(path)
    if meta is None:
        sys.exit(f"no metadata found in {path}")
    json.dump(meta, sys.stdout, indent=2, sort_keys=True, default=str)
    print()


if __name__ == "__main__":
    _main()
