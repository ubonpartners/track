"""Native-annotation parsers, one module per source format, each exposing
`read(...) -> TrackSet` (repo_cleanup.md stage 3). Nothing here writes
files; the tier-0 -> tier-1 converters in src/trackset_import.py call
these and do the copying.

`load(path)` is the extension dispatch that TrackSet(path) used to do
for MOT seqinfo.ini and MEVA .geom.yml paths; TrackSet(path) itself now
only reads track's own formats (UBTRK2, json/yaml).
"""
import stuff

from src.trackset import TrackSet


def load(path, **kw):
    """Open any supported annotation file as a TrackSet."""
    if path.endswith(".ini"):
        from src.formats import mot
        return mot.read(path)
    if path.endswith(".geom.yml") or path.endswith(".geom.yaml"):
        from src.formats import meva
        return meva.read(path, **kw)
    if (path.endswith(".ubtrk2") or stuff.is_ubtrk2_file(path)
            or path.endswith((".yml", ".yaml", ".json"))):
        return TrackSet(path, **kw)
    raise ValueError(f"no parser for {path}")
