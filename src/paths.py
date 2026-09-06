"""Every filesystem root the package uses, in one place (repo_cleanup.md
stage 2).

Each root is read from an environment variable at CALL time, with the
dev-box value as the default, so nothing changes on this machine and a
different layout is one `export` away:

    TRACK_MLDATA          /mldata                      parent of everything below
    TRACK_DOWNLOADS       $TRACK_MLDATA/downloaded_datasets   tier 0
    TRACK_TIER1           $TRACK_MLDATA/tracking_original     tier 1 (canonical import)
    TRACK_TIER2           $TRACK_MLDATA/tracking              tier 2 (eval copies)
    TRACK_RESULTS         $TRACK_MLDATA/results
    TRACK_CONFIG_DIR      $TRACK_MLDATA/config/track
    TRACK_TRACKER_CONFIG  $TRACK_CONFIG_DIR/trackers/uc_v11.yaml    the production tracker config
    TRACK_SEARCH_YAML     $TRACK_CONFIG_DIR/search/track_search_v11_mc.yaml   THE objective
    TRACK_AUTOLABEL_CACHE $TRACK_MLDATA/autolabel_cache
    TRACK_VIDEO           $TRACK_MLDATA/video
    AUTOLABEL_PATH        <sibling dir of this repo>/autolabel   the autolabel checkout

Rules: no other module may contain a literal `/mldata`, `/home/` or `~/`
path (tests/test_no_literal_paths.py enforces it). Function DEFAULT
arguments that are paths are `None` and resolved inside the function, so
the environment is consulted when the function runs, not when the module
is imported. `python track.py --paths` prints the resolved values.
"""
import os


def _env(name, default):
    v = os.environ.get(name)
    if not v:
        return default
    v = v.rstrip("/")            # "/t1/" must not become "/t1//clip" in string builds
    return v or "/"


def _join(root, parts):
    return os.path.join(root, *parts) if parts else root


def mldata(*parts):
    return _join(_env("TRACK_MLDATA", "/mldata"), parts)


def downloads(*parts):
    """tier 0: raw acquisitions, never modified."""
    return _join(_env("TRACK_DOWNLOADS", mldata("downloaded_datasets")), parts)


def tier1(*parts):
    """tier 1: /mldata/tracking_original — canonical import, autolabel's input."""
    return _join(_env("TRACK_TIER1", mldata("tracking_original")), parts)


def tier2(*parts):
    """tier 2: /mldata/tracking — the derived eval-spec copies track.py reads."""
    return _join(_env("TRACK_TIER2", mldata("tracking")), parts)


def results(*parts):
    return _join(_env("TRACK_RESULTS", mldata("results")), parts)


def config_dir(*parts):
    return _join(_env("TRACK_CONFIG_DIR", mldata("config", "track")), parts)


def tracker_config():
    """The production tracker config (shared with every deployed box)."""
    return _env("TRACK_TRACKER_CONFIG", config_dir("trackers", "uc_v11.yaml"))


def search_yaml():
    """THE objective config: search and eval both read this one file."""
    return _env("TRACK_SEARCH_YAML", config_dir("search", "track_search_v11_mc.yaml"))


def autolabel_cache(*parts):
    return _join(_env("TRACK_AUTOLABEL_CACHE", mldata("autolabel_cache")), parts)


def video(*parts):
    """Raw footage that is not a dataset drop (raw_movies source)."""
    return _join(_env("TRACK_VIDEO", mldata("video")), parts)


def repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def autolabel_sibling():
    """Where the autolabel checkout lives by convention: beside this repo."""
    return os.path.join(os.path.dirname(repo_root()), "autolabel")


def autolabel_repo():
    """The autolabel checkout: $AUTOLABEL_PATH, else the sibling dir."""
    return _env("AUTOLABEL_PATH", autolabel_sibling())


def describe():
    """{name: resolved path} for `track.py --paths`."""
    return {
        "mldata": mldata(), "downloads": downloads(), "tier1": tier1(),
        "tier2": tier2(), "results": results(), "config_dir": config_dir(),
        "tracker_config": tracker_config(), "search_yaml": search_yaml(),
        "autolabel_cache": autolabel_cache(), "video": video(),
        "autolabel_repo": autolabel_repo(), "repo_root": repo_root(),
    }
