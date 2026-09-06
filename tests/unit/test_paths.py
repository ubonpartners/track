import os

import src.paths as paths


def test_defaults_are_the_dev_box_layout(monkeypatch):
    for k in list(os.environ):
        if k.startswith("TRACK_") or k == "AUTOLABEL_PATH":
            monkeypatch.delenv(k)
    assert paths.mldata() == "/mldata"
    assert paths.tier1("antare_bwc") == "/mldata/tracking_original/antare_bwc"
    assert paths.tier2("mot", "annotation") == "/mldata/tracking/mot/annotation"
    assert paths.downloads("other", "JAAD") == "/mldata/downloaded_datasets/other/JAAD"
    assert paths.tracker_config() == "/mldata/config/track/trackers/uc_v11.yaml"
    assert paths.search_yaml() == "/mldata/config/track/search/track_search_v11_mc.yaml"
    assert paths.autolabel_repo().endswith("/autolabel")
    assert set(paths.describe()) >= {"tier1", "tier2", "tracker_config", "search_yaml"}


def test_env_overrides_are_read_at_call_time(monkeypatch):
    monkeypatch.setenv("TRACK_MLDATA", "/elsewhere")
    assert paths.tier1() == "/elsewhere/tracking_original"        # cascades from MLDATA
    monkeypatch.setenv("TRACK_TIER1", "/t1")
    assert paths.tier1("c") == "/t1/c"                             # specific root wins
    monkeypatch.setenv("TRACK_TRACKER_CONFIG", "/cfg/x.yaml")
    assert paths.tracker_config() == "/cfg/x.yaml"
    monkeypatch.setenv("AUTOLABEL_PATH", "/al")
    assert paths.autolabel_repo() == "/al"
    monkeypatch.setenv("TRACK_TIER1", "")                          # empty = unset
    assert paths.tier1() == "/elsewhere/tracking_original"
