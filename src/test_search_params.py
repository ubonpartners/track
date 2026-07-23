# §15 search-parameter addressing: dotted paths, (hint:x) variant
# create-on-write, initial-value seeding from base, split_hints
# expansion — and the §7 protect guard.

import pytest

import src.track_search as tsr


def _cfg():
    return {
        "conf_thr": 0.05,
        "motiontrack": {"mad_delta": 60.0, "alpha": 0.2, "max_width": 640},
        "utrack": {"kf_weight": 0.4, "alpha": 0.9},
        "thumbnail_stream": {"max_width": 320},
    }


def test_bare_name_unique_still_works():
    c = _cfg()
    tsr._set_nested_param(c, "kf_weight", 0.7)
    assert c["utrack"]["kf_weight"] == 0.7


def test_bare_name_ambiguous_asserts():
    with pytest.raises(AssertionError, match="ambiguous"):
        tsr._set_nested_param(_cfg(), "alpha", 0.5)
    with pytest.raises(AssertionError, match="ambiguous"):
        tsr._set_nested_param(_cfg(), "max_width", 100)


def test_bare_name_missing_asserts():
    with pytest.raises(AssertionError, match="Can't find"):
        tsr._set_nested_param(_cfg(), "nonexistent", 1)


def test_dotted_path_reaches_ambiguous_keys():
    c = _cfg()
    tsr._set_nested_param(c, "motiontrack.alpha", 0.5)
    tsr._set_nested_param(c, "utrack.alpha", 0.1)
    tsr._set_nested_param(c, "motiontrack.max_width", 512)
    assert c["motiontrack"]["alpha"] == 0.5
    assert c["utrack"]["alpha"] == 0.1
    assert c["motiontrack"]["max_width"] == 512
    assert c["thumbnail_stream"]["max_width"] == 320


def test_dotted_path_missing_plain_asserts():
    with pytest.raises(AssertionError, match="Can't find"):
        tsr._set_nested_param(_cfg(), "motiontrack.nonexistent", 1)
    with pytest.raises(AssertionError, match="Can't find"):
        tsr._set_nested_param(_cfg(), "roi_scan.min_age_lo", 1)  # no section


def test_variant_path_creates_block():
    c = _cfg()
    tsr._set_nested_param(c, "utrack(hint:bodycam).kf_weight", 0.9)
    assert c["utrack(hint:bodycam)"] == {"kf_weight": 0.9}
    assert c["utrack"]["kf_weight"] == 0.4  # base untouched
    # section absent from base entirely: still created for a variant
    tsr._set_nested_param(c, "roi_scan(hint:wide).min_age_lo", 2.0)
    assert c["roi_scan(hint:wide)"]["min_age_lo"] == 2.0
    # flat-key variant
    tsr._set_nested_param(c, "conf_thr(hint:bodycam)", 0.1)
    assert c["conf_thr(hint:bodycam)"] == 0.1


def test_initial_seeds_variant_from_base():
    names = ["utrack.kf_weight", "utrack(hint:bodycam).kf_weight",
             "motiontrack.mad_delta", "kf_weight"]
    initial = [None, None, None, None]
    tsr._update_initial_parameters(names, initial, _cfg(), None, "test")
    assert initial[0] == 0.4          # dotted path
    assert initial[1] == 0.4          # variant absent -> base value
    assert initial[2] == 60.0
    assert initial[3] == 0.4          # bare name walk unchanged


def test_split_hints_expansion():
    sp = {
        "utrack.kf_weight": {"min": 0, "max": 1, "step": 0.1,
                             "split_hints": ["bodycam", "dashcam"]},
        "conf_thr": {"min": 0, "max": 1, "step": 0.01,
                     "split_hints": ["bodycam"]},
        "motiontrack.mad_delta": {"min": 0, "max": 500, "step": 1},
    }
    out = tsr._expand_split_hints(sp)
    assert set(out) == {
        "utrack.kf_weight",
        "utrack(hint:bodycam).kf_weight",
        "utrack(hint:dashcam).kf_weight",
        "conf_thr",
        "conf_thr(hint:bodycam)",
        "motiontrack.mad_delta",
    }
    # spec copied, split_hints stripped everywhere
    for spec in out.values():
        assert "split_hints" not in spec
    assert out["utrack(hint:dashcam).kf_weight"]["step"] == 0.1


def _protect_results(fitness_cctv):
    return [
        {"params": {"test_key": "search_config", "ds_key": "__ovrcctv"},
         "result": {"fitness": fitness_cctv}},
        {"params": {"test_key": "search_config", "ds_key": "_overall"},
         "result": {"fitness": 0.9}},
    ]


def test_protect_guard():
    config = {"result_test_opt_key": "search_config",
              "protect": [{"group": "cctv", "param": "fitness",
                           "floor": 0.5}]}
    assert tsr._check_protect(config, _protect_results(0.6)) is None
    rule = tsr._check_protect(config, _protect_results(0.4))
    assert rule is not None and rule["group"] == "cctv"
    # missing rollup is a loud misconfig, not a silent pass
    with pytest.raises(AssertionError, match="__ovrmissing"):
        tsr._check_protect(
            {"result_test_opt_key": "search_config",
             "protect": [{"group": "missing", "param": "fitness",
                          "floor": 0.5}]},
            _protect_results(0.6))


def test_no_protect_is_noop():
    assert tsr._check_protect({"result_test_opt_key": "x"}, []) is None


def test_flat_variant_lands_beside_its_base():
    # kf_weight lives under utrack: — its (hint:x) variant must be written
    # INTO utrack (the C side resolves variants against siblings), never at
    # top level where nothing reads it.
    c = _cfg()
    tsr._set_nested_param(c, "kf_weight(hint:bodycam)", 0.9)
    assert c["utrack"]["kf_weight(hint:bodycam)"] == 0.9
    assert "kf_weight(hint:bodycam)" not in c   # NOT at top level
    # a genuinely top-level key keeps top-level variant placement
    tsr._set_nested_param(c, "conf_thr(hint:bodycam)", 0.1)
    assert c["conf_thr(hint:bodycam)"] == 0.1


def test_flat_variant_seeds_from_section_scoped_base():
    # Regression: vbox_expand(hint:bodycam) failed to seed because the base
    # lookup only checked the config TOP LEVEL — the base lives in utrack:.
    names = ["vbox_expand(hint:bodycam)"]
    initial = [None]
    tsr._update_initial_parameters(
        names, initial, {"utrack": {"vbox_expand": 0.25}}, None, "test")
    assert initial == [0.25]
