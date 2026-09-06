"""src/cli.py verbs and the track.py flag translation (repo_cleanup.md
stage 6). Every old flag combination the plan lists must map to a verb,
and every verb must reach the same entry point with the same arguments.
No tracker, no GPU: the entry points are monkeypatched."""
import pytest

import src.cli as cli
import track as shim


def _ns(argv):
    return cli.build_parser().parse_args(argv)


def test_translate_every_old_form():
    t = shim.translate
    assert t(["--paths"]) == ["paths"]
    assert t(["--eval"]) == ["--logging", "info", "eval", "--split", "both", "--permissive", "auto"]
    assert t(["--eval", "x.yaml", "--eval-split", "val", "--eval-permissive", "on", "--pm", "2",
              "--results-location", "R", "--tracker-config", "T"]) == \
        ["--logging", "info", "--pm", "2", "eval", "x.yaml", "--split", "val", "--permissive", "on",
         "--results-location", "R", "--tracker-config", "T"]
    assert t(["--search", "s.yaml"]) == ["--logging", "info", "search", "s.yaml"]
    assert t(["--test", "t.yaml", "--pm", "1"]) == ["--logging", "info", "--pm", "1", "test", "t.yaml"]
    assert t(["--track", "--trackset", "g.json", "--config", "c.yaml", "--display", "--output", "o.mp4",
              "--save-trackset", "r.ubtrk2", "--proxy", "1.2.3.4:5"]) == \
        ["--logging", "info", "track", "g.json", "--config", "c.yaml", "--display", "--output", "o.mp4",
         "--save-trackset", "r.ubtrk2", "--proxy", "1.2.3.4:5"]
    assert t(["--view", "--trackset", "run.ubtrk2"]) == ["--logging", "info", "view", "run.ubtrk2"]
    assert t(["--compare", "cmp.yaml", "--trackset", "g.json"]) == ["--logging", "info", "compare", "cmp.yaml"]
    for name in ("mot", "jaad", "otw", "meva", "cevo"):
        assert t([f"--{name}"]) == ["--logging", "info", "import", name]
    assert t(["--personpath22", "--personpath22-amodal"]) == ["--logging", "info", "import", "personpath22", "--amodal"]
    assert t(["--logging", "debug:file", "--eval"])[:2] == ["--logging", "debug:file"]
    assert t([]) is None                                   # "No option specified"
    # old dispatch precedence: an importer flag wins over --eval, --track over --compare
    assert t(["--mot", "--eval"]) == ["--logging", "info", "import", "mot"]
    assert t(["--track", "--compare", "c.yaml"])[2] == "track"


def test_every_translation_parses():
    for argv in (["--eval"], ["--eval", "x.yaml", "--eval-split", "val"], ["--search", "s"], ["--test", "t"],
                 ["--track", "--trackset", "g", "--display"], ["--view", "--trackset", "v"],
                 ["--compare", "c"], ["--mot"], ["--personpath22", "--personpath22-amodal"], ["--paths"]):
        _ns(shim.translate(argv))


def test_eval_dispatch(monkeypatch, tmp_path, capsys):
    calls = {}
    import src.track_search as ts_
    monkeypatch.setattr(ts_, "eval_track", lambda *a, **k: calls.update(args=a, kw=k))
    monkeypatch.setattr(cli.stuff, "rmdir", lambda *a: None)
    monkeypatch.setattr(cli.stuff, "makedir", lambda *a: None)
    monkeypatch.setattr(cli.stuff, "configure_root_logger", lambda *a, **k: None)
    import src.eval.runner as runner
    old = runner.PM_OVERRIDE
    assert cli.main(["--pm", "3", "eval", "--split", "val", "--permissive", "off",
                     "--results-location", "R", "--tracker-config", "T"]) == 0
    assert calls["args"] == (None,)
    assert calls["kw"] == {"split": "val", "convention_permissive": False,
                           "results_location": "R", "tracker_config": "T"}
    assert runner.PM_OVERRIDE == 3
    runner.PM_OVERRIDE = old
    assert cli.main(["eval", "probe.yaml"]) == 0 and calls["args"] == ("probe.yaml",)


def test_other_dispatches(monkeypatch):
    for f in ("rmdir", "makedir", "configure_root_logger"):
        monkeypatch.setattr(cli.stuff, f, lambda *a, **k: None)
    seen = []
    import src.track_search as ts_
    import src.eval.runner as runner
    import src.core.display as disp
    monkeypatch.setattr(ts_, "search_track", lambda y: seen.append(("search", y)))
    monkeypatch.setattr(runner, "track_test", lambda y: seen.append(("test", y)))
    monkeypatch.setattr(disp, "display_trackset", lambda **k: seen.append(("view", k["trackset_gt"])))
    monkeypatch.setattr(cli, "test_track", lambda *a, **k: seen.append(("track", a, k)))
    monkeypatch.setattr(cli, "compare_track", lambda *a, **k: seen.append(("compare", a, k)))
    monkeypatch.setitem(cli.IMPORTERS, "mot", lambda **k: seen.append(("import", "mot", k)))
    cli.main(["search", "s.yaml"]); cli.main(["test", "t.yaml"]); cli.main(["view", "v.json"])
    cli.main(["track", "g.json", "--config", "c.yaml", "--display", "--save-trackset", "r"])
    cli.main(["compare", "c.yaml", "--no-display"]); cli.main(["import", "mot"])
    assert seen == [("search", "s.yaml"), ("test", "t.yaml"), ("view", "v.json"),
                    ("track", ("g.json", "c.yaml"), {"display": True, "output": None, "proxy": None, "save_trackset": "r"}),
                    ("compare", (None,), {"compare_config": "c.yaml", "display": False}),
                    ("import", "mot", {"amodal": False})]


def test_corpus_dispatch(monkeypatch):
    for f in ("rmdir", "makedir", "configure_root_logger"):
        monkeypatch.setattr(cli.stuff, f, lambda *a, **k: None)
    import src.corpus.manifest as manifest
    import src.corpus.derive as derive
    seen = []
    monkeypatch.setattr(manifest, "build", lambda c: seen.append(("build", c)))
    monkeypatch.setattr(manifest, "verify", lambda c: seen.append(("verify", c)) or False)
    monkeypatch.setattr(derive, "derive_tracking", lambda c, **k: seen.append(("derive", c, k)))
    monkeypatch.setattr(derive, "check_tracking", lambda c, **k: seen.append(("check", c, k)))
    assert cli.main(["corpus", "build", "a", "b"]) == 0
    assert cli.main(["corpus", "verify", "a"]) == 1                       # False -> exit 1, as the old CLI
    assert cli.main(["corpus", "derive", "a", "--hint", "bodycam", "--divisor", "1", "--max-seconds", "120"]) == 0
    assert cli.main(["corpus", "check", "a", "--purge-legacy"]) == 0
    assert seen == [("build", "a"), ("build", "b"), ("verify", "a"),
                    ("derive", "a", {"hint": "bodycam", "max_seconds": 120.0, "divisor": 1}),
                    ("check", "a", {"purge_legacy": True})]


def test_paths_verb(capsys):
    assert cli.main(["paths"]) == 0
    assert "tier1" in capsys.readouterr().out


def test_unknown_verb():
    with pytest.raises(SystemExit):
        cli.main(["frobnicate"])
