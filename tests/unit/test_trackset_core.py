"""TrackSet core behaviour on the shared tiny fixture: load, frame times,
time interpolation between frames."""


def test_tiny_trackset_loads(tiny_trackset):
    ts = tiny_trackset
    assert ts.metadata["classes"] == ["person", "vehicle", "other"]
    assert ts.frame_times == [0.0, 0.1]
    assert sorted(ts.frames[0]["objects"]) == ["1", "2"]


def test_objects_at_time_interpolates(tiny_trackset):
    objs = {o.track_id: o for o in tiny_trackset.objects_at_time(0.05)}
    assert set(objs) == {"1", "2"}
    # person box moves 0.10 -> 0.11 in x over the frame; halfway is 0.105
    assert abs(objs["1"].box[0] - 0.105) < 1e-6
    assert tiny_trackset.object_class_name(objs["2"]) == "vehicle"
