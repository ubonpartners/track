"""Batch converters (tier 0 -> tier 1): one convert_<name>() per corpus,
walking the downloaded drop and writing trackset JSON (+ mp4) pairs into
the tier-1 corpus dir, plus the autolabel-folder importer and the dataset
reduction helpers. Parsers live in src/formats/.

Moved verbatim from src/trackset_import.py (repo_cleanup.md stage 4b).
"""
import json
import os

import cv2
import stuff

from src.corpus.media import _native_fps, _remux_to_mp4, _transcode_h264, _video_codec
from src.corpus.migrations import fix_cevo25_vfr_times
from src.formats import chirla as fmt_chirla, jaad as fmt_jaad, meva as fmt_meva, mot as fmt_mot, otw as fmt_otw, personpath22 as fmt_personpath22, roundabouthd as fmt_roundabouthd, uvg_vcm as fmt_uvg_vcm
import src.paths as paths
import src.trackset as trackset


def convert_mot(lite=True):
    output_folder=paths.tier1("mot")
    folders=[paths.downloads("other", "MOT20", "train"),
             paths.downloads("other", "MOT17", "train")]
    stuff.makedir(output_folder+"/annotation/")
    stuff.makedir(output_folder+"/video/")
    for f in folders:
        seqs=os.listdir(f)
        for s in seqs:
            input_path=f+"/"+s+"/seqinfo.ini"
            output_path=output_folder+"/annotation/"+s+".json"
            output_video_path=output_folder+"/video/"+s+".mp4"
            print("Processing",f,s,"....")
            ts=fmt_mot.read(input_path)
            ts.export_yaml(output_path, output_video_path)
    if lite:
        # tier 2 eval-spec derive; MOT mixes static and moving cameras,
        # so the moving MOT17 sequences override the corpus-level hint
        from src.corpus_manifest import derive_tracking
        derive_tracking("mot", hint="static",
                        hint_overrides={"MOT17-05": "bodycam",
                                        "MOT17-10": "bodycam",
                                        "MOT17-11": "bodycam",
                                        "MOT17-13": "bodycam"})


def convert_personpath22(src_root=None, output_folder=None,
                         anno_variant="visible", lite=True):
    """Convert PersonPath22 (gluoncv-motion format) into MOT-equivalent
    JSON+mp4 pairs under output_folder/{annotation,video}/.

    anno_variant: "visible" or "amodal" — picks anno_visible_2022 or anno_amodal_2022.
    The src_root default matches the layout produced by download.py
    (which nests under <root>/dataset/personpath22/{annotation,raw_data}).
    """
    src_root = src_root or paths.downloads("other", "personpath22", "dataset", "personpath22")
    output_folder = output_folder or paths.tier1("personpath22")
    variant_stem = {
        "visible": "anno_visible_2022",
        "amodal":  "anno_amodal_2022",
    }[anno_variant]
    anno_index_path = os.path.join(src_root, "annotation", variant_stem + ".json")
    anno_per_sample_dir = os.path.join(src_root, "annotation", variant_stem)
    videos_root = os.path.join(src_root, "raw_data")

    stuff.makedir(output_folder + "/annotation/")
    stuff.makedir(output_folder + "/video/")

    print(f"Loading {anno_index_path} ...")
    with open(anno_index_path, 'r') as f:
        all_annos = json.load(f)
    samples = all_annos.get("samples", {})
    print(f"Found {len(samples)} samples in PersonPath22 ({anno_variant})")

    skipped_missing = 0
    for uid, index_sample in samples.items():
        stem = uid[:-4] if uid.endswith(".mp4") else uid
        video_path = os.path.join(videos_root, stem + ".mp4")
        if not os.path.isfile(video_path):
            print(f"  skipping {uid}: video not found at {video_path}")
            skipped_missing += 1
            continue
        out_anno = output_folder + "/annotation/" + stem + ".json"
        out_video = output_folder + "/video/" + stem + ".mp4"
        if os.path.isfile(out_anno) and os.path.isfile(out_video):
            continue

        # Per-sample annotations live in a sibling directory; the index file
        # only carries metadata + sample_file=True for samples that defer.
        if index_sample.get("sample_file"):
            per_sample_path = os.path.join(anno_per_sample_dir, uid + ".json")
            with open(per_sample_path, 'r') as f:
                sample = json.load(f)
        else:
            sample = index_sample

        print(f"Processing {uid}...")
        ts = trackset.TrackSet()
        fmt_personpath22.read_into(ts, uid, sample, video_path)
        if os.path.isfile(out_video):
            # Skip the mp4 copy; just rewrite the JSON pointing at the existing copy.
            ts.metadata["original_video"] = out_video
            ts.export_yaml(out_anno, output_video=None)
        else:
            ts.export_yaml(out_anno, out_video)

    if skipped_missing:
        print(f"Done. Skipped {skipped_missing} samples with no source video.")
    if lite:
        # tier 2 eval-spec derive (handheld/moving-camera footage)
        from src.corpus_manifest import derive_tracking
        derive_tracking("personpath22", hint="bodycam")


def convert_jaad(src_root=None, output_folder=None, lite=True):
    """Convert JAAD XML + mp4 clips into MOT-like JSON+mp4 tracksets."""
    src_root = src_root or paths.downloads("other", "JAAD")
    output_folder = output_folder or paths.tier1("jaad")
    annotations_dir = os.path.join(src_root, "annotations")
    videos_dir = os.path.join(src_root, "JAAD_clips")

    stuff.makedir(output_folder + "/annotation/")
    stuff.makedir(output_folder + "/video/")

    xml_files = sorted([f for f in os.listdir(annotations_dir) if f.endswith(".xml")])
    print(f"Found {len(xml_files)} JAAD annotation files")

    skipped_missing_video = 0
    skipped_failed = 0
    for xml_name in xml_files:
        stem = xml_name[:-4]
        annotation_xml_path = os.path.join(annotations_dir, xml_name)
        video_path = os.path.join(videos_dir, stem + ".mp4")
        if not os.path.isfile(video_path):
            print(f"  skipping {xml_name}: video not found at {video_path}")
            skipped_missing_video += 1
            continue

        out_anno = output_folder + "/annotation/" + stem + ".json"
        out_video = output_folder + "/video/" + stem + ".mp4"
        if os.path.isfile(out_anno) and os.path.isfile(out_video):
            continue

        print(f"Processing {xml_name}...")
        ts = trackset.TrackSet()
        try:
            fmt_jaad.read_into(ts, annotation_xml_path, video_path)
        except Exception as e:
            print(f"  failed {xml_name}: {e}")
            skipped_failed += 1
            continue

        if os.path.isfile(out_video):
            # Keep existing copied mp4; just rewrite annotation JSON.
            ts.metadata["original_video"] = out_video
            ts.export_yaml(out_anno, output_video=None)
        else:
            ts.export_yaml(out_anno, out_video)

    print(f"Done. missing_video={skipped_missing_video} failed={skipped_failed}")
    if lite:
        # tier 2 eval-spec derive (dashcam = moving camera)
        from src.corpus_manifest import derive_tracking
        derive_tracking("jaad", hint="bodycam")


def _convert_meva_clip(args):
    """Convert one MEVA clip (worker for convert_meva's pool).
    Returns (status, stem) with status in done/missing/failed/empty."""
    geom_path, output_folder = args
    stem = os.path.basename(geom_path)
    for suf in (".geom.yml", ".geom.yaml"):
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
            break
    out_anno = output_folder + "/annotation/" + stem + ".json"
    out_video = output_folder + "/video/" + stem + ".mp4"
    ts = trackset.TrackSet()
    try:
        fmt_meva.read_into(ts, geom_path)
        if "original_video" not in ts.metadata:
            return ("missing", stem)
        if len(ts.frames) == 0:
            return ("empty", stem)
        if not os.path.isfile(out_video):
            _remux_to_mp4(ts.metadata["original_video"], out_video)
        ts.metadata["original_video"] = out_video
        ts.export_yaml(out_anno)
    except Exception as e:
        print(f"  failed {stem}: {e}")
        return ("failed", stem)
    return ("done", stem)


def convert_chirla(src_root=None, output_folder=None, lite=True):
    """Convert CHIRLA camera clips into trackset JSON + mp4 pairs.

    Output stems: chirla_<seq>_<camera>_<timestamp> with ':' -> '-'
    (colons break too many downstream tools). mpeg4-in-avi sources are
    remuxed to mp4 (transcode fallback on broken pts, see
    _remux_to_mp4). Existing outputs are not redone.
    """
    src_root = src_root or paths.downloads("other", "chirla")
    output_folder = output_folder or paths.tier1("chirla")
    stuff.makedir(output_folder + "/annotation/")
    stuff.makedir(output_folder + "/video/")

    pairs = []
    anno_root = os.path.join(src_root, "annotations")
    for seq in sorted(os.listdir(anno_root)):
        seq_dir = os.path.join(anno_root, seq)
        if not os.path.isdir(seq_dir):
            continue
        for f in sorted(os.listdir(seq_dir)):
            if not f.endswith(".json"):
                continue
            video = os.path.join(src_root, "videos", seq, f[:-5] + ".avi")
            pairs.append((seq, os.path.join(seq_dir, f), video))
    print(f"Found {len(pairs)} CHIRLA annotation files")

    counts = {"done": 0, "already": 0, "missing": 0, "failed": 0,
              "empty_gt": 0}
    for seq, anno_path, video_path in pairs:
        stem = ("chirla_" + seq + "_" +
                os.path.basename(anno_path)[:-5].replace(":", "-"))
        out_anno = output_folder + "/annotation/" + stem + ".json"
        out_video = output_folder + "/video/" + stem + ".mp4"
        if os.path.isfile(out_anno) and os.path.isfile(out_video):
            counts["already"] += 1
            continue
        if not os.path.isfile(video_path):
            print(f"  missing video for {anno_path}")
            counts["missing"] += 1
            continue
        ts = trackset.TrackSet()
        try:
            fmt_chirla.read_into(ts, anno_path, video_path)
            if not any(f["objects"] for f in ts.frames):
                # 12/70 cameras were never visited; keep them out of the
                # corpus rather than shipping all-empty "GT"
                counts["empty_gt"] += 1
                continue
            if not os.path.isfile(out_video):
                _remux_to_mp4(video_path, out_video)
            ts.metadata["original_video"] = out_video
            ts.export_yaml(out_anno)
        except Exception as e:
            print(f"  failed {stem}: {e}")
            counts["failed"] += 1
            continue
        counts["done"] += 1
        if counts["done"] % 10 == 0:
            print(f"  {counts}")
    print(f"chirla done. {counts}")
    if lite:
        # tier 2 eval-spec derive (fixed indoor cameras)
        from src.corpus_manifest import derive_tracking
        derive_tracking("chirla", hint="static")


def convert_roundabouthd(src_root=None, output_folder=None, lite=True):
    """Convert the 4 RoundaboutHD cameras into trackset JSON + mp4.

    Source videos are 4K MPEG-4 Part 2 (Simple Profile) — no desktop
    hardware decode at that resolution, unplayable in practice — so
    they are transcoded to h264 (NVENC when available, else x264,
    crf/cq 22, 2s keyframes) rather than copied.
    """
    src_root = src_root or paths.downloads("other", "bath_1574", "RoundaboutHD")
    output_folder = output_folder or paths.tier1("roundabouthd")
    import subprocess
    stuff.makedir(output_folder + "/annotation/")
    stuff.makedir(output_folder + "/video/")

    def _transcode(src, dst):
        tmp = dst + ".part.mp4"
        for codec_args in ((["-c:v", "h264_nvenc", "-preset", "p5",
                             "-rc", "vbr", "-cq", "22", "-b:v", "0"]),
                           (["-c:v", "libx264", "-preset", "medium",
                             "-crf", "22"])):
            from src.dataset_lite import audio_args, probe_audio
            cmd = (["ffmpeg", "-y", "-v", "error", "-i", src] + codec_args +
                   ["-g", "30", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart"] + audio_args(probe_audio(src))
                   + [tmp])
            if subprocess.run(cmd, capture_output=True).returncode == 0:
                os.replace(tmp, dst)
                return
        raise RuntimeError(f"transcode failed for {src}")

    for cam in ("c001", "c002", "c003", "c004"):
        stem = "roundabouthd_" + cam
        out_anno = output_folder + "/annotation/" + stem + ".json"
        out_video = output_folder + "/video/" + stem + ".mp4"
        if os.path.isfile(out_anno) and os.path.isfile(out_video):
            print(f"  {stem} already converted")
            continue
        cam_dir = os.path.join(src_root, "images" + cam)
        video = os.path.join(cam_dir, "video.mp4")
        sct = os.path.join(cam_dir, "SCT", "SCT_GT.txt")
        if not (os.path.isfile(video) and os.path.isfile(sct)):
            print(f"  {stem}: missing {video} or {sct}")
            continue
        ts = trackset.TrackSet()
        fmt_roundabouthd.read_into(ts, sct, video)
        if not os.path.isfile(out_video):
            _transcode(video, out_video)
        ts.metadata["original_video"] = out_video
        ts.export_yaml(out_anno)
        nb = sum(len(f["objects"]) for f in ts.frames)
        print(f"  {stem}: {len(ts.frames)} frames, {nb} boxes")
    print("roundabouthd done.")
    if lite:
        # tier 2 eval-spec derive (fixed elevated cameras)
        from src.corpus_manifest import derive_tracking
        derive_tracking("roundabouthd", hint="static")


def convert_uvg_vcm(src_root=None, output_folder=None, lite=True):
    """Convert downloaded UVG-VCM sequences into trackset JSON + mp4.

    Only sequences with BOTH a raw YUV on disk and a tracking-schema
    annotation JSON are converted (license-plate sequences use an
    incompatible schema and are skipped). The one-time transcode is
    x264 crf 18 yuv420p at native fps — raw source is yuv444p16le
    parsed from the filename `<Seq>_<W>x<H>_<fps>fps_..._<frames>.yuv`.
    """
    src_root = src_root or paths.downloads("other", "uvg_vcm")
    output_folder = output_folder or paths.tier1("uvg_vcm")
    import re
    import subprocess
    stuff.makedir(output_folder + "/annotation/")
    stuff.makedir(output_folder + "/video/")

    for seq in sorted(os.listdir(src_root)):
        seq_dir = os.path.join(src_root, seq)
        if not os.path.isdir(seq_dir):
            continue
        yuvs = [f for f in os.listdir(seq_dir) if f.endswith((".yuv", "_yuv"))]
        annos = [f for f in os.listdir(seq_dir)
                 if f.endswith(".json") and "annotation" in f.lower()]
        if not yuvs or not annos:
            continue
        anno_path = os.path.join(seq_dir, annos[0])
        with open(anno_path) as fh:
            head = json.load(fh)
        objs = next((v for k, v in head.items() if k.isdigit() and v), None)
        if not objs or not isinstance(objs[0], dict) \
                or "track_id" not in objs[0]:
            print(f"  {seq}: not tracking-schema annotation, skipped")
            continue
        m = re.search(r"_(\d+)x(\d+)_(\d+)fps", yuvs[0])
        if not m:
            print(f"  {seq}: cannot parse geometry from {yuvs[0]}, skipped")
            continue
        w, h, fps = int(m.group(1)), int(m.group(2)), int(m.group(3))

        stem = "uvgvcm_" + seq
        out_anno = output_folder + "/annotation/" + stem + ".json"
        out_video = output_folder + "/video/" + stem + ".mp4"
        if os.path.isfile(out_anno) and os.path.isfile(out_video):
            print(f"  {stem} already converted")
            continue
        if not os.path.isfile(out_video):
            tmp = out_video + ".part.mp4"
            cmd = ["ffmpeg", "-y", "-v", "error",
                   "-f", "rawvideo", "-pix_fmt", "yuv444p16le",
                   "-s", f"{w}x{h}", "-r", str(fps),
                   "-i", os.path.join(seq_dir, yuvs[0]),
                   "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                   "-pix_fmt", "yuv420p", "-movflags", "+faststart", tmp]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                print(f"  {stem}: transcode failed: {r.stderr[-200:]}")
                continue
            os.replace(tmp, out_video)
        ts = trackset.TrackSet()
        fmt_uvg_vcm.read_into(ts, anno_path, out_video, w, h, fps)
        ts.export_yaml(out_anno)
        nb = sum(len(f["objects"]) for f in ts.frames)
        print(f"  {stem}: {len(ts.frames)} frames, {nb} boxes")
    print("uvg_vcm done.")
    if lite:
        # tier 2 eval-spec derive (fixed cameras)
        from src.corpus_manifest import derive_tracking
        derive_tracking("uvg_vcm", hint="static")


def convert_meva(src_root=None, output_folder=None,
                 workers=8, augment=True, augment_limit=0, lite=True):
    """Convert MEVA (KF1) KPF clips into trackset JSON + video pairs under
    output_folder/{annotation,video}/.

    Walks src_root/annotations recursively for `<clip>.geom.yml`;
    import_meva auto-locates each clip's video (needed for exact
    width/height/fps) and the sibling `.types.yml`. Clips whose video is
    missing, or that contain no boxes, are skipped and counted. The local
    MEVA drop may be partial (the current one is a single 5-minute
    two-camera-set fixture) — re-run after adding clips; existing outputs
    are not redone. MEVA ships h264-in-avi; output videos are losslessly
    remuxed to mp4 (see _remux_to_mp4).
    """
    src_root = src_root or paths.downloads("other", "MEVA")
    output_folder = output_folder or paths.tier1("meva")
    stuff.makedir(output_folder + "/annotation/")
    stuff.makedir(output_folder + "/video/")

    geom_paths = []
    for root, _dirs, files in os.walk(os.path.join(src_root, "annotations")):
        for f in files:
            if f.endswith(".geom.yml") or f.endswith(".geom.yaml"):
                geom_paths.append(os.path.join(root, f))
    print(f"Found {len(geom_paths)} MEVA geom files")

    # cheap pre-pass: skip clips whose outputs already exist, before paying
    # the (slow) KPF yaml parse in a worker
    todo = []
    already = 0
    for geom_path in sorted(geom_paths):
        stem = os.path.basename(geom_path)
        for suf in (".geom.yml", ".geom.yaml"):
            if stem.endswith(suf):
                stem = stem[: -len(suf)]
                break
        if (os.path.isfile(output_folder + "/annotation/" + stem + ".json")
                and os.path.isfile(output_folder + "/video/" + stem + ".mp4")):
            already += 1
            continue
        todo.append((geom_path, output_folder))
    print(f"{already} already converted, {len(todo)} to process with {workers} workers")

    counts = {}
    if len(todo) > 0:
        import multiprocessing
        with multiprocessing.Pool(workers) as pool:
            for i, (status, stem) in enumerate(pool.imap_unordered(_convert_meva_clip, todo)):
                counts[status] = counts.get(status, 0) + 1
                if (i + 1) % 100 == 0 or i + 1 == len(todo):
                    print(f"  {i+1}/{len(todo)} {counts}")

    print(f"meva done. already={already} " +
          " ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    if augment:
        # MEVA labels only scripted-activity actors; add missing
        # high-confidence bystander tracks so the dataset is fully
        # annotated (GPU-serial second pass; resume-safe).
        from src.autolabel_bridge import augment_dataset
        augment_dataset(output_folder, limit=augment_limit)
    if lite:
        # tier 2 eval-spec derive AFTER augmentation (autolabel needs
        # native framerate; native truth stays in tier 1)
        from src.corpus_manifest import derive_tracking
        derive_tracking(os.path.basename(output_folder.rstrip("/")),
                        hint="static", max_seconds=120)


def convert_otw(src_root=None, output_folder=None,
                augment=True, augment_limit=0, lite=True):
    """Convert Out the Window (OTW) into MOT-equivalent JSON+mp4 pairs
    under output_folder/{annotation,video}/.

    Expects src_root to contain the extracted otw.tar.gz layout:
    {homes,lots}/video/*.mp4 + {homes,lots}/annotations.csv. Output stems
    are prefixed with the collection name so the two collections cannot
    collide. Videos whose annotations carry no object tracks (all of lots/
    — it only has actor-less activity regions) are skipped.
    """
    src_root = src_root or paths.downloads("other", "otw", "otw")
    output_folder = output_folder or paths.tier1("otw")
    import csv as _csv

    stuff.makedir(output_folder + "/annotation/")
    stuff.makedir(output_folder + "/video/")

    for collection in ("homes", "lots"):
        anno_path = os.path.join(src_root, collection, "annotations.csv")
        videos_dir = os.path.join(src_root, collection, "video")
        if not os.path.isfile(anno_path):
            print(f"  skipping collection {collection}: no {anno_path}")
            continue

        by_video = {}
        with open(anno_path, newline='') as f:
            for row in _csv.reader(f):
                if len(row) < 9:
                    continue
                video_id = row[0].strip()
                if video_id == "" or video_id.startswith("#"):
                    continue  # header row
                by_video.setdefault(video_id, []).append(row)
        print(f"{collection}: {len(by_video)} annotated videos")

        skipped_missing = 0
        skipped_failed = 0
        skipped_empty = 0
        for video_id in sorted(by_video.keys()):
            stem = video_id[:-4] if video_id.endswith(".mp4") else video_id
            video_path = os.path.join(videos_dir, stem + ".mp4")
            if not os.path.isfile(video_path):
                print(f"  skipping {video_id}: video not found at {video_path}")
                skipped_missing += 1
                continue
            out_stem = collection + "_" + stem
            out_anno = output_folder + "/annotation/" + out_stem + ".json"
            out_video = output_folder + "/video/" + out_stem + ".mp4"
            if os.path.isfile(out_anno) and os.path.isfile(out_video):
                continue

            print(f"Processing {collection}/{stem}...")
            ts = trackset.TrackSet()
            try:
                fmt_otw.read_into(ts, video_id, by_video[video_id], video_path)
            except Exception as e:
                print(f"  failed {video_id}: {e}")
                skipped_failed += 1
                continue
            if len(ts.frames) == 0:
                skipped_empty += 1
                continue

            # OTW mp4s can carry broken container timestamps (backward
            # pts, "doorbell disease"); route the copy through the
            # pts-monotonicity-checked remux (transcode fallback) rather
            # than copying bytes verbatim — mirrors the MEVA worker.
            if not os.path.isfile(out_video):
                _remux_to_mp4(video_path, out_video)
            ts.metadata["original_video"] = out_video
            ts.export_yaml(out_anno, output_video=None)

        print(f"{collection} done. missing_video={skipped_missing} failed={skipped_failed} empty={skipped_empty}")

    if augment:
        # OTW annotates only activity actors; add missing high-confidence
        # tracks so the dataset is fully annotated (GPU-serial pass).
        from src.autolabel_bridge import augment_dataset
        augment_dataset(output_folder, limit=augment_limit)
    if lite:
        # tier 2 eval-spec derive AFTER augmentation (native-fps
        # invariant; broken-pts sources are already handled at tier-1
        # import via the pts-checked _remux_to_mp4)
        from src.corpus_manifest import derive_tracking
        derive_tracking(os.path.basename(output_folder.rstrip("/")),
                        hint="static")


def convert_cevo(lite=True):
    output_folder=paths.tier1("cevo_april25")
    folder=paths.downloads("IndiaOfficeFrontDoor")
    stuff.makedir(output_folder+"/annotation/")
    stuff.makedir(output_folder+"/video/")
    seqs=os.listdir(folder)
    for s in seqs:
        if not s.endswith(".mp4"):
            continue
        if not os.path.isfile(folder+"/"+s):
            continue
        input=folder+"/"+s
        output_path=output_folder+"/annotation/"+s[:-4]+".json"
        output_video_path=output_folder+"/video/"+s
        print("Processing",folder,s,"....")
        ts=trackset.TrackSet(input)
        # cevo GT draws visible/partial-extent boxes (e.g. seated people
        # boxed minimally) — declare it so consumers can match conventions
        ts.metadata["box_convention"]="visible"
        ts.export_yaml(output_path, output_video_path)
    fix_cevo25_vfr_times()
    if lite:
        # tier 2 eval-spec derive (fixed front-door camera)
        from src.corpus_manifest import derive_tracking
        derive_tracking("cevo_april25", hint="static")


def convert_bdd100k_kaggle(src_root=None, output_folder=None, limit=0, lite=True):
    """Convert the 50-video Kaggle BDD100K MOT subset (original 30fps
    .mov clips + flattened scalabel CSV) into JSON+mp4 tracksets.

    Frame mapping MEASURED 2026-07-22: label frameIndex k corresponds to
    video time (k-1)/5 s, NOT k/5 — detector-vs-GT IoU sweep over 40
    sampled pedestrian boxes peaks at -1.0 interval (mean best-IoU 0.592
    vs 0.307 at k/5; offsets -2..0 swept). Likely 1-based jpg numbering
    in the original 5 fps extraction. frameIndex 0 clamps to t=0.

    Classes: pedestrian/rider -> person; car/truck/bus/train/motorcycle/
    bicycle/trailer/"other vehicle" -> vehicle; crowd-attribute boxes and
    "other person" -> other (ignore regions). box_convention "fullbody":
    occluded objects carry estimated full extents (visually verified on
    overlapping pedestrians/vehicles), matching MOT/JAAD semantics.
    """
    src_root = src_root or paths.downloads("other", "BDD100k_kaggle")
    output_folder = output_folder or paths.tier1("bdd100k_mot")
    import csv as _csv
    import subprocess
    person_cats = {"pedestrian", "rider"}
    vehicle_cats = {"car", "truck", "bus", "train", "motorcycle",
                    "bicycle", "trailer", "other vehicle"}
    videos_dir = os.path.join(src_root, "bdd100k", "videos", "train")
    stuff.makedir(output_folder + "/annotation/")
    stuff.makedir(output_folder + "/video/")

    by_video = {}
    with open(os.path.join(src_root, "mot_labels.csv")) as fh:
        for r in _csv.DictReader(fh):
            if r.get("haveVideo") != "True" or not r.get("category"):
                continue
            by_video.setdefault(r["videoName"], []).append(r)

    done = 0
    for vid in sorted(by_video):
        src_video = os.path.join(videos_dir, vid + ".mov")
        if not os.path.isfile(src_video):
            print(f"  skip {vid}: no video")
            continue
        out_anno = output_folder + "/annotation/bdd_" + vid + ".json"
        out_video = output_folder + "/video/bdd_" + vid + ".mp4"
        if os.path.isfile(out_anno) and os.path.isfile(out_video):
            continue
        cap = cv2.VideoCapture(src_video)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        cap.release()
        if not os.path.isfile(out_video):
            # RE-ENCODE, not stream copy: the source iPhone movs are
            # portrait streams with +-90 rotation metadata; consumers
            # that read the raw stream (NVDEC/C paths) would see them
            # sideways. Re-encoding applies the rotation to pixels and
            # strips the metadata (ffmpeg autorotate default).
            tmp = f"{out_video}.part{os.getpid()}.mp4"
            rc = subprocess.run(
                ["ffmpeg", "-y", "-v", "error", "-i", src_video,
                 "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                 "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                 tmp])
            if rc.returncode != 0:
                if os.path.exists(tmp):
                    os.remove(tmp)
                print(f"  FAIL remux {vid}")
                continue
            os.replace(tmp, out_video)
        track_id_map = {}
        by_frame = {}
        for r in by_video[vid]:
            cat = r["category"]
            if r.get("attributes.crowd") == "True" or cat == "other person":
                cl = 2
            elif cat in person_cats:
                cl = 0
            elif cat in vehicle_cats:
                cl = 1
            else:
                continue
            x1 = round(max(0.0, min(1.0, float(r["box2d.x1"]) / width)), 4)
            y1 = round(max(0.0, min(1.0, float(r["box2d.y1"]) / height)), 4)
            x2 = round(max(0.0, min(1.0, float(r["box2d.x2"]) / width)), 4)
            y2 = round(max(0.0, min(1.0, float(r["box2d.y2"]) / height)), 4)
            if x2 <= x1 or y2 <= y1:
                continue
            tid = r["id"]
            if tid not in track_id_map:
                track_id_map[tid] = len(track_id_map) + 1
            k = int(r["frameIndex"])
            by_frame.setdefault(k, {})[track_id_map[tid]] = {
                "box": [x1, y1, x2, y2], "class": cl, "conf": 1.0}
        frames = []
        for k in sorted(by_frame):
            t = max(0.0, (k - 1) / 5.0)
            frames.append({"frame_id": k, "frame_time": round(t, 6),
                           "objects": by_frame[k]})
        doc = {"metadata": {
                   "frame_rate": fps, "width": width, "height": height,
                   "classes": ["person", "vehicle", "other"],
                   "box_convention": "fullbody",
                   "original_video": out_video},
               "frames": frames}
        with open(out_anno, "w") as fh:
            json.dump(doc, fh, indent=4)
        done += 1
    print(f"convert_bdd100k_kaggle: {done} sequences -> {output_folder}")
    if lite:
        # tier 2 eval-spec derive (dashcam = moving camera)
        from src.corpus_manifest import derive_tracking
        derive_tracking("bdd100k_mot", hint="bodycam")


def convert_autolabel_folder(src_folder, output_folder, shard="",
                             convention="fullbody", cuts=False):
    """Generic importer: fully autolabel every mp4 in src_folder into a
    dataset at output_folder/{annotation,video} suitable for utrack
    optimization. Requires the autolabel checkout (see
    src/autolabel_bridge.py — helpful error if missing).

    Resume-by-skip per video; shard="i/N" runs every N-th video (launch
    N processes to overlap GPU/decode). The autolabel export is already
    in track's annotation JSON format; videos are copied in unchanged.
    cuts=True enables autolabel's scene-cut detection (edited multi-shot
    sources, e.g. movies).
    """
    from src.autolabel_bridge import autolabel_video
    import shutil
    stuff.makedir(output_folder + "/annotation/")
    stuff.makedir(output_folder + "/video/")
    vids = sorted(f for f in os.listdir(src_folder)
                  if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv")))
    si, sn = 0, 1
    if shard:
        si, sn = (int(x) for x in shard.split("/"))
    done = 0
    for k, v in enumerate(vids):
        if k % sn != si:
            continue
        stem = os.path.splitext(v)[0]
        out_anno = output_folder + "/annotation/" + stem + ".json"
        out_video = output_folder + "/video/" + stem + ".mp4"
        src = os.path.join(src_folder, v)
        try:
            if not (os.path.isfile(out_anno) and os.path.isfile(out_video)):
                # AV1 sources: rfdetr's worker env can't decode them, so
                # transcode to h264 as the dataset-local copy up front and
                # autolabel that file instead of the original
                if _video_codec(src) == "av1":
                    if not os.path.isfile(out_video):
                        _transcode_h264(src, out_video)
                    src = out_video
                autolabel_video(src, out_anno, convention=convention,
                                cuts=cuts)
            elif (json.load(open(out_anno)).get("metadata", {})
                    .get("min_annotated_height")):
                continue
            # else: annotation exists but was written by an external
            # labeller run — fall through to stamp it
            if not os.path.isfile(out_video):
                if os.path.splitext(src)[1].lower() != ".mp4":
                    # .mov/.avi/.mkv sources: repackage into a real mp4
                    # container (pts-checked, transcode fallback) rather
                    # than copying foreign bytes to a .mp4 name
                    _remux_to_mp4(src, out_video)
                else:
                    shutil.copy(src, out_video)
            # point the annotation at the dataset-local video copy
            d = json.load(open(out_anno))
            d.setdefault("metadata", {})["original_video"] = out_video
            # autolabel stamps its processing rate (native/stride) as
            # frame_rate; track consumers treat frame_rate as the video's
            # native clock (the tracker times decoded frames as index/fps,
            # so a halved rate plays detections at 2x their true time)
            d["metadata"]["frame_rate"] = _native_fps(out_video)
            # fully-autolabelled: annotation completeness only above the
            # detector-reliability knee (normalized height, per-class)
            d["metadata"]["min_annotated_height"] = {"person": 0.045}
            with open(out_anno, "w") as fh:
                json.dump(d, fh, indent=4)
            done += 1
            print(f"autolabelled {stem} ({done})", flush=True)
        except Exception as e:
            # one poisoned video must not kill the batch; resume-by-skip
            # re-attempts it on the next run
            print(f"FAIL {stem}: {type(e).__name__}: {e}", flush=True)
    print(f"convert_autolabel_folder: {done} videos -> {output_folder}")


def convert_raw_movies(src_folder=None, output_folder=None, shard="", lite=True):
    """Raw movie/trailer mp4s, fully autolabelled. Edited multi-shot
    content, so autolabel's scene-cut detection (TransNetV2) is enabled:
    tracks must not survive or merge across cuts. Moving camera: the
    final lite pass decimates to the analytics grid (hint:bodycam)."""
    src_folder = src_folder or paths.video("youtube")
    output_folder = output_folder or paths.tier1("raw_movies")
    convert_autolabel_folder(src_folder, output_folder, shard=shard,
                             cuts=True)
    if lite:
        from src.corpus_manifest import derive_tracking
        derive_tracking(os.path.basename(output_folder.rstrip("/")),
                        hint="bodycam")


def convert_bwc_videotext(src_folder=None, output_folder=None, shard="", lite=True):
    """Body-worn-camera eval videos, fully autolabelled (use case 2)."""
    src_folder = src_folder or paths.downloads("other", "BWC-VideoText-359", "eval_videos")
    output_folder = output_folder or paths.tier1("bwc-videotext")
    convert_autolabel_folder(src_folder, output_folder, shard=shard)
    if lite:
        from src.corpus_manifest import derive_tracking
        derive_tracking(os.path.basename(output_folder.rstrip("/")),
                        hint="bodycam")


def reduce_dataset(folder, n=50, group_fn=None, manifest=None):
    """Offline heuristic selection of the top-N clips of a converted
    dataset (no GPU): rich in annotations, diverse across scenes.

    Scoring per clip (from the annotation JSON alone):
      score = person_boxes + 2*person_tracks + 0.5*vehicle_boxes
              + 10*mean_objects_per_frame
    (annotation volume dominates; density bonus favours busy scenes).
    Diversity: clips are grouped by scene/camera (group_fn(stem); default
    strips date/time+digits — MEVA "<date>.<t0>.<t1>.<site>.<cam>" groups
    to "site.cam", OTW "homes_00012" to "homes_") and selection is a
    round-robin over groups in descending score, so one prolific camera
    cannot fill the whole budget.

    Writes <folder>/reduced_<n>.json (list of stems) and returns it.
    Consumers: autolabel_bridge.augment_dataset(names=...), or any
    importer that wants a "reduced" mode.
    """
    anno_dir = os.path.join(folder, "annotation")

    def default_group(stem):
        parts = stem.split(".")
        if len(parts) >= 5:               # MEVA-style dotted stems
            return ".".join(parts[3:])[:40]
        return "".join(c for c in stem if not c.isdigit())[:40]

    gf = group_fn or default_group
    scored = []
    for f in sorted(os.listdir(anno_dir)):
        if not f.endswith(".json"):
            continue
        try:
            d = json.load(open(os.path.join(anno_dir, f)))
        except Exception:
            continue
        frames = d.get("frames", [])
        if not frames:
            continue
        pb = vb = 0
        ptr, vtr = set(), set()
        for fr in frames:
            for tid, o in fr["objects"].items():
                if o.get("class") == 0:
                    pb += 1
                    ptr.add(tid)
                elif o.get("class") == 1:
                    vb += 1
                    vtr.add(tid)
        dens = (pb + vb) / max(len(frames), 1)
        score = pb + 2 * len(ptr) + 0.5 * vb + 10 * dens
        if score <= 0:
            continue
        stem = f[:-5]
        scored.append((score, stem, gf(stem)))

    by_group = {}
    for sc, stem, g in sorted(scored, reverse=True):
        by_group.setdefault(g, []).append((sc, stem))
    picked = []
    while len(picked) < n and any(by_group.values()):
        # round-robin: best remaining clip of each group, richest group
        # first, until the budget is filled
        for g in sorted(by_group,
                        key=lambda g: -(by_group[g][0][0]
                                        if by_group[g] else -1)):
            if by_group[g]:
                picked.append(by_group[g].pop(0)[1])
                if len(picked) >= n:
                    break
    out = manifest or os.path.join(folder, f"reduced_{n}.json")
    with open(out, "w") as fh:
        json.dump(sorted(picked), fh, indent=1)
    print(f"reduce_dataset: {len(picked)}/{len(scored)} clips "
          f"({len(by_group)} scene groups) -> {out}")
    return out


def apply_reduction(folder, manifest=None, n=50):
    """Prune a converted dataset to its reduced_N selection: clips NOT in
    the manifest move to <folder>_unreduced/{annotation,video} (same-fs
    rename — instant and reversible; delete that dir to reclaim space).
    Runs reduce_dataset first if the manifest doesn't exist yet."""
    import shutil
    manifest = manifest or os.path.join(folder, f"reduced_{n}.json")
    if not os.path.isfile(manifest):
        reduce_dataset(folder, n=n, manifest=manifest)
    keep = set(json.load(open(manifest)))
    arch = folder.rstrip("/") + "_unreduced"
    stuff.makedir(arch + "/annotation/")
    stuff.makedir(arch + "/video/")
    moved = 0
    for sub, ext in (("annotation", ".json"), ("video", ".mp4")):
        d = os.path.join(folder, sub)
        for f in sorted(os.listdir(d)):
            stem, e = os.path.splitext(f)
            if e != ext or stem in keep:
                continue
            shutil.move(os.path.join(d, f), os.path.join(arch, sub, f))
            moved += 1
    # keep the manifest with the reduced set
    print(f"apply_reduction: kept {len(keep)}, moved {moved} files "
          f"-> {arch}")
