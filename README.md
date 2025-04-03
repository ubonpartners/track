# 🛰️ Track

**Basic tools for object tracking** — including utilities for running, testing, evaluating, and optimizing object tracking algorithms.

---

## 📁 Supported Test Sequence Formats

The tools work on annotated video sequences ("tracksets"). Two formats are currently supported:

### 1. MOT Format
As described on the [MOT Challenge](https://motchallenge.net/) website:
- A `seqinfo.ini` file with framerate and resolution metadata
- `gt/` folder containing `gt.txt` annotations
- `img1/` folder containing JPEGs for each annotated frame

### 2. Ubon Format
- An MP4 video file
- A JSON file for annotations (intended to be self-explanatory)

---

## 🧠 Supported Tracking Algorithms

- Built-in **Ultralytics tracker**, supporting both **ByteTrack** and **BoT-SORT**
- Separate support for an independent implementation of **ByteTrack**

Each algorithm is defined via a YAML config file, which includes:
- The tracker type
- The object detector to use
- All necessary parameters

---

## 🛠️ Tools

### 🔍 Visualize an Annotated Video
```bash
python track.py --view --trackset <trackset>
```
- `trackset` can be a JSON file or MOT `seqinfo.ini`

---

### 🚀 Run Tracking and Compare Against Ground Truth (with `pymotmetrics`)
```bash
python track.py --track --trackset <trackset> --config <config>
```
- `config`: YAML file specifying the tracker to use

---

### 📊 Benchmark Multiple Trackers Across Datasets
```bash
python track.py --test <test_yaml>
```
- `test_yaml`: YAML file describing datasets and trackers

---

### 🔧 Optimize Tracker Parameters Automatically
```bash
python track.py --search <search_yaml>
```

---

## 📜 License

This project (including the weights) is dual-licensed under:

- **AGPL** for **non-commercial use only**
- The **[Ubon Cooperative License](https://github.com/ubonpartners/license/blob/main/LICENSE)**

Please reach out with questions:  
📬 [bernandocribbenza@gmail.com](mailto:bernandocribbenza@gmail.com?subject=yolo-dpa%20question&body=Your-code-is-rubbish!)
