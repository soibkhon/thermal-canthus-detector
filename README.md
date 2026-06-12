# Thermal Inner Canthus Detector

Detects the two **inner eye corners (inner canthi)** in thermal infrared face images.
Outputs pixel coordinates and an annotated image with blue dots.

Uses a **two-stage pipeline** — a robust face detector that works at any distance,
followed by a high-precision canthus regressor that only runs when the face is close
enough for the canthi to actually be resolvable.

---

## How it works

```
Thermal image
      │
      ▼
┌─────────────────────────────────────────┐
│  Stage 1 — Face Detector                │
│  thermal_detector.pt                    │
│  Trained on SF-TL54 + TFW (18k images) │
│  Detects faces at any distance          │
└─────────────────────────────────────────┘
      │
      ├─ No face found ──────────────────────▶ return None
      │
      ├─ Face bbox width < MIN_FACE_PX ──────▶ "too far"
      │   (canthi are sub-pixel, not useful)     face_box returned,
      │                                          left/right = None
      │
      └─ Face large enough
            │
            │  crop + pad face region
            │  (upscales small faces automatically)
            ▼
┌─────────────────────────────────────────┐
│  Stage 2 — Canthus Regressor            │
│  thermal_canthus_yolo.pt                │
│  Trained on SF-TL54 only (4 107 images) │
│  3.98 px mean error on test set         │
└─────────────────────────────────────────┘
      │
      ▼
  left (x, y)  +  right (x, y)
  mapped back to original image coordinates
```

**Why two models?**
The face detector was trained on 18k diverse thermal images including outdoor and
far-away subjects, making it robust at distance. The canthus regressor was trained
only on close-up lab images with precise inner-canthus annotations — mixing the two
tasks into one model degraded keypoint precision. Keeping them separate gives the
best of both: detection range and localization accuracy.

**Distance cutoff (`MIN_FACE_PX = 80`):**
When the detected face bounding box is narrower than 80 px, the inter-canthus
distance is only 1–2 px — too small to report meaningfully. The pipeline flags
these as `too_far = True` and skips stage 2. You can tune this threshold to match
your camera and use-case (see [Tuning the distance cutoff](#tuning-the-distance-cutoff)).

---

## Benchmark (549 held-out thermal test images, IS2AI SF-TL54)

| Model | Det % | Mean err | Median err | < 5 px | < 10 px |
|---|---|---|---|---|---|
| **This model (stage 2)** | **100 %** | **3.98 px** | **3.87 px** | **77 %** | **100 %** |
| T-FAKE (CVPR 2025) | 100 % | 5.38 px | 4.74 px | 54 % | 93 % |
| FAN 68-pt | 98 % | 6.42 px | 5.23 px | 47 % | 89 % |
| MediaPipe 478-pt | 82 % | 11.50 px | 6.14 px | 39 % | 72 % |

Benchmark is on close-up controlled images. Stage 1 additionally handles
outdoor and far-away faces that these baselines miss entirely.

---

## Install

```bash
pip install ultralytics opencv-python numpy
# or
pip install -r requirements.txt
```

---

## Usage

### Command line

```bash
# detect and annotate (saves <name>_canthus.png)
python detect_canthus.py face.png

# specify output path
python detect_canthus.py face.png --out result.png

# print coordinates only, no image saved
python detect_canthus.py face.png --no-save

# lower the distance cutoff (attempt canthus on smaller faces)
python detect_canthus.py face.png --min-face 50
```

Output when close enough:
```
Left  inner canthus: (207, 128)
Right inner canthus: (253, 131)
Saved → face_canthus.png
```

Output when too far:
```
Face detected at (120, 45, 165, 95) but too far — canthi not resolvable.
```

### Python API

```python
from detect_canthus import load_models, detect, annotate

models = load_models()          # load both models once, reuse for many images

result = detect(models, "face.png")

result.face_box   # (x1, y1, x2, y2) bounding box, or None if no face
result.too_far    # True if face found but too small for canthus detection
result.left       # (x, y) inner canthus, LEFT  side of image — or None
result.right      # (x, y) inner canthus, RIGHT side of image — or None
```

Example with all cases handled:

```python
result = detect(models, "face.png")

if result.face_box is None:
    print("No face detected")
elif result.too_far:
    print("Face detected but person is too far away")
else:
    print(f"Left  canthus: {result.left}")
    print(f"Right canthus: {result.right}")
    img = annotate("face.png", result)
    cv2.imwrite("result.png", img)
```

Batch processing:

```python
from pathlib import Path
from detect_canthus import load_models, detect

models = load_models()

for path in Path("images/").glob("*.png"):
    result = detect(models, path)
    if not result.too_far and result.left:
        print(path.name, result.left, result.right)
```

### Live video

```bash
python live_canthus.py                   # webcam 0
python live_canthus.py --source 1        # second camera / thermal USB
python live_canthus.py --source video.mp4
python live_canthus.py --device cuda     # run on GPU
python live_canthus.py --save out.mp4    # record output
python live_canthus.py --min-face 50     # lower distance cutoff
```

Press **Q** or **Esc** to quit.

---

## Tuning the distance cutoff

`MIN_FACE_PX` (default `80`) is the face bounding box width in pixels below which
canthus detection is skipped.

**Find your value:** run the live script and walk to the furthest distance where
you want measurements. The status bar shows the face width when it is too small:
```
too far  (face 43px wide)
```
Use that number as your `MIN_FACE_PX`.

**Change it permanently** — edit line 52 of `detect_canthus.py`:
```python
MIN_FACE_PX = 80   # ← set to your value
```

**Change it per-call** without editing the file:
```python
result = detect(models, frame, min_face_px=50)
```

**Change it from the CLI:**
```bash
python detect_canthus.py face.png --min-face 50
```

| Value | Effect |
|---|---|
| Higher (e.g. 120) | Only close-up faces — most precise |
| Lower (e.g. 40) | More range — noisier on small faces |
| 0 | Always attempt — not recommended |

---

## Files

| File | Description |
|---|---|
| `thermal_detector.pt` | Stage 1 — face detector (SF-TL54 + TFW, 18k images) |
| `thermal_canthus_yolo.pt` | Stage 2 — canthus regressor (SF-TL54 only, 4 107 images) |
| `detect_canthus.py` | Detection pipeline (CLI + Python API) |
| `live_canthus.py` | Live video / webcam script |

---

## Input

- Any thermal infrared face image: PNG, JPG
- Works with **gray** and **iron** false-colour palettes
- Typical size: 320 × 256 or 640 × 512 (models auto-scale)

## Output

| Field | Meaning |
|---|---|
| `left` | Inner canthus on the **left side of the image** (subject's right eye) |
| `right` | Inner canthus on the **right side of the image** (subject's left eye) |
| `face_box` | `(x1, y1, x2, y2)` bounding box in original image pixels |
| `too_far` | `True` when face is detected but too small to localise canthi |

All coordinates are in **pixel space** relative to the original image.
