# Thermal Inner Canthus Detector

Detects the two **inner eye corners (inner canthi)** in thermal infrared face images.  
Outputs pixel coordinates and an annotated image with blue dots.

Built on YOLOv8s-pose fine-tuned on the [IS2AI SF-TL54](https://github.com/IS2AI/thermal-facial-landmarks-detection) thermal facial landmarks dataset (4 107 images, gray + iron palettes).

## Benchmark (549 held-out thermal test images, IS2AI SF-TL54)

| Model | Detection rate | Mean error | Median error | Within 5 px | Within 10 px |
|---|---|---|---|---|---|
| **This model** | **100 %** | **3.98 px** | **3.87 px** | **77 %** | **100 %** |
| T-FAKE (CVPR 2025) | 100 % | 5.38 px | 4.74 px | 54 % | 93 % |
| FAN 68-pt | 98 % | 6.42 px | 5.23 px | 47 % | 89 % |
| MediaPipe 478-pt | 82 % | 11.50 px | 6.14 px | 39 % | 72 % |

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
# annotate an image (saves <name>_canthus.png next to the original)
python detect_canthus.py face.png

# specify output path
python detect_canthus.py face.png --out result.png

# print coordinates only, no image saved
python detect_canthus.py face.png --no-save
```

Output:
```
Left  inner canthus: (207, 128)
Right inner canthus: (253, 131)
Saved → face_canthus.png
```

### Python API

```python
from detect_canthus import load_model, detect, annotate

model = load_model()                          # load once, reuse for many images

left, right = detect(model, "face.png")
# left  = (x, y) — inner canthus, LEFT  side of image (person's right eye)
# right = (x, y) — inner canthus, RIGHT side of image (person's left  eye)
# either may be None if detection fails

print(f"Left canthus:  {left}")
print(f"Right canthus: {right}")

# draw blue dots and save
import cv2
img = annotate("face.png", left, right)
cv2.imwrite("result.png", img)
```

Batch processing:

```python
from pathlib import Path
from detect_canthus import load_model, detect

model = load_model()

for path in Path("images/").glob("*.png"):
    left, right = detect(model, path)
    print(path.name, left, right)
```

---

## Input

- Any thermal infrared face image: PNG, JPG
- Works with **gray** and **iron** false-colour palettes
- Typical size: 320 × 256 or 640 × 512 (model auto-scales)

## Output

| Value | Meaning |
|---|---|
| `left` | Inner canthus on the **left side of the image** (subject's right eye) |
| `right` | Inner canthus on the **right side of the image** (subject's left eye) |

Coordinates are in **pixel space** relative to the original image.

---

## Model details

- Architecture: YOLOv8s-pose, 2-keypoint head  
- Training: 150 epochs, AdamW, CLAHE preprocessing, IS2AI SF-TL54 dataset  
- Weights: `thermal_canthus_yolo.pt` (23 MB)
