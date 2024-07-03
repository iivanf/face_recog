## Face Recognition CLI Tool in Python

This Python script `detector.py` is a command-line interface (CLI) tool for face recognition using the `face_recognition` library and `PIL` (Python Imaging Library). It provides functionality to encode known faces and validate faces in a validation directory.

### Features

- **Encoding Known Faces**:
  - Detects faces in images from a specified training directory (`training/`).
  - Computes facial encodings using the `face_recognition` library.
  - Saves the computed encodings along with corresponding names to a pickle file (`output/encodings.pkl`).

- **Recognize Faces**:
  - Loads precomputed face encodings from `encodings.pkl`.
  - Detects faces in images from a validation directory (`validation/`).
  - Compares detected face encodings with the precomputed encodings to recognize known faces.
  - Draws bounding boxes around recognized faces and displays the result using `PIL`.

### Requirements

- Python 3.x
- `face_recognition` library (`pip install face_recognition`)
- `PIL` (Python Imaging Library, `pip install pillow`)

### Usage

#### Training (Encode Known Faces)

```bash
python detector.py --train
```

- Encodes known faces from images in the `training/` directory.
- Saves the facial encodings to `output/encodings.pkl`.

#### Validation (Validate Faces)

```bash
python detector.py --validate
```

- Validates faces in images from the `validation/` directory.
- Recognizes known faces using the precomputed encodings from `output/encodings.pkl`.

### Implementation Details

- **Encoding Known Faces**:
  - Iterates through images in the `training/` directory.
  - Uses `face_recognition` to detect face locations and compute face encodings.
  - Serializes the names and encodings into a pickle file (`encodings.pkl`).

- **Recognize Faces**:
  - Loads face encodings from `encodings.pkl`.
  - Detects faces in images from the `validation/` directory.
  - Compares detected face encodings against stored encodings to identify known faces.
  - Draws bounding boxes around recognized faces and displays annotated images.
