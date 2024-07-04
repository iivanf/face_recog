## Face Recognition CLI Tool in Python

This Python script `detector.py` is a command-line interface (CLI) tool for face recognition using the `face_recognition` library and `PIL` (Python Imaging Library). It provides functionality to encode known faces and validate faces in a validation directory. Additionally, it supports training and validation using the `InsightFace` model.

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

- **InsightFace Integration**:
  - Utilizes the InsightFace model for training and validation.
  - Prepares training data by extracting embeddings from face images.
  - Trains a classification model using SVM (Support Vector Machine) on the extracted embeddings.
  - Evaluates the trained model on validation data, providing confidence scores for recognized faces.

### Requirements

- Python 3.x
- `face_recognition` library (`pip install face_recognition`)
- `PIL` (Python Imaging Library, `pip install pillow`)
- `insightface` library

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

#### InsightFace Integration (Train and Validate)
```bash
python detector.py --insightface
```

- Prepares training data using InsightFace from images in the training/ directory.
- Trains a model and evaluates its performance on faces from the validation/ directory.
- Provides detailed recognition results including confidence scores.

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

- **InsightFace Integration**:
  - Utilizes insightface library to extract embeddings and perform face recognition tasks.
  - Integrates with SVM for face classification based on extracted embeddings.
  - Evaluates recognition results with confidence scores and visual feedback.
