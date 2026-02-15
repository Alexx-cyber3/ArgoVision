# AgroVision AI

AgroVision AI is a full-stack crop disease detection system that uses deep learning to identify diseases from images of leaves, roots, stems, and flowers.

## Project Structure

- `backend/`: FastAPI backend with SQLite database and ML inference logic.
- `mobile/`: Flutter mobile application source code.
- `ml_training/`: Python scripts for training the TensorFlow models.

## Setup Instructions

### 1. Backend Setup

1.  Navigate to `backend/`.
2.  Create a virtual environment: `python -m venv venv`
3.  Activate it: `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Linux/Mac)
4.  Install dependencies: `pip install -r requirements.txt`
5.  Run the server: `uvicorn app.main:app --reload --host 0.0.0.0 --port 8000`

### 2. Mobile App Setup

1.  Navigate to `mobile/`.
2.  Run `flutter pub get`.
3.  Run on emulator: `flutter run`.

### 3. ML Training (Optional)

1.  Navigate to `ml_training/`.
2.  Place your dataset in `dataset/` organized by class folders.
3.  Run `python train.py` to train models.
4.  Move generated `.h5` or `.tflite` models to `backend/app/models/`.

## Features

- **Multi-Part Detection**: Identifies Leaf, Root, Stem, Flower.
- **Disease Diagnosis**: Part-specific disease classification.
- **Treatment Recommendations**: Chemical and organic solutions.
- **History Tracking**: Saves prediction history locally.
