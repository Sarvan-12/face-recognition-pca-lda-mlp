# Face Recognition with PCA, LDA & MLP

A demo project for face recognition using Principal Component Analysis (PCA), Linear Discriminant Analysis (LDA), and a Multi-Layer Perceptron (MLP) classifier. Includes a simple Flask web app for uploading and predicting faces.

## Features
- Dimensionality reduction via PCA and LDA
- Classification using an MLP neural network
- Simple web interface with Flask
- Easily extensible dataset structure

## Installation

1. **Clone the repo:**
   ```bash
   git clone https://github.com/Sarvan-12/face-recognition-pca-lda-mlp.git
   cd face-recognition-pca-lda-mlp
   ```

2. **Set up the environment:**
   - For Windows:
     ```bash
     python -m venv .venv
     .venv\Scripts\activate
     ```
   - For Linux/Mac:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   # or if requirements.txt is missing:
   pip install flask opencv-python numpy scikit-learn
   ```

## Usage

1. **Prepare the dataset:**
   - Organize images into `dataset/faces/<person_name>/` directories, where each folder is a class label.

2. **Run the app:**
   ```bash
   python code/app.py
   # Or depending on your project structure
   # python3 code/app.py
   ```

3. **Open in browser:**
   Visit [http://127.0.0.1:5000](http://127.0.0.1:5000) and upload a face image.

## Algorithms Used
- **PCA (Principal Component Analysis):** Reduces image dimensionality while preserving variance.
- **LDA (Linear Discriminant Analysis):** Projects data in a way that maximizes class separability.
- **MLP (Multi-Layer Perceptron):** Neural network classifier for faces after feature engineering.

## Example
> _Coming soon: Add usage gifs or UI screenshots!_
