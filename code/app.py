from flask import Flask, render_template, request, redirect, url_for, flash
import cv2
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Set paths
UPLOAD_FOLDER = "code/uploads"
DATASET_FOLDER = "dataset/faces"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def prepare_data():
    X, y, target_names = [], [], []
    person_id = 0
    h, w = 300, 300  # Resize dimensions
    class_names = []

    for person_name in os.listdir(DATASET_FOLDER):
        dir_path = os.path.join(DATASET_FOLDER, person_name)
        if os.path.isdir(dir_path):
            class_names.append(person_name)
            for image_name in os.listdir(dir_path):
                image_path = os.path.join(dir_path, image_name)
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                img_resized = cv2.resize(img, (h, w))
                X.append(img_resized.flatten())
                y.append(person_id)
            person_id += 1

    return np.array(X), np.array(y), class_names, h, w

X, y, class_names, h, w = prepare_data()
n_classes = len(class_names)

# Train the models
pca = PCA(n_components=150, svd_solver='randomized', whiten=True)
X_pca = pca.fit_transform(X)

lda = LinearDiscriminantAnalysis()
X_lda = lda.fit_transform(X_pca, y)

clf = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000, random_state=42, verbose=True)
clf.fit(X_lda, y)

@app.route('/')
def home():
    return render_template('index.html', class_names=class_names)

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Process the uploaded image
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        img_resized = cv2.resize(img, (h, w)).flatten().reshape(1, -1)
        img_pca = pca.transform(img_resized)
        img_lda = lda.transform(img_pca)
        pred = clf.predict(img_lda)
        predicted_person = class_names[pred[0]]

        return render_template('index.html', result=predicted_person, class_names=class_names)

    flash('Invalid file format. Please upload a PNG, JPG, or JPEG image.')
    return redirect(url_for('home'))

if __name__ == "__main__":
    app.run(debug=True)
