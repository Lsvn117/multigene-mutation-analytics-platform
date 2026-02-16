from flask import Flask, render_template, request
import numpy as np
import joblib
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the trained SVM model and scaler
model = joblib.load('svm_model.pkl')         # Trained SVC model
scaler = joblib.load('scaler.pkl')           # Trained MinMaxScaler

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'embedding_file' not in request.files:
        return "No file uploaded", 400

    file = request.files['embedding_file']
    if file.filename == '':
        return "No selected file", 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    try:
        # Load and flatten embedding
        embedding = np.load(filepath).flatten().reshape(1, -1)

        # Pad to match training shape
        expected_len = model.support_vectors_.shape[1]
        if embedding.shape[1] < expected_len:
            embedding = np.pad(embedding, ((0, 0), (0, expected_len - embedding.shape[1])), 'constant')
        elif embedding.shape[1] > expected_len:
            embedding = embedding[:, :expected_len]

        # Normalize
        embedding_scaled = scaler.transform(embedding)

        # Predict
        prediction = model.predict(embedding_scaled)[0]

        label = "Pathogenic" if prediction == 1 else "Benign"
        result = f"{label}"

    except Exception as e:
        return f"Error during prediction: {e}", 500
    finally:
        os.remove(filepath)

    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
