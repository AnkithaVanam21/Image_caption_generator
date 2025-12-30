from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, Embedding, LSTM, Add
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import pickle
import os
from werkzeug.utils import secure_filename

# ========== CONSTANTS ==============
MODEL_PATH = 'model/best_model.h5'
TOKENIZER_PATH = 'model/tokenizer.pkl'
MAX_LENGTH = 35
VOCAB_SIZE = 8485
EMBEDDING_DIM = 256
LSTM_UNITS = 256
UPLOAD_FOLDER = 'static/uploads'
# ===================================

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load tokenizer
# Load tokenizer with legacy keras support
class LegacyKerasUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "keras.src.legacy.preprocessing.text":
            module = "keras.preprocessing.text"
        if module == "keras.src.legacy.preprocessing.sequence":
            module = "keras.preprocessing.sequence"
        return super().find_class(module, name)

with open(TOKENIZER_PATH, 'rb') as f:
    tokenizer = LegacyKerasUnpickler(f).load()


# Load VGG16 for image feature extraction
vgg = VGG16(weights='imagenet')
vgg = Model(inputs=vgg.inputs, outputs=vgg.get_layer('fc2').output)

# Custom NotEqual layer
class NotEqual(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.math.not_equal(inputs, 0)

    def get_config(self):
        return super().get_config()

# Rebuild model architecture (matches trained model)
def build_model():
    inputs1 = Input(shape=(4096,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)

    inputs2 = Input(shape=(MAX_LENGTH,))
    se1 = Embedding(VOCAB_SIZE, EMBEDDING_DIM, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(LSTM_UNITS)(se2)

    decoder = Add()([fe2, se3])
    decoder = Dense(256, activation='relu')(decoder)
    outputs = Dense(VOCAB_SIZE, activation='softmax')(decoder)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    return model

# Load model and weights
model = build_model()
model.load_weights(MODEL_PATH)
print("✅ Model loaded!")

# Extract image features using VGG16
def extract_features(image_path):
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    feature = vgg.predict(image, verbose=0)
    return feature  # shape (1, 4096)

# Generate caption
def generate_caption(photo):
    in_text = 'startseq'
    for _ in range(MAX_LENGTH):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=MAX_LENGTH)
        y_pred = model.predict([photo, sequence], verbose=0)
        y_pred = np.argmax(y_pred)
        word = next((w for w, i in tokenizer.word_index.items() if i == y_pred), None)
        if word is None or word == 'endseq':
            break
        in_text += ' ' + word
    return in_text.replace('startseq', '').strip()

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    features = extract_features(filepath)
    caption = generate_caption(features)

    return jsonify({'caption': caption})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
