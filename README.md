📘 Image Caption Generator
📌 Project Overview

This project is an Image Caption Generator that automatically generates a meaningful textual description (caption) for a given image.
It uses Deep Learning (CNN + LSTM) architecture and is deployed as a Flask web application.

🧠 How It Works

1. User uploads an image through the web interface.
2. The image is processed using VGG16 (CNN) to extract visual features.
3. Extracted features are passed to an LSTM-based model.
4. The LSTM model generates a caption word by word using a trained tokenizer.
5. The final caption is displayed to the user.

🏗️ Architecture
Image Upload
     ↓
VGG16 CNN (Feature Extraction)
     ↓
LSTM Decoder + Tokenizer
     ↓
Generated Caption


🛠️ Tech Stack

- Python
- TensorFlow / Keras
- Flask
- NumPy
- Pillow
- HTML / CSS

📂 Project Structure

image_caption_generator/
│
├── app.py                  # Main Flask application
├── requirements.txt        # Project dependencies
├── model/
│   ├── best_model.h5       # Trained LSTM model
│   └── tokenizer.pkl       # Tokenizer for caption generation
│
├── templates/
│   └── index.html          # Frontend UI
│
├── static/
│   └── uploads/            # Uploaded images

⚙️ Installation & Setup
1️⃣ Create Virtual Environment
python -m venv venv
venv\Scripts\activate


2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Run the Application
python app.py

4️⃣ Open in Browser
http://127.0.0.1:5000


📥 Input

Any image file (JPG / PNG)

📤 Output

A generated caption describing the image

Example:
Input Image → Surfing Image
Output Caption → "a man in a wetsuit surfing on a wave"


⚠️ Limitations

1. The model is trained on general images, so captions may be inaccurate for:
- Medical images
- Ultrasound images
- Illustrations
2. Some captions may contain repetition or grammatical issues due to LSTM limitations.

🚀 Future Improvements

1. Use Beam Search instead of greedy decoding
2. Train on a larger dataset like MS COCO
3. Replace LSTM with Transformer-based models
4. Improve grammar and caption accuracy
5. Add confidence scores for predictions

🎓 Learning Outcomes

- Understanding CNN + LSTM architecture
- Building an end-to-end ML web application
- Handling legacy model compatibility
- Flask backend integration with ML models



