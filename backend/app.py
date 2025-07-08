from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from keras.models import load_model
import torch
import timm
from torchvision import transforms
from PIL import Image
import numpy as np
import io
import os

app = Flask(__name__)
CORS(app)

# === Load Models === #
unet_model = load_model('models/unet_model.h5', compile=False)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
swin_model = timm.create_model("convnext_tiny.fb_in22k", pretrained=False, num_classes=2)
swin_model.load_state_dict(torch.load('models/convnext_tiny.fb_in22k_vmodel.pth', map_location=DEVICE))
swin_model.to(DEVICE)
swin_model.eval()

# === Transforms === #
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# === Folder Setup === #
UPLOAD_FOLDER = 'uploads/'
SEGMENTED_FOLDER = 'segmented_images/'
LEFT_BREAST_FOLDER = 'left_breast_images/'
RIGHT_BREAST_FOLDER = 'right_breast_images/'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SEGMENTED_FOLDER, exist_ok=True)
os.makedirs(LEFT_BREAST_FOLDER, exist_ok=True)
os.makedirs(RIGHT_BREAST_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return "✅ Backend is running. Use POST /upload to test with an image."

def segment_and_classify(image_bytes, filename):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB').resize((256, 256))
    img_np = np.array(img) / 255.0
    img_input = np.expand_dims(img_np, axis=0)

    pred_mask = unet_model.predict(img_input)[0, :, :, 0]
    binary_mask = (pred_mask > 0.5).astype(np.uint8)
    masked_img_np = img_np * binary_mask[..., np.newaxis]

    segmented_image_path = os.path.join(SEGMENTED_FOLDER, f"seg_{filename}")
    segmented_img = Image.fromarray((masked_img_np * 255).astype(np.uint8))
    segmented_img.save(segmented_image_path)

    h, w, _ = masked_img_np.shape
    left_half = masked_img_np[:, :w//2, :]
    right_half = masked_img_np[:, w//2:, :]

    left_path = os.path.join(LEFT_BREAST_FOLDER, f"left_{filename}")
    right_path = os.path.join(RIGHT_BREAST_FOLDER, f"right_{filename}")
    Image.fromarray((left_half * 255).astype(np.uint8)).save(left_path)
    Image.fromarray((right_half * 255).astype(np.uint8)).save(right_path)

    def classify(img_half):
        img_uint8 = (img_half * 255).astype(np.uint8)
        tensor = transform(img_uint8).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = swin_model(tensor)
            probs = torch.nn.functional.softmax(output, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred].item()
        label = "Benign" if pred == 0 else "Malignant"
        return label, confidence

    left_label, left_conf = classify(left_half)
    right_label, right_conf = classify(right_half)

    return {
        "left": left_label,
        "left_confidence": round(left_conf * 100, 2),
        "right": right_label,
        "right_confidence": round(right_conf * 100, 2),
        "segmented_image_url": f'http://localhost:5000/segmented/seg_{filename}',
        "left_breast_image_url": f'http://localhost:5000/left_breast/left_{filename}',
        "right_breast_image_url": f'http://localhost:5000/right_breast/right_{filename}',
        "original_image_url": f'http://localhost:5000/uploads/{filename}'
    }

@app.route("/upload", methods=["POST"])
def upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    filename = file.filename
    image_bytes = file.read()

    try:
        result = segment_and_classify(image_bytes, filename)
        with open(os.path.join(UPLOAD_FOLDER, filename), 'wb') as f:
            f.write(image_bytes)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/segmented/<filename>')
def segmented_file(filename):
    return send_from_directory(SEGMENTED_FOLDER, filename)

@app.route('/left_breast/<filename>')
def left_breast_file(filename):
    return send_from_directory(LEFT_BREAST_FOLDER, filename)

@app.route('/right_breast/<filename>')
def right_breast_file(filename):
    return send_from_directory(RIGHT_BREAST_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)
