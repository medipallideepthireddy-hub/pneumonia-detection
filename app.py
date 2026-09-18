"""
AI-Based Pneumonia Detection Using Attention-Enhanced CNN
Flask Web Application Backend
"""

import os
import uuid
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template, url_for
from werkzeug.utils import secure_filename

# Import local architecture and Grad-CAM modules
try:
    import keras
    from train import ChannelAttention, SpatialAttention, build_attention_enhanced_cnn
except ImportError:
    from tensorflow import keras
    from train import ChannelAttention, SpatialAttention, build_attention_enhanced_cnn

from gradcam import generate_gradcam_heatmap, overlay_heatmap_on_image

# ==============================================================================
# Flask App Configuration
# ==============================================================================

app = Flask(__name__)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
MODEL_PATH = os.path.join(BASE_DIR, "model", "pneumonia_attention_cnn.keras")

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB limit

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
TARGET_IMAGE_SIZE = (224, 224)
DECISION_THRESHOLD = 0.5

# Global model cache
MODEL = None
MODEL_LOADED_FROM_DISK = False


def allowed_file(filename):
    """Check if uploaded file has an allowed image extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def load_ai_model():
    """
    Loads or initializes the Attention-Enhanced CNN model.
    If the trained weights file exists on disk, it is loaded.
    Otherwise, an initialized Attention-Enhanced CNN is created so the app is always functional.
    """
    global MODEL, MODEL_LOADED_FROM_DISK

    if MODEL is not None:
        return MODEL

    custom_objects = {
        "ChannelAttention": ChannelAttention,
        "SpatialAttention": SpatialAttention,
        "PneumoniaAttention>ChannelAttention": ChannelAttention,
        "PneumoniaAttention>SpatialAttention": SpatialAttention
    }

    if os.path.exists(MODEL_PATH):
        try:
            print(f"[Model Loader] Loading trained model from {MODEL_PATH}...")
            MODEL = keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)
            MODEL_LOADED_FROM_DISK = True
            print("[Model Loader] Successfully loaded model.")
            return MODEL
        except Exception as e:
            print(f"[Model Loader Error] Failed to load model from disk: {e}")

    # Fallback to architecture-initialized model
    print("[Model Loader] Initializing fresh Attention-Enhanced CNN architecture...")
    MODEL = build_attention_enhanced_cnn(input_shape=(TARGET_IMAGE_SIZE[0], TARGET_IMAGE_SIZE[1], 3))
    MODEL.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    MODEL_LOADED_FROM_DISK = False
    return MODEL


def preprocess_image(image_path, target_size=TARGET_IMAGE_SIZE):
    """
    Preprocesses uploaded image for CNN model input:
    - Opens with Pillow
    - Converts to RGB (handles RGBA or Grayscale X-rays)
    - Resizes to target dimension (224, 224)
    - Normalizes pixel values to [0.0, 1.0]
    - Expands batch dimension to (1, 224, 224, 3)
    """
    with Image.open(image_path) as img:
        img_rgb = img.convert("RGB")
        img_resized = img_rgb.resize(target_size, Image.Resampling.BILINEAR)
        img_array = np.array(img_resized, dtype=np.float32) / 255.0
        img_batch = np.expand_dims(img_array, axis=0)
        return img_batch


# ==============================================================================
# Routes
# ==============================================================================

@app.route("/", methods=["GET"])
def index():
    """Renders the main diagnostic dashboard."""
    return render_template("index.html")


@app.route("/model-status", methods=["GET"])
def model_status():
    """Returns AI model runtime diagnostic status."""
    global MODEL_LOADED_FROM_DISK
    model = load_ai_model()
    return jsonify({
        "status": "loaded" if MODEL_LOADED_FROM_DISK else "initialized",
        "model_name": "Attention-Enhanced CNN",
        "model_path": MODEL_PATH if os.path.exists(MODEL_PATH) else None,
        "input_size": list(TARGET_IMAGE_SIZE),
        "threshold": DECISION_THRESHOLD
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Handles image upload, preprocessing, model inference, and Grad-CAM attention heatmap.
    """
    # 1. Check for file in request
    if "file" not in request.files:
        return jsonify({"error": "No file part provided in request."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No image selected for upload."}), 400

    if not allowed_file(file.filename):
        return jsonify({
            "error": "Invalid file format. Supported extensions: JPG, JPEG, PNG."
        }), 400

    # 2. Secure file saving
    try:
        unique_id = uuid.uuid4().hex[:10]
        ext = secure_filename(file.filename).rsplit(".", 1)[1].lower()
        saved_filename = f"{unique_id}_original.{ext}"
        saved_file_path = os.path.join(app.config["UPLOAD_FOLDER"], saved_filename)
        file.save(saved_file_path)
    except Exception as e:
        return jsonify({"error": f"Failed to save uploaded image: {str(e)}"}), 500

    # 3. Image Preprocessing
    try:
        input_tensor = preprocess_image(saved_file_path, target_size=TARGET_IMAGE_SIZE)
    except Exception as e:
        return jsonify({"error": f"Failed to preprocess image: {str(e)}"}), 400

    # 4. Model Inference
    try:
        model = load_ai_model()
        prediction_raw = model.predict(input_tensor, verbose=0)
        prob_pneumonia = float(prediction_raw[0][0])
    except Exception as e:
        return jsonify({"error": f"Inference execution failed: {str(e)}"}), 500

    # 5. Evaluate Classification & Confidence Score
    if prob_pneumonia >= DECISION_THRESHOLD:
        prediction_label = "PNEUMONIA"
        confidence = prob_pneumonia * 100.0
        if confidence >= 85.0:
            message = "The model predicts pneumonia with high confidence."
        else:
            message = "The model predicts pneumonia with moderate confidence."
    else:
        prediction_label = "NORMAL"
        confidence = (1.0 - prob_pneumonia) * 100.0
        if confidence >= 85.0:
            message = "The model predicts a normal chest X-ray with high confidence."
        else:
            message = "The model predicts a normal chest X-ray with moderate confidence."

    confidence = round(confidence, 2)
    raw_score = round(prob_pneumonia, 4)

    # 6. Grad-CAM Attention Visualization
    heatmap_filename = f"{unique_id}_heatmap.png"
    heatmap_output_path = os.path.join(app.config["UPLOAD_FOLDER"], heatmap_filename)
    heatmap_url = None

    try:
        # Generate Grad-CAM targeting the post-attention conv layer
        heatmap = generate_gradcam_heatmap(
            model=model,
            img_array=input_tensor,
            target_layer_name="target_conv_layer",
            pred_index=1 if prob_pneumonia >= DECISION_THRESHOLD else 0
        )

        if heatmap is not None:
            overlay_path = overlay_heatmap_on_image(
                original_img_path=saved_file_path,
                heatmap=heatmap,
                output_path=heatmap_output_path,
                alpha=0.45
            )
            if overlay_path and os.path.exists(overlay_path):
                heatmap_url = url_for("static", filename=f"uploads/{heatmap_filename}")
    except Exception as err:
        print(f"[Warning] Grad-CAM generation encountered issue: {err}")
        heatmap_url = None

    original_url = url_for("static", filename=f"uploads/{saved_filename}")

    # 7. Construct JSON response
    response_data = {
        "prediction": prediction_label,
        "confidence": confidence,
        "raw_score": raw_score,
        "message": message,
        "heatmap_url": heatmap_url,
        "original_url": original_url,
        "threshold": DECISION_THRESHOLD,
        "model_status": "trained" if MODEL_LOADED_FROM_DISK else "initialized"
    }

    return jsonify(response_data), 200


# Error Handlers
@app.errorhandler(413)
def file_too_large(e):
    return jsonify({"error": "File exceeds the 16MB maximum upload limit."}), 413


@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "An internal server error occurred."}), 500


if __name__ == "__main__":
    import argparse
    import socket

    parser = argparse.ArgumentParser(description="Run Pneumonia Detection Web Server")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 5000)), help="Port number (default: 5000)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    args = parser.parse_args()

    port = args.port

    # On macOS, port 5000 is reserved by default for AirPlay Receiver (AirTunes)
    # Check if port is already taken, and switch smoothly to 5001 if needed
    def is_port_in_use(p):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(("127.0.0.1", p)) == 0

    if port == 5000 and is_port_in_use(5000):
        print("\n[Notice] Port 5000 is occupied (standard macOS AirPlay Receiver).")
        print("[Notice] Automatically switching to http://127.0.0.1:5001\n")
        port = 5001

    # Preload model on startup
    load_ai_model()

    print(f"\n=======================================================")
    print(f"  AI Pneumonia Detection Server Live:")
    print(f"  --> http://{args.host}:{port}")
    print(f"=======================================================\n")

    # Run Flask server
    app.run(host=args.host, port=port, debug=False)
