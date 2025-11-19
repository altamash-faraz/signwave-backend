import os
import json
import cv2
import numpy as np
import mediapipe as mp
import base64
import tensorflow as tf
from PIL import Image
from io import BytesIO
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from dotenv import load_dotenv
import pyttsx3
import tempfile
import threading

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
MODEL_PATH = os.getenv('MODEL_PATH', './models/alphanumeric_model.h5')
LABELS_PATH = os.getenv('LABELS_PATH', './models/labels.json')
CONF_THRESHOLD = float(os.getenv('CONF_THRESHOLD', '0.5'))
MAX_HANDS = int(os.getenv('MAX_HANDS', '2'))

# Global variables
model = None
labels_map = None
idx_to_label = None
mp_hands = None
tts_engine = None


def init_model():
    """Initialize model, labels, mediapipe, TTS"""
    global model, labels_map, idx_to_label, mp_hands, tts_engine

    try:
        # Load TensorFlow model
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"✅ Model loaded from {MODEL_PATH}")

        # Load labels
        with open(LABELS_PATH, "r") as f:
            labels_map = json.load(f)
        idx_to_label = {v: k for k, v in labels_map.items()}
        print(f"✅ Labels loaded: {len(labels_map)} classes")

        # Init MediaPipe Hands
        mp_hands = mp.solutions.hands
        print("✅ MediaPipe initialized")

        # Init TTS (skip if not available on server)
        try:
            tts_engine = pyttsx3.init()
            tts_engine.setProperty('rate', 150)
            tts_engine.setProperty('volume', 0.9)
            print("✅ TTS engine initialized")
        except Exception as tts_error:
            print(f"⚠️  TTS not available (normal on servers): {str(tts_error)}")
            tts_engine = None

        return True
    except Exception as e:
        print(f"❌ Error initializing model: {str(e)}")
        return False


# Initialize model at module level (runs on import - works with Gunicorn)
print("🚀 Starting SignWave Backend API...")
init_success = init_model()
if init_success:
    print(f"✅ Initialization complete - {len(labels_map)} gesture classes loaded")
else:
    print("⚠️  Model initialization incomplete - some features may not work")


def extract_hand_features(img_bgr, hands_detector, max_hands=2):
    """Extract features using mediapipe"""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(img_rgb)
    out = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            lm = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)
            out.append(lm)

        # Sort left → right
        out.sort(key=lambda arr: float(np.mean(arr[:, 0])))
        out = out[:max_hands]

    # Flatten to feature vector
    feat = np.zeros((max_hands * 21 * 3 + 1,), dtype=np.float32)

    for i, hand in enumerate(out):
        start = i * 21 * 3
        feat[start:start + 21 * 3] = hand.reshape(-1)

    feat[-1] = float(len(out))  # number of hands

    return feat.reshape(1, -1), results


def base64_to_image(base64_string):
    """Convert base64 → OpenCV image"""
    try:
        if 'base64,' in base64_string:
            base64_string = base64_string.split('base64,')[1]

        image_data = base64.b64decode(base64_string)
        image = Image.open(BytesIO(image_data))

        img_array = np.array(image)
        if len(img_array.shape) == 3:
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = img_array

        return img_bgr
    except Exception as e:
        print(f"❌ Error converting base64 to image: {str(e)}")
        return None


# ------------------------------------------------------
# ROUTES
# ------------------------------------------------------

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'labels_loaded': labels_map is not None,
        'mediapipe_ready': mp_hands is not None,
        'total_classes': len(labels_map) if labels_map else 0
    })


@app.route('/api/detect', methods=['POST'])
def detect_gesture():
    try:
        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 400

        data = request.get_json()
        if 'image' not in data:
            return jsonify({'error': 'Missing image field'}), 400

        img = base64_to_image(data['image'])
        if img is None:
            return jsonify({'error': 'Invalid image data'}), 400

        # MediaPipe hands detector
        with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=MAX_HANDS,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as hands_detector:

            feat, results = extract_hand_features(img, hands_detector, max_hands=MAX_HANDS)

            response = {
                'hands_detected': False,
                'prediction': None,
                'confidence': 0.0,
                'landmarks': []
            }

            if results.multi_hand_landmarks:
                response['hands_detected'] = True

                landmarks = []
                for hand_landmarks in results.multi_hand_landmarks:
                    hand_points = []
                    for lm in hand_landmarks.landmark:
                        hand_points.append({
                            'x': lm.x,
                            'y': lm.y,
                            'z': lm.z
                        })
                    landmarks.append(hand_points)

                response['landmarks'] = landmarks

                preds = model.predict(feat, verbose=0)[0]
                idx = int(np.argmax(preds))
                conf = float(preds[idx])

                if conf >= CONF_THRESHOLD:
                    response['prediction'] = idx_to_label[idx]
                else:
                    response['prediction'] = 'uncertain'

                response['confidence'] = conf

            return jsonify(response)

    except Exception as e:
        print(f"❌ Error in detect_gesture: {str(e)}")
        return jsonify({'error': f'Processing error: {str(e)}'}), 500


@app.route('/api/text-to-speech', methods=['POST'])
def text_to_speech():
    try:
        if tts_engine is None:
            return jsonify({'error': 'TTS engine not initialized'}), 500

        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 400

        data = request.get_json()
        if 'text' not in data:
            return jsonify({'error': 'Missing text field'}), 400

        text = data['text'].strip()
        if not text:
            return jsonify({'error': 'Text cannot be empty'}), 400

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
            temp_wav_path = temp_wav.name

        try:
            tts_engine.save_to_file(text, temp_wav_path)
            tts_engine.runAndWait()

            if not os.path.exists(temp_wav_path) or os.path.getsize(temp_wav_path) == 0:
                raise Exception("Failed to generate WAV file")

            def remove_temp():
                try:
                    os.unlink(temp_wav_path)
                except:
                    pass

            threading.Timer(5.0, remove_temp).start()

            return send_file(
                temp_wav_path,
                mimetype='audio/wav',
                as_attachment=True,
                download_name='speech.wav'
            )

        except Exception as e:
            try:
                os.unlink(temp_wav_path)
            except:
                pass
            raise e

    except Exception as e:
        print(f"❌ TTS error: {str(e)}")
        return jsonify({'error': f'TTS processing error: {str(e)}'}), 500


@app.route('/api/model/info', methods=['GET'])
def get_model_info():
    try:
        return jsonify({
            'model_path': MODEL_PATH,
            'labels_path': LABELS_PATH,
            'total_classes': len(labels_map),
            'confidence_threshold': CONF_THRESHOLD,
            'max_hands': MAX_HANDS,
            'input_shape': model.input_shape,
            'output_shape': model.output_shape
        })
    except Exception as e:
        return jsonify({'error': f'Error getting model info: {str(e)}'}), 500


@app.route('/api/labels', methods=['GET'])
def get_labels():
    try:
        return jsonify({
            'labels': list(labels_map.keys()),
            'total_count': len(labels_map),
            'label_mapping': labels_map
        })
    except Exception as e:
        return jsonify({'error': f'Error getting labels: {str(e)}'}), 500


# ------------------------------------------------------
# MAIN ENTRY POINT (for local development with `python app.py`)
# ------------------------------------------------------
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))
    print(f"🔥 Development server running on port {port}")
    app.run(host='0.0.0.0', port=port, debug=True)