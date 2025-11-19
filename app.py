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
PORT = int(os.getenv('PORT', '5000'))
MAX_HANDS = int(os.getenv('MAX_HANDS', '2'))

# Global variables
model = None
labels_map = None
idx_to_label = None
tts_engine = None

# MediaPipe globals (set at module level)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

def init_model():
    global model, labels_map, idx_to_label, tts_engine
    try:
        # Load TensorFlow model
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"✅ Model loaded from {MODEL_PATH}")

        # Load labels
        with open(LABELS_PATH, "r") as f:
            labels_map = json.load(f)
        idx_to_label = {v: k for k, v in labels_map.items()}
        print(f"✅ Labels loaded: {len(labels_map)} classes")

        # Initialize TTS engine
        tts_engine = pyttsx3.init()
        tts_engine.setProperty('rate', 150)
        tts_engine.setProperty('volume', 0.9)
        print("✅ TTS engine initialized")

        return True
    except Exception as e:
        print(f"❌ Error initializing model: {str(e)}")
        return False

def extract_hand_features(img_bgr, hands_detector, max_hands=2):
    """Extract hand landmark features from image"""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(img_rgb)
    out = []
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            lm = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)
            out.append(lm)
        # Sort left→right
        out.sort(key=lambda arr: float(np.mean(arr[:, 0])))
        out = out[:max_hands]

    # Create feature vector (same format as training)
    feat = np.zeros((max_hands * 21 * 3 + 1,), dtype=np.float32)
    for i, hand in enumerate(out):
        start = i * 21 * 3
        feat[start:start + 21 * 3] = hand.reshape(-1)
    feat[-1] = float(len(out))
    return feat.reshape(1, -1), results

def base64_to_image(base64_string):
    """Convert base64 string to OpenCV image"""
    try:
        # Remove data URL prefix if present
        if 'base64,' in base64_string:
            base64_string = base64_string.split('base64,')[1]
        
        # Decode base64
        image_data = base64.b64decode(base64_string)
        image = Image.open(BytesIO(image_data))
        
        # Convert to OpenCV format
        img_array = np.array(image)
        if len(img_array.shape) == 3:
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = img_array
        
        return img_bgr
    except Exception as e:
        print(f"Error converting base64 to image: {str(e)}")
        return None

# API Endpoints

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    status = {
        'status': 'healthy',
        'model_loaded': model is not None,
        'labels_loaded': labels_map is not None,
        'mediapipe_ready': mp_hands is not None,
        'total_classes': len(labels_map) if labels_map else 0
    }
    return jsonify(status)

@app.route('/api/detect', methods=['POST'])
def detect_gesture():
    """Main gesture detection endpoint"""
    try:
        # Validate request
        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 400
        
        data = request.get_json()
        if 'image' not in data:
            return jsonify({'error': 'Missing image field'}), 400
        
        # Convert base64 to image
        img = base64_to_image(data['image'])
        if img is None:
            return jsonify({'error': 'Invalid image data'}), 400
        
        # Initialize MediaPipe hands detector
        with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=MAX_HANDS,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as hands_detector:
            
            # Extract features
            feat, results = extract_hand_features(img, hands_detector, max_hands=MAX_HANDS)
            
            response = {
                'hands_detected': False,
                'prediction': None,
                'confidence': 0.0,
                'landmarks': []
            }
            
            # Only predict if hands are detected
            if results.multi_hand_landmarks:
                response['hands_detected'] = True
                
                # Get landmarks
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
                
                # Make prediction
                preds = model.predict(feat, verbose=0)[0]
                idx = int(np.argmax(preds))
                conf = float(preds[idx])
                
                if conf >= CONF_THRESHOLD:
                    label = idx_to_label[idx]
                    response['prediction'] = label
                    response['confidence'] = conf
                else:
                    response['prediction'] = 'uncertain'
                    response['confidence'] = conf
            
            return jsonify(response)
    
    except Exception as e:
        print(f"Error in detect_gesture: {str(e)}")
        return jsonify({'error': f'Processing error: {str(e)}'}), 500

@app.route('/api/text-to-speech', methods=['POST'])
def text_to_speech():
    """Convert text to speech and return audio file"""
    try:
        # Check if TTS engine is available
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
        
        # Create temporary WAV file (skip MP3 conversion to avoid pydub issues)
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
            temp_wav_path = temp_wav.name
        
        try:
            # Generate speech using pyttsx3
            print(f"Generating speech for: {text}")
            tts_engine.save_to_file(text, temp_wav_path)
            tts_engine.runAndWait()
            
            # Check if WAV file was created
            if not os.path.exists(temp_wav_path) or os.path.getsize(temp_wav_path) == 0:
                raise Exception("Failed to generate WAV file")
            
            # Return the WAV file directly (most browsers can play WAV)
            def remove_temp_file():
                try:
                    os.unlink(temp_wav_path)
                except:
                    pass
            
            # Schedule cleanup after response is sent
            threading.Timer(5.0, remove_temp_file).start()
            
            return send_file(temp_wav_path, mimetype='audio/wav', as_attachment=True, download_name='speech.wav')
            
        except Exception as e:
            # Cleanup on error
            try:
                os.unlink(temp_wav_path)
            except:
                pass
            raise e
            
    except Exception as e:
        print(f"Error in text_to_speech: {str(e)}")
        return jsonify({'error': f'TTS processing error: {str(e)}'}), 500

@app.route('/api/model/info', methods=['GET'])
def get_model_info():
    """Get model metadata"""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        info = {
            'model_path': MODEL_PATH,
            'labels_path': LABELS_PATH,
            'total_classes': len(labels_map),
            'confidence_threshold': CONF_THRESHOLD,
            'max_hands': MAX_HANDS,
            'input_shape': model.input_shape if model else None,
            'output_shape': model.output_shape if model else None
        }
        return jsonify(info)
    
    except Exception as e:
        return jsonify({'error': f'Error getting model info: {str(e)}'}), 500

@app.route('/api/labels', methods=['GET'])
def get_labels():
    """Get all available gesture labels"""
    try:
        if labels_map is None:
            return jsonify({'error': 'Labels not loaded'}), 500
        
        return jsonify({
            'labels': list(labels_map.keys()),
            'total_count': len(labels_map),
            'label_mapping': labels_map
        })
    
    except Exception as e:
        return jsonify({'error': f'Error getting labels: {str(e)}'}), 500

if __name__ == '__main__':
    print("🚀 Starting SignWave Backend API...")
    
    # Initialize model
    if not init_model():
        print("❌ Failed to initialize model. Exiting.")
        exit(1)
    
    print(f"🔥 Server starting on port {PORT}")
    print(f"📊 Model: {len(labels_map)} gesture classes")
    print(f"🎯 Confidence threshold: {CONF_THRESHOLD}")
    print(f"✋ Max hands: {MAX_HANDS}")
    
    # Run Flask app
    app.run(host='0.0.0.0', port=PORT, debug=True)