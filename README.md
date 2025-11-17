# SignWave Backend

A REST API for real-time sign language recognition with text-to-speech functionality, built with Flask, TensorFlow, and MediaPipe.

## 🚀 Features

- **Sign Language Recognition**: Detects 76 different gestures (A-Z, 0-9, common phrases)
- **Hand Landmark Detection**: Real-time hand tracking using MediaPipe
- **Text-to-Speech**: Converts recognized text to natural speech
- **REST API**: Complete RESTful endpoints for frontend integration
- **CORS Enabled**: Ready for web frontend integration

## 📋 API Endpoints

### Health Check
```
GET /api/health
```
Returns server status and component health.

### Get Gesture Labels
```
GET /api/labels
```
Returns all 76 supported gesture labels.

### Model Information
```
GET /api/model/info
```
Returns model specifications and configuration.

### Gesture Detection
```
POST /api/detect
Content-Type: application/json

{
  "image": "base64_encoded_image_string"
}
```
Detects sign language gestures from base64 encoded images.

**Response:**
```json
{
  "hands_detected": true,
  "prediction": "A",
  "confidence": 0.95,
  "landmarks": [...]
}
```

### Text-to-Speech
```
POST /api/text-to-speech
Content-Type: application/json

{
  "text": "Hello world"
}
```
Converts text to speech and returns WAV audio file.

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/signwave-backend.git
   cd signwave-backend
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the server**
   ```bash
   python app.py
   ```

   The server will start on `http://localhost:5000`

## ☁️ Deployment on Render

### Quick Deploy
1. Fork this repository
2. Connect your GitHub account to [Render](https://render.com)
3. Create a new Web Service
4. Select this repository
5. Configure deployment settings:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn --bind 0.0.0.0:$PORT app:app`
   - **Environment**: Python 3

### Environment Variables (Optional)
Set these in your Render dashboard if you want to customize:
- `PORT`: Server port (default: 5000)
- `CONF_THRESHOLD`: Confidence threshold (default: 0.5)
- `MAX_HANDS`: Maximum hands to detect (default: 2)

## 🌐 Frontend Integration (Vercel)

### For React/Next.js Applications

1. **Install Axios or Fetch for API calls**
   ```bash
   npm install axios
   ```

2. **Set up environment variables in Vercel**
   ```env
   NEXT_PUBLIC_API_URL=https://your-render-app.onrender.com
   ```

3. **Example API integration**
   ```javascript
   // utils/api.js
   const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL;

   // Health check
   export const checkHealth = async () => {
     const response = await fetch(`${API_BASE_URL}/api/health`);
     return response.json();
   };

   // Detect gesture from image
   export const detectGesture = async (base64Image) => {
     const response = await fetch(`${API_BASE_URL}/api/detect`, {
       method: 'POST',
       headers: {
         'Content-Type': 'application/json',
       },
       body: JSON.stringify({ image: base64Image }),
     });
     return response.json();
   };

   // Text to speech
   export const textToSpeech = async (text) => {
     const response = await fetch(`${API_BASE_URL}/api/text-to-speech`, {
       method: 'POST',
       headers: {
         'Content-Type': 'application/json',
       },
       body: JSON.stringify({ text }),
     });
     return response.blob(); // Returns audio file
   };
   ```

4. **Example React component**
   ```jsx
   import { useState } from 'react';
   import { detectGesture, textToSpeech } from '../utils/api';

   export default function SignDetection() {
     const [result, setResult] = useState(null);
     
     const handleImageUpload = async (event) => {
       const file = event.target.files[0];
       if (file) {
         const reader = new FileReader();
         reader.onload = async (e) => {
           const base64 = e.target.result.split(',')[1]; // Remove data URL prefix
           const detection = await detectGesture(base64);
           setResult(detection);
           
           // Play audio if gesture detected
           if (detection.prediction) {
             const audio = await textToSpeech(detection.prediction);
             const audioUrl = URL.createObjectURL(audio);
             const audioElement = new Audio(audioUrl);
             audioElement.play();
           }
         };
         reader.readAsDataURL(file);
       }
     };

     return (
       <div>
         <input type="file" accept="image/*" onChange={handleImageUpload} />
         {result && (
           <div>
             <p>Detected: {result.prediction}</p>
             <p>Confidence: {result.confidence}</p>
           </div>
         )}
       </div>
     );
   }
   ```

### Webcam Integration Example
```jsx
import { useRef, useCallback } from 'react';
import Webcam from 'react-webcam';

export default function LiveDetection() {
  const webcamRef = useRef(null);
  
  const capture = useCallback(async () => {
    const imageSrc = webcamRef.current.getScreenshot();
    const base64 = imageSrc.split(',')[1];
    const result = await detectGesture(base64);
    // Handle result...
  }, [webcamRef]);

  return (
    <div>
      <Webcam ref={webcamRef} />
      <button onClick={capture}>Detect Sign</button>
    </div>
  );
}
```

## 🏗️ Project Structure

```
signwave-backend/
├── app.py              # Main Flask application
├── models/
│   ├── alphanumeric_model.h5  # Trained gesture recognition model
│   └── labels.json            # Gesture labels mapping
├── requirements.txt    # Python dependencies
├── README.md          # This file
└── .gitignore        # Git ignore rules
```

## 🔧 Supported Gestures

The model recognizes 76 different gestures:
- **Numbers**: 0-9
- **Letters**: A-Z (American Sign Language)
- **Common Phrases**: "hello", "please", "thank you", "I am fine", etc.

## 🐛 Troubleshooting

### Common Issues

1. **Server won't start**: Check if all dependencies are installed
2. **Model loading error**: Ensure `models/` directory contains both `.h5` and `.json` files
3. **CORS errors**: The server includes CORS headers for cross-origin requests
4. **TTS not working**: pyttsx3 may require additional system dependencies

### Development Tips

- Use `python app.py` for development
- Check `/api/health` endpoint to verify all components are loaded
- Test with small images first (< 1MB) for faster processing

## 📄 License

This project is licensed under the MIT License.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📞 Support

For issues and questions, please create an issue in the GitHub repository.