import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from PIL import Image
import io
import base64
import pickle

app = Flask(__name__)

MODEL_PATH = 'model/emnist_cnn.h5'
MAPPING_PATH = 'model/mapping.pkl'
model = None
ems_mapping = None

def load_model():
    global model, ems_mapping
    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(MAPPING_PATH):
            print(f"Loading trained EMNIST model from {MODEL_PATH}...")
            model = tf.keras.models.load_model(MODEL_PATH)
            with open(MAPPING_PATH, 'rb') as f:
                ems_mapping = pickle.load(f)
            print("Model loaded successfully!")
            return True
        else:
            print(f"Files not found: {MODEL_PATH} or {MAPPING_PATH}")
            return False
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

# Initial load
load_model()

def preprocess_and_segment(image):
    # Convert PIL Image to RGB first to ensure consistency
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    img = np.array(image)
    
    # Convert RGB to BGR logic remains
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # BACK TO BASICS: Use Otsu's Thresholding with smart inversion
    # Calculate average brightness
    avg_brightness = np.mean(gray)
    
    # If image is dark (avg < 127), invert it because text is likely light.
    # If image is light, text is dark, so we invert it to make text white (for contour finding).
    if avg_brightness < 127:
        # Dark background, light text.
        # We want Text=White (255), Background=Black (0)
        # So we just carry on, as threshold will make light pixels white.
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        # Light background, dark text.
        # adaptive threshold or check
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    filtered_contours = []
    boundingBoxes = []
    
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        
        # RELAXED FILTERS:
        # A dot is ~4-9 pixels.
        if area < 10: continue 
        if w < 2 or h < 2: continue
        
        filtered_contours.append(c)
        boundingBoxes.append((x, y, w, h))
    
    if not filtered_contours:
        return []

    # SORTING LOGIC:
    # 1. Sort by Y (Top-Down)
    chars = sorted(zip(filtered_contours, boundingBoxes), key=lambda b: b[1][1])
    
    # 2. Group into lines
    lines = []
    current_line = []
    
    if len(chars) > 0:
        current_y = chars[0][1][1]
        last_h = chars[0][1][3]
        # Tolerance: 60% of character height
        y_tolerance = max(10, last_h * 0.6)
        
        for c, box in chars:
            x, y, w, h = box
            if abs(y - current_y) > y_tolerance:
                 # New line
                 lines.append(current_line)
                 current_line = []
                 current_y = y
                 last_h = h
                 y_tolerance = max(10, h * 0.6)
            
            # Update mean Y of the current line? Or just keep anchor?
            # Keeping simple anchor for now.
            current_line.append((c, box))
        
        if current_line:
            lines.append(current_line)
    
    # 3. Sort each line by X (Left-Right)
    final_segments = []
    
    for line in lines:
        # Sort line by X
        line.sort(key=lambda b: b[1][0])
        
        # Add to final
        for c, (x, y, w, h) in line:
            # Crop & Pad
            pad = 4
            y1 = max(0, y-pad)
            y2 = min(thresh.shape[0], y+h+pad)
            x1 = max(0, x-pad)
            x2 = min(thresh.shape[1], x+w+pad)
            
            roi = thresh[y1:y2, x1:x2]
            
            # Simple Resize
            roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
            roi = roi.reshape(28, 28, 1).astype('float32') / 255.0
            final_segments.append(roi)
            
        # Add a special marker for newline in the list?
        # For simple robustness now, let's just create a flattened list of images.
        # To add space visually, we would need to check distance x.
        # Let's keep it simple: Just characters.
        final_segments.append("NEWLINE") 

    return final_segments

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Lazy load attempt
    if model is None:
        success = load_model()
        if not success:
            return jsonify({'error': 'Model could not be loaded.'}), 500

    try:
        data = request.json
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'error': 'No image provided'}), 400

        # Decode base64
        header, encoded = image_data.split(",", 1)
        binary_data = base64.b64decode(encoded)
        image = Image.open(io.BytesIO(binary_data))
        
        # Get segments
        segments = preprocess_and_segment(image)
        
        if not segments:
             return jsonify({'prediction': 'No text detected', 'confidence': '0%'})
        
        # Separate images and markers
        images_to_predict = []
        map_indices = [] # Stores (type, value) where type=0 is pred, type=1 is text
        
        for seg in segments:
            if isinstance(seg, str):
                map_indices.append((1, "\n"))
            else:
                images_to_predict.append(seg)
                map_indices.append((0, len(images_to_predict)-1))
        
        if not images_to_predict:
             return jsonify({'prediction': '?', 'confidence': '0%'})

        # Batch predict
        predictions = model.predict(np.array(images_to_predict))
        
        result_str = ""
        confidences = []
        
        for type_code, val in map_indices:
            if type_code == 1:
                result_str += val # Newline
            else:
                p = predictions[val]
                idx = np.argmax(p)
                conf = float(np.max(p))
                confidences.append(conf)
                
                if ems_mapping:
                    char = ems_mapping.get(idx, '?')
                else:
                    char = str(idx)
                result_str += char

        avg_conf = sum(confidences) / len(confidences) if confidences else 0
        
        return jsonify({
            'prediction': result_str.strip(),
            'confidence': f"{avg_conf*100:.2f}%"
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
