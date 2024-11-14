from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
import base64
from ultralytics import YOLO
import tempfile

app = Flask(__name__)

# Load the classification model to distinguish between cars and bikes
vehicle_classification_model = YOLO('best22.pt')  # Replace with your vehicle classification model path

# Load the parts detection YOLO model
parts_model = YOLO('besttrainptn.pt')  # Replace with your parts detection model path

# Load the severity classification YOLO model
severity_model = YOLO('best.pt')  # Replace with your severity classification model path

# Define the upload folder for images
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create the upload folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Define the allowed file extensions
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def check_vehicle_type(image_path):
    # Predict if the image is a car or a bike
    results = vehicle_classification_model.predict(source=image_path, save=False)
    # Assuming the classification model provides top class predictions
    for result in results:
        if result.probs is not None and len(result.probs) > 0:
            top_class = result.probs.top1  # Get the index of the top class
            vehicle_type = vehicle_classification_model.names[int(top_class)]
            return vehicle_type
    return "Unknown"

def process_image(image_path):
    # Read the image using OpenCV
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image could not be read. Please check the image file.")

    # Get predictions from the parts detection model
    results = parts_model.predict(source=image_path, save=False)

    damaged_parts = []

    for result in results:
        boxes = result.boxes.xyxy.numpy()  # Bounding boxes
        scores = result.boxes.conf.numpy()  # Confidence scores
        classes = result.boxes.cls.numpy()  # Class indices

        for box, score, cls in zip(boxes, scores, classes):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            part_name = parts_model.names[int(cls)]
            severity = "Unknown"
            cropped_img = img[y1:y2, x1:x2]
            if cropped_img.size > 0:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                    temp_file_path = temp_file.name
                    cv2.imwrite(temp_file_path, cropped_img)

                severity_results = severity_model.predict(source=temp_file_path, save=False)
                for sev_result in severity_results:
                    if sev_result.probs is not None and len(sev_result.probs) > 0:
                        top_class = sev_result.probs.top1
                        severity = severity_model.names[int(top_class)]
                        break

                os.remove(temp_file_path)

            label = f"{part_name} - {severity}"
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)

            damaged_parts.append({
                'class': part_name,
                'score': float(score),
                'box': [x1, y1, x2, y2],
                'severity': severity
            })

    _, buffer = cv2.imencode('.jpg', img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return damaged_parts, img_base64

@app.route('/analyze', methods=['POST'])
def analyze_damage():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part in the request'}), 400

    image = request.files['image']
    if image.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not allowed_file(image.filename):
        return jsonify({'error': 'Unsupported file type'}), 400

    filename = secure_filename(image.filename)
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image.save(image_path)

    try:
        vehicle_type = check_vehicle_type(image_path)
        if vehicle_type.lower() != "car":
            return jsonify({'error': 'Image of a car is required'}), 400

        damaged_parts, img_base64 = process_image(image_path)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(image_path):
            os.remove(image_path)

    return jsonify({
        'damaged_parts': damaged_parts,
        'processed_image_base64': img_base64
    }), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
