from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
import base64
from ultralytics import YOLO
import tempfile

app = Flask(__name__)

# Load the parts detection YOLO model
parts_model = YOLO('best6k.pt')  # Replace with your parts detection model path

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
            # Extract bounding box coordinates
            x1, y1, x2, y2 = map(int, box)

            # Draw bounding box on the image
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Get the class name
            part_name = parts_model.names[int(cls)]

            # Initialize severity
            severity = "Unknown"

            # Crop the detected part from the image
            cropped_img = img[y1:y2, x1:x2]
            if cropped_img.size == 0:
                print(f"Warning: Cropped image for class '{part_name}' is empty.")
            else:
                # Save the cropped image to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                    temp_file_path = temp_file.name
                    cv2.imwrite(temp_file_path, cropped_img)

                # Predict severity using the severity model
                severity_results = severity_model.predict(source=temp_file_path, save=False)

                # Assuming the severity model returns classification results
                for sev_result in severity_results:
                    if sev_result.probs is not None and len(sev_result.probs) > 0:
                        top_class = sev_result.probs.top1  # Index of the top class
                        severity = severity_model.names[int(top_class)]
                        break  # Assuming one prediction per cropped image

                # Clean up the temporary file
                os.remove(temp_file_path)

            # Put the text label with part name and severity
            label = f"{part_name} - {severity}"
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)

            # Append the details to the damaged_parts list
            damaged_parts.append({
                'class': part_name,
                'score': float(score),
                'box': [x1, y1, x2, y2],
                'severity': severity
            })

    # Encode the annotated image to Base64
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

    # Secure the filename and save it to the upload folder
    filename = secure_filename(image.filename)
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image.save(image_path)

    try:
        damaged_parts, img_base64 = process_image(image_path)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        # Optionally, remove the uploaded image after processing
        if os.path.exists(image_path):
            os.remove(image_path)

    return jsonify({
        'damaged_parts': damaged_parts,
        'processed_image_base64': img_base64
    }), 200


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
