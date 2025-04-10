from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import uuid
from deepfake_detection_engine import init, process
from werkzeug.wrappers import Response

app = Flask(__name__)
CORS(app)  # Enable CORS
app.config['UPLOAD_FOLDER'] = './temp'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB limit
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov'}

# Define the base directory for your project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Initialize the model once at startup using relative paths
MODELS_DIR = os.path.join(BASE_DIR, 'weights')    # Assumes a folder named "weights" in your project
CFG_FILE = os.path.join(BASE_DIR, 'config.json')    # Assumes "config.json" is located in your project root
DEVICE = 'cpu'  # Use 'cpu' if no GPU

init(MODELS_DIR, CFG_FILE, DEVICE)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/detect', methods=['POST'])
def detect_deepfake():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    # Save the file temporarily
    temp_filename = str(uuid.uuid4()) + '_' + secure_filename(file.filename)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
    file.save(save_path)
    
    try:
        score = process(save_path)
        # Ensure score is a Python float
        score = float(score)
        # Convert NumPy boolean to Python boolean
        is_deepfake = bool(score > 0.5)
        os.remove(save_path)
        return jsonify({
            'score': score,
            'is_deepfake': is_deepfake
        })
    except Exception as e:
        os.remove(save_path)
        return jsonify({'error': str(e)}), 500

def handler(request):
    # The 'Response.from_app' helper converts your Flask WSGI app into a proper response.
    return Response.from_app(app, request.environ)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
