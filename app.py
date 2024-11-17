import os
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from pydub import AudioSegment

# Initialize Flask app
app = Flask(__name__)

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a', 'ogg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def convert_to_wav(input_file, output_file):
    """Convert audio to WAV format using pydub."""
    try:
        # Load audio file
        audio = AudioSegment.from_file(input_file)
        # Export to WAV
        audio.export(output_file, format="wav")
        return True
    except Exception as e:
        print(f"Error during audio conversion: {e}")
        return False

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'file' in request.files:
            return handle_external_upload(request.files['file'])
        elif 'audio_data' in request.files:
            return handle_internal_upload(request.files['audio_data'])
        else:
            return jsonify({"status": "error", "message": "Invalid upload request."}), 400
    return render_template('upload.html')

def handle_external_upload(uploaded_file):
    """Handle external file uploads."""
    if uploaded_file.filename == '':
        return jsonify({"status": "error", "message": "No file selected."}), 400

    if allowed_file(uploaded_file.filename):
        filename = secure_filename(uploaded_file.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        uploaded_file.save(input_path)

        # Convert to WAV if not already WAV
        if not filename.lower().endswith('.wav'):
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{os.path.splitext(filename)[0]}.wav")
            if convert_to_wav(input_path, output_path):
                os.remove(input_path)  # Remove the original file
                return jsonify({"status": "success", "message": f"File converted and saved as: {output_path}"}), 200
            else:
                os.remove(input_path)  # Clean up the temporary file
                return jsonify({"status": "error", "message": "Failed to convert file to WAV."}), 500
        else:
            # If already WAV, keep it as is
            return jsonify({"status": "success", "message": f"File uploaded and saved as: {input_path}"}), 200

    return jsonify({"status": "error", "message": "File type not allowed."}), 400

def handle_internal_upload(file):
    """Handle internal uploads from recording."""
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], "recorded_audio.wav")
    file.save(output_path)
    return jsonify({"status": "success", "message": "Recorded audio saved as WAV!"}), 200

if __name__ == '__main__':
    app.run(debug=True)
