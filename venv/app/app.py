from flask import Flask, request, jsonify, render_template
from transformers import BlipProcessor, BlipForConditionalGeneration
from gtts import gTTS
from PIL import Image
import os

app = Flask(__name__)
UPLOAD_FOLDER = '/Users/venkat/Projects/hearable/venv/app/uploads'
ALLOWED_EXTENSIONS = {'pdf', 'jpg', 'jpeg', 'png'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files.get('file')
    if file and allowed_file(file.filename):
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        return jsonify({"success": True, "filepath": filepath}), 200
    return jsonify({"success": False, "error": "Invalid file type"}), 400

@app.route('/text-to-audio', methods=['POST'])
def text_to_audio():
    text = request.json.get('text', '')
    if not text:
        return jsonify({"success": False, "error": "Text is required"}), 400
    tts = gTTS(text)
    audio_path = os.path.join(UPLOAD_FOLDER, "audio.mp3")
    tts.save(audio_path)
    return jsonify({"success": True, "audioPath": audio_path}), 200

@app.route('/image-caption', methods=['POST'])
def image_caption():
    image_path = request.json.get('imagePath', '')
    if not os.path.exists(image_path):
        return jsonify({"success": False, "error": "Image not found"}), 400

    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt")
    outputs = model.generate(**inputs)
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    return jsonify({"success": True, "caption": caption}), 200

if __name__ == '__main__':
    app.run(debug=True)
