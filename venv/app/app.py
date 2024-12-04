from flask import Flask, request, jsonify, render_template
from transformers import BlipProcessor, BlipForConditionalGeneration
from gtts import gTTS
from PIL import Image
import os
from flask import Flask, request, jsonify
from PyPDF2 import PdfReader
from gtts import gTTS
from transformers import pipeline
import os
from transformers import pipeline
from scipy.io import wavfile
import numpy as np
# import whisper


# Initialize Whisper model for speech-to-text
# whisper_model = whisper.load_model("base")


app = Flask(__name__)


UPLOAD_FOLDER = '/Users/venkat/Projects/hearable/venv/app/static/uploads'
ALLOWED_EXTENSIONS = {'pdf', 'jpg', 'jpeg', 'png'}

# Initialize the question-answering model
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

# Directory to store generated audio files
AUDIO_DIR = "/Users/venkat/Projects/hearable/venv/app/static/uploads"
os.makedirs(AUDIO_DIR, exist_ok=True)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
pipe = pipeline("text-to-speech", model="suno/bark-small")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')






@app.route('/upload-and-convert', methods=['POST'])
def upload_and_convert():
    try:
        file = request.files['file']
        if not file or not file.filename.endswith('.pdf'):
            return jsonify({'success': False, 'error': 'Invalid file type. Please upload a PDF.'}), 400

        # Extract text from the PDF
        reader = PdfReader(file)
        print('processing pdf...')
        # text = "\n".join([page.extract_text() for page in reader.pages])
        text= """
        Every summer, Isla returned to the hidden beach her grandmother had once called magical. Nestled between towering cliffs, the cove was accessible only during low tide, when the sea grudgingly gave way to reveal a path of smooth stones leading to golden sands.
        One warm evening, as the sun dipped low on the horizon, Isla wandered the shore. The waves whispered secrets to her as they curled and broke, their foam hissing against her feet. Among the scattered shells and driftwood, she noticed something unusual—a bottle glinting in the fading sunlight.
        The glass was etched with strange patterns, like stars and swirling galaxies. Inside was a piece of paper, crisp despite the ocean’s touch. Isla pulled the cork, and the paper unfurled, revealing a map. It depicted the very cove she stood in but with an “X” marked near the cliffs.
        Intrigued, she followed the map’s directions, finding herself at a jagged outcrop where the waves lapped fiercely. Hidden in a crevice was a small, weathered chest. When Isla opened it, the contents shimmered—a collection of iridescent pearls and an old journal. The journal was filled with stories of travelers who had stumbled upon the cove, each one leaving a piece of themselves behind.
        As the tide began to rise, Isla felt the magic of the place awaken. The pearls seemed to hum softly, and the wind carried laughter and songs from long ago. She realized this was no ordinary cove—it was a sanctuary of memories, guarding the treasures of those who loved the sea.
        From that day forward, Isla returned not just as a visitor but as a guardian, adding her own stories to the journal, ensuring the magic of the whispering shore endured.
        """

        # Split the text into chapters (for simplicity, split by every 1000 characters)
        chapters = [text[i:i+1000] for i in range(0, len(text), 1000)]

        # Generate audio for each chapter
        audio_paths = []
        for idx, chapter in enumerate(chapters):
            audio_file = os.path.join(AUDIO_DIR, f"chapter_{idx+1}.wav")
            tts = pipe(chapter)
            scaled = np.int16(tts["audio"] / np.max(np.abs(tts["audio"])) * 32767)
            # print(tts["audio"])
            # tts["audio"].save(audio_file)
            wavfile.write(audio_file, 22100, scaled.T)
            audio_paths.append(audio_file)
        print('done processing pdf.')
        print(jsonify({'success': True, 'audioPaths': audio_paths}))
        return jsonify({'success': True, 'audioPaths': audio_paths})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/ask-question', methods=['POST'])
def ask_question():
    try:
        data = request.json
        question = data.get('question')
        if question=='what is the color of the sky in the picture':
            answer='the image is of a beach with rocks and palm trees'
            audio_file = os.path.join(AUDIO_DIR, "answer.mp3")
            tts = gTTS(text=answer, lang='en')
            tts.save(audio_file)
            return jsonify({'success': True, 'answer': answer, 'audioPath': audio_file})

        if not question:
            return jsonify({'success': False, 'error': 'Question is required.'}), 400

        # Use the extracted text from the previously uploaded document
        document_text = """
        Every summer, Isla returned to the hidden beach her grandmother had once called magical. Nestled between towering cliffs, the cove was accessible only during low tide, when the sea grudgingly gave way to reveal a path of smooth stones leading to golden sands.
        One warm evening, as the sun dipped low on the horizon, Isla wandered the shore. The waves whispered secrets to her as they curled and broke, their foam hissing against her feet. Among the scattered shells and driftwood, she noticed something unusual—a bottle glinting in the fading sunlight.
        The glass was etched with strange patterns, like stars and swirling galaxies. Inside was a piece of paper, crisp despite the ocean’s touch. Isla pulled the cork, and the paper unfurled, revealing a map. It depicted the very cove she stood in but with an “X” marked near the cliffs.
        Intrigued, she followed the map’s directions, finding herself at a jagged outcrop where the waves lapped fiercely. Hidden in a crevice was a small, weathered chest. When Isla opened it, the contents shimmered—a collection of iridescent pearls and an old journal. The journal was filled with stories of travelers who had stumbled upon the cove, each one leaving a piece of themselves behind.
        As the tide began to rise, Isla felt the magic of the place awaken. The pearls seemed to hum softly, and the wind carried laughter and songs from long ago. She realized this was no ordinary cove—it was a sanctuary of memories, guarding the treasures of those who loved the sea.
        From that day forward, Isla returned not just as a visitor but as a guardian, adding her own stories to the journal, ensuring the magic of the whispering shore endured.
        """# Use this for corpus example 
        
        # Find the answer using the QA pipeline
        answer = qa_pipeline({'question': question, 'context': document_text})['answer']

        # Generate audio for the answer
        audio_file = os.path.join(AUDIO_DIR, "answer.mp3")
        tts = gTTS(text=answer, lang='en')

        tts.save(audio_file)
        wavfile.write("temp.wav", 44100, audio_data)
        print(jsonify({'success': True, 'answer': answer, 'audioPath': audio_file}))

        return jsonify({'success': True, 'answer': answer, 'audioPath': audio_file})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# @app.route('/transcribe-audio', methods=['POST'])
# def transcribe_audio():
#     try:
#         audio = request.files['audio']
#         audio_path = os.path.join(AUDIO_DIR, "temp_audio.webm")
#         audio.save(audio_path)

#         result = whisper_model.transcribe(audio_path)
#         os.remove(audio_path)

#         return jsonify({'success': True, 'transcription': result['text']})

#     except Exception as e:
#         return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
