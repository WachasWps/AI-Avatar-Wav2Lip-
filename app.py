from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import google.generativeai as genai
from gtts import gTTS
import subprocess
from moviepy import *
from moviepy.editor import VideoFileClip, AudioFileClip
import numpy as np


app = Flask(__name__)  # Fix: Corrected __name__ here
CORS(app)

# Configure Gemini API
genai.configure(api_key="AIzaSyBtpKzAxx2pwMQ1eMO_jtRxk28rRaglVc0")  # Replace with your Gemini API key

# Paths
VIDEO_PATH = "avatr.webp"  # Replace with your avatar video path
AUDIO_PATH = "generated_audio.wav"  # Path for generated audio
OUTPUT_VIDEO = "final_avatar_video.mp4"  # Final output path
WAV2LIP_CHECKPOINT = "Wav2Lip/checkpoints/wav2lip_gan.pth"  # Wav2Lip checkpoint path

# Helper: Generate audio using text-to-speech
def generate_audio(text, output_path):
    tts = gTTS(text=text, lang='en')
    tts.save(output_path)
    return output_path

# Helper: Perform lip sync using Wav2Lip
def perform_lip_sync(video_path, audio_path, output_path):
    command = [
        "python", "inference.py",  # Ensure the path to inference.py is correct
        "--checkpoint_path", WAV2LIP_CHECKPOINT,
        "--face", video_path,
        "--audio", audio_path,
        "--outfile", output_path
    ]
    subprocess.run(command, check=True)
    return output_path

# Helper: Query Gemini API
def get_gemini_response(question):
    try:
        model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest")
        response = model.generate_content([question])
        return response.text.strip() if response and response.text else "I'm sorry, I couldn't generate a response."
    except Exception as e:
        print(f"Gemini Error: {e}")
        return "I'm sorry, I couldn't generate a response."

@app.route('/ask', methods=['POST'])
def ask_avatar():
    try:
        # Get question from request
        question = request.json.get("question")
        if not question:
            return jsonify({"error": "No question provided"}), 400

        # Generate response using Gemini
        response_text = get_gemini_response(question)

        # Generate audio for the response
        generate_audio(response_text, AUDIO_PATH)

        # Perform lip sync with the avatar video
        lip_synced_video = "lip_synced_video.mp4"
        perform_lip_sync(VIDEO_PATH, AUDIO_PATH, lip_synced_video)

        # Combine video and audio
        final_video = OUTPUT_VIDEO
        video_clip = VideoFileClip(lip_synced_video)
        audio_clip = AudioFileClip(AUDIO_PATH)
        video_clip.set_audio(audio_clip).write_videofile(final_video, codec="libx264")

        return jsonify({
            "response": response_text,
            "video_path": final_video
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':  # Fix: Corrected if condition to __name__ == '__main__'
    app.run(host="0.0.0.0", port=5600)
