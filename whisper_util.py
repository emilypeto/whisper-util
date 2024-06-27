import argparse
from openai import OpenAI
import os

def transcribe_audio(file_path, api_key):
    if not os.path.exists(file_path):
        print(f"Error: The file at path '{file_path}' does not exist.")
        return

    client = OpenAI(api_key=api_key)
    with open(file_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="text"
        )

    # Save the transcription to a text file
    basename, _ = os.path.splitext(file_path)
    output_path = basename + '.txt'
    with open(output_path, 'w') as output_file:
        output_file.write(transcription)
        print(f"Transcription saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe an audio file using OpenAI's Whisper model")
    parser.add_argument('--path', type=str, required=True, help="Path to the audio file to transcribe")
    parser.add_argument('--api-key', type=str, required=True, help="OpenAI API key")
    
    args = parser.parse_args()
    transcribe_audio(args.path, args.api_key)