import argparse
from google.cloud import speech
import io

def transcribe_audio(audio_file_path):
    client = speech.SpeechClient()

    with io.open(audio_file_path, "rb") as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,  # or change if needed
        sample_rate_hertz=16000,  # adjust if your file uses different rate
        language_code="en-US",
    )

    response = client.recognize(config=config, audio=audio)

    transcript = ""
    for result in response.results:
        transcript += result.alternatives[0].transcript + "\n"

    return transcript

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", nargs='?')
    args = parser.parse_args()

    if args.input_file:
        audio_file_path = args.input_file
        text = transcribe_audio(audio_file_path)

        with open("transcribed_text.txt", "w") as file:
            file.write(text)
    else:
        print("Please provide the path to the input file.")

if __name__ == "__main__":
    main()
