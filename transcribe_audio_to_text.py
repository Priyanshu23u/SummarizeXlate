import whisper
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", nargs='?')
    args = parser.parse_args()
    
    if args.input_file:
        audio_file_path = args.input_file
        model = whisper.load_model("base")
        transcription = model.transcribe(audio_file_path, fp16=False)

        def save_text_to_file():
            with open("transcribed_text.txt", 'w') as file:
                file.write(transcription['text'])

        save_text_to_file()

    else:
        print("Please provide the path to the input file.")

if __name__ == "__main__":
    main()