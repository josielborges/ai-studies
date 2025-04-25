import moviepy as mp
import speech_recognition as sr
import os

def extract_audio_from_video(video_path, audio_path):
    video = mp.VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path)

def transcribe_audio_to_text(audio_path, text_output_path):
    recognizer = sr.Recognizer()
    audio_file = sr.AudioFile(audio_path)

    with audio_file as source:
        audio_data = recognizer.record(source)

    try:
        text = recognizer.recognize_google(audio_data, language='pt-BR')
        
        print('Transcription:\n', text)
       
        with open(text_output_path, 'w') as f:
            f.write(text)
    
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand the audio")

    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_video_path = os.path.join(script_dir, 'data/input_audio_extraction_from_video.mp4')
    audio_output_path = os.path.join(script_dir, 'data/output_audio.wav')
    text_output_path = os.path.join(script_dir, 'data/output_transcription.txt')

    extract_audio_from_video(input_video_path, audio_output_path)
    transcribe_audio_to_text(audio_output_path, text_output_path)

if __name__ == '__main__':
    main()