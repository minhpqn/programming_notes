"""
Demo how to record sound input and save to wav file with speech_recognition

pip install SpeechRecognition
"""
import speech_recognition as sr

	

r = sr.Recognizer()
with sr.Microphone() as source:
	audio = r.listen(source)

with open("audio_file.wav", "wb") as file:
    file.write(audio.get_wav_data())
