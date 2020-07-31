"""
Demo how to record sound input and save to wave file
"""
import sounddevice as sd
from scipy.io.wavfile import write

fs = 16000  # Sample rate
seconds = 15  # Duration of recording

myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
sd.wait()  # Wait until recording is finished
write('output.wav', fs, myrecording)  # Save as WAV file 


