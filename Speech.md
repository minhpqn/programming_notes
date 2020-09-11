# Ghi chép về Speech Processing

Tác giả: Phạm Quang Nhật Minh

Ngày tạo: 9/7/2020

## Cách lấy ra file audio khi biết timestamp

Sử dụng thư viện [pydub](https://github.com/jiaaro/pydub)

```
from pydub import AudioSegment
t1 = t1 * 1000 #Works in milliseconds
t2 = t2 * 1000
newAudio = AudioSegment.from_wav("oldSong.wav")
newAudio = newAudio[t1:t2]
newAudio.export('newSong.wav', format="wav") #Exports to a wav file in the current path.
```

Reference: [How to split a .wav file into multiple .wav files?](https://stackoverflow.com/questions/37999150/how-to-split-a-wav-file-into-multiple-wav-files/43367691#43367691)

## Nhận input âm thanh bằng python

Các thư viện cho phép nhận sound input từ microphone:

- python-sounddevice
- pyaudio
- SpeechRecognition

SpeechRecognition sẽ tự ngắt khi người dùng không nói nữa, các thư viện khác sẽ có tham số
quy định thời gian ghi âm.

Tham khảo:

- [Playing and Recording Sound in Python](https://realpython.com/playing-and-recording-sound-python/#pyaudio_1)
- [The Ultimate Guide To Speech Recognition With Python](https://realpython.com/python-speech-recognition/#installing-speechrecognition)
- [https://gist.github.com/mabdrabo/8678538](https://gist.github.com/mabdrabo/8678538)

