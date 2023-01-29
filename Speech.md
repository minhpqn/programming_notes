# Ghi chép về Speech Processing

Tác giả: Phạm Quang Nhật Minh

Ngày tạo: 9/7/2020

## Convert file mp3 về định dạng wav 16kHz



## ASR Evaluation với sclite

Sử dụng tool: [https://github.com/usnistgov/SCTK](https://github.com/usnistgov/SCTK)

Chuẩn bị reference và hypothesis file theo format

```
Today is sunny (test.wav)
What to do on weekends (test1.wav)
```

Chạy lệnh sau đây:

```
sclite -i wsj -r ref.txt -h hyp.txt -o all -O tmp
```

## Audio Segment với pydub

Segment một file audio thành nhiều chunk với duration xác định với [pydub](http://pydub.com/).

```
from pydub import AudioSegment
t1 = t1 * 1000 #Works in milliseconds
t2 = t2 * 1000
newAudio = AudioSegment.from_wav("oldSong.wav")
newAudio = newAudio[t1:t2]
newAudio.export('newSong.wav', format="wav") #Exports to a wav file in the current path.
```

Tham khảo: [https://stackoverflow.com/questions/37999150/how-to-split-a-wav-file-into-multiple-wav-files](https://stackoverflow.com/questions/37999150/how-to-split-a-wav-file-into-multiple-wav-files)

## Convert file về mono channel

```
ffmpeg -i 111.mp3 -acodec pcm_s16le -ac 1 -ar 16000 out.wav
```

Tham khảo: [https://stackoverflow.com/questions/13358287/how-to-convert-any-mp3-file-to-wav-16khz-mono-16bit](https://stackoverflow.com/questions/13358287/how-to-convert-any-mp3-file-to-wav-16khz-mono-16bit)

## Check Duration of a wav file with librosa

```
samples, sample_rate = librosa.load(sample_file, sr=16000)
duration = librosa.get_duration(y=samples, sr=16000)
```

Tham khảo: [https://librosa.org/doc/main/generated/librosa.get_duration.html](https://librosa.org/doc/main/generated/librosa.get_duration.html)

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

