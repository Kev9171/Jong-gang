import speech_recognition as sr
from gtts import gTTS 
import os 
import time 
from playsound import playsound

def stt():
    r = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        r.adjust_for_ambient_noise(source)
        print("Record start")
        audio = r.listen(source,phrase_time_limit=10)

    #    text = r.recognize_houndify(audio_data = audio, client_id = "fgGCYTsipn34hDTLCx2I5w==", client_key = "zMQ4hwV83-pCYDgdfdpF9H2todhXyTNSgW1uQuEXO25qrUwODy7cOvucl6USZkCgRmhaWeXbKz1QXC5pLpinYg==")
        text = r.recognize_google(audio_data = audio, language= "ko-kr")
        print(text)
        return text

def tts(text):
    tts = gTTS(text=text, lang='ko')
    filename='voice.mp3'
    tts.save(filename)
    playsound(filename)

tts(stt())
