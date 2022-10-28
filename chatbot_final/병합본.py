"""
라이브러리 정보
  pandas : 1.1.5
  numpy : 1.19.5
  tensorflow : 2.4.0
  tensorflow_datasets : 4.5.2
  RPi.GPIO : 라즈베리파이 내장 버전
  SpeechRecognition : 3.8.1
  gTTs : 2.2.4
  pygame : 2.1.2
"""

import pandas as pd
import re
import time
import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow import keras

tokenizer = tfds.deprecated.text.SubwordTextEncoder.load_from_file("tokenizer")

START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]
VOCAB_SIZE = tokenizer.vocab_size + 2

MAX_LENGTH = 40

def preprocess_sentence(sentence):
  sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
  sentence = re.sub(r'[" "]+', " ", sentence)
  sentence = sentence.strip()
  return sentence

def create_padding_mask(x):
  mask = tf.cast(tf.math.equal(x, 0), tf.float32)
  # (batch_size, 1, 1, key의 문장 길이)
  return mask[:, tf.newaxis, tf.newaxis, :]

def evaluate(sentence):
  sentence = preprocess_sentence(sentence)

  sentence = tf.expand_dims(
      START_TOKEN + tokenizer.encode(sentence) + END_TOKEN, axis=0)

  output = tf.expand_dims(START_TOKEN, 0)

  # 디코더의 예측 시작
  for i in range(MAX_LENGTH):
    predictions = model(inputs=[sentence, output], training=False)

    # 현재(마지막) 시점의 예측 단어를 받아온다.
    predictions = predictions[:, -1:, :]
    predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

    # 만약 마지막 시점의 예측 단어가 종료 토큰이라면 예측을 중단
    if tf.equal(predicted_id, END_TOKEN[0]):
      break

    # 마지막 시점의 예측 단어를 출력에 연결한다.
    # 이는 for문을 통해서 디코더의 입력으로 사용될 예정이다.
    output = tf.concat([output, predicted_id], axis=-1)

  return tf.squeeze(output, axis=0)

def predict(sentence):
  prediction = evaluate(sentence)

  predicted_sentence = tokenizer.decode(
      [i for i in prediction if i < tokenizer.vocab_size])

  print('Input: {}'.format(sentence))
  print('Output: {}'.format(predicted_sentence))

  return predicted_sentence


model = keras.models.load_model("saved_model", custom_objects={'create_padding_mask': create_padding_mask})

import speech_recognition as sr
from gtts import gTTS 
import os 
import time 
import pygame

pygame.mixer.init()

idx = 0

def stt():
    r = sr.Recognizer()
    mic = sr.Microphone()
    try:
        with mic as source:
            r.adjust_for_ambient_noise(source)
            print("Record Start")
            audio = r.listen(source,phrase_time_limit=10)
            text = r.recognize_google(audio_data = audio, language= "ko-kr")
            print("Record Stop")
            return text
    except Exception as e:
        print(str(e))

def tts(text):
    global idx
    tts = gTTS(text=text, lang='ko')
    filename='voice' + str(idx) + '.mp3'
    tts.save(filename)
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()
    time.sleep(2)
    pygame.mixer.music.unload()
    os.remove(filename)
    idx += 1

import RPi.GPIO as GPIO

ledR = 19
ledG = 26
btn = 21

GPIO.setmode(GPIO.BCM)
GPIO.setup(ledR, GPIO.OUT)
GPIO.setup(ledG, GPIO.OUT)
GPIO.setup(btn, GPIO.IN, pull_up_down = GPIO.PUD_UP)

r = True

while True:
  if (r):
    GPIO.output(ledG, GPIO.HIGH)
    r = False
  isBtnPushed = GPIO.input(btn)
  if (isBtnPushed):
    GPIO.output(ledG, GPIO.LOW)
    GPIO.output(ledR, GPIO.HIGH)
    output = predict(stt())
    tts(output)
    GPIO.output(ledR, GPIO.LOW)
    r = True
    
    