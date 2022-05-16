import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import urllib.request
import time
import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow import keras

data = pd.read_csv('ChatBotData.csv')

f = open("KETI_conv_kr.txt","r",encoding='utf8')
lines = f.readlines()

Q = []
A = []

for i in range(len(lines)) :
    if i%2 == 0 :
        Q.append(lines[i][2:-1])
        A.append(lines[i+1][2:-1])

import pandas as pd
df = pd.DataFrame()

df['Q'] = Q
df['A'] = A
df['label'] = 1

train_data = pd.concat([data, df],ignore_index=True)
train_data = train_data.sample(frac=1).reset_index(drop=True)

# 특수기호 띄어쓰기
def preprocess_sentence(sentence):
  sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
  sentence = re.sub(r'[" "]+', " ", sentence)
  sentence = sentence.strip()
  return sentence

questions = []
answers = []
for sentence in train_data['Q']:
    questions.append(preprocess_sentence(sentence))

for sentence in train_data['A']:
    answers.append(preprocess_sentence(sentence))

    #서브워드텍스트인코더 사용
tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
    questions + answers, target_vocab_size=2**13)
    

START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]
VOCAB_SIZE = tokenizer.vocab_size + 2

print('임의의 질문 샘플을 정수 인코딩 : {}'.format(tokenizer.encode(questions[20])))

MAX_LENGTH = 40

# 정수 인코딩 & 패딩
def tokenize_and_filter(inputs, outputs):
  tokenized_inputs, tokenized_outputs = [], []

  for (sentence1, sentence2) in zip(inputs, outputs):
    # encode(토큰화 + 정수 인코딩), 시작 토큰과 종료 토큰 추가
    sentence1 = START_TOKEN + tokenizer.encode(sentence1) + END_TOKEN
    sentence2 = START_TOKEN + tokenizer.encode(sentence2) + END_TOKEN

    tokenized_inputs.append(sentence1)
    tokenized_outputs.append(sentence2)

  # 패딩
  tokenized_inputs = tf.keras.preprocessing.sequence.pad_sequences(
      tokenized_inputs, maxlen=MAX_LENGTH, padding='post')
  tokenized_outputs = tf.keras.preprocessing.sequence.pad_sequences(
      tokenized_outputs, maxlen=MAX_LENGTH, padding='post')

  return tokenized_inputs, tokenized_outputs

def create_padding_mask(x):
  mask = tf.cast(tf.math.equal(x, 0), tf.float32)
  # (batch_size, 1, 1, key의 문장 길이)
  return mask[:, tf.newaxis, tf.newaxis, :]

# 디코더의 첫번째 서브층(sublayer)에서 미래 토큰을 Mask하는 함수
def create_look_ahead_mask(x):
  seq_len = tf.shape(x)[1]
  look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
  padding_mask = create_padding_mask(x) # 패딩 마스크도 포함
  return tf.maximum(look_ahead_mask, padding_mask)

#encoder
def encoder(vocab_size, num_layers, dff,
            d_model, num_heads, dropout,
            name="encoder"):
  inputs = tf.keras.Input(shape=(None,), name="inputs")

  # 인코더는 패딩 마스크 사용
  padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

  # 포지셔널 인코딩 + 드롭아웃
  embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
  embeddings *= tf.keras.layers.Lambda(lambda d_model: tf.math.sqrt(tf.cast(d_model, tf.float32)))(d_model)
  embeddings = PositionalEncoding(vocab_size,d_model)(embeddings)
  outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)

  # 인코더를 num_layers개 쌓기
  for i in range(num_layers):
    outputs = encoder_layer(dff=dff, d_model=d_model, num_heads=num_heads,
        dropout=dropout, name="encoder_layer_{}".format(i),
    )([outputs, padding_mask])

  return tf.keras.Model(
      inputs=[inputs, padding_mask], outputs=outputs, name=name)

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
from playsound import playsound

idx = 0

def stt():
    r = sr.Recognizer()
    mic = sr.Microphone()
    try:
        with mic as source:
            r.adjust_for_ambient_noise(source)
            print("Record start")
            audio = r.listen(source,phrase_time_limit=10)

        #    text = r.recognize_houndify(audio_data = audio, client_id = "fgGCYTsipn34hDTLCx2I5w==", client_key = "zMQ4hwV83-pCYDgdfdpF9H2todhXyTNSgW1uQuEXO25qrUwODy7cOvucl6USZkCgRmhaWeXbKz1QXC5pLpinYg==")
            text = r.recognize_google(audio_data = audio, language= "ko-kr")
            print(text)
            return text
    except Exception as e:
        print(str(e))

def tts(text):
    global idx
    tts = gTTS(text=text, lang='ko')
    filename='voice' + str(idx) + '.mp3'
    tts.save(filename)
    playsound(filename )
    time.sleep(2)
    os.remove('voice' + str(idx ) + '.mp3')
    idx += 1

output = predict(stt())

tts(output)