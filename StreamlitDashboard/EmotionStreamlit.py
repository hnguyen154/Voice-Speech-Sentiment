## This file generates a Streamlit dashboard, and depends on data from GenerateAnalyticsData.py

import streamlit as st
import pandas as pd
import numpy as np
from wordcloud import WordCloud, STOPWORDS
from streamlit import caching
import matplotlib.pyplot as plt
import speech_recognition as sr
caching.clear_cache()

## Header
st.title('UPS Customer Contact Center - Customer Sentiment Detection')

st.subheader('Synthetic Data Generated for Hackathon')
st.write('Select Columns to View')

## Load data
@st.cache(persist=True)
def load_data():
    data = pd.read_csv(r'C:\Users\hongh\Documents\GitHub\Speech-Sentiment-Analysis-\StreamlitDashboard\input\data\AnalyticsData.csv')
    return data
data_load_state = st.text('Loading data...')
data = load_data()
data_load_state.text("Success: Data Loaded")

## Display dataframe and select columns
cols = ["Month", "AudioSent", "TextSent", "Transcript"]
st_ms = st.multiselect("Columns", data.columns.tolist(), default=cols)
st.write(data[st_ms])

## Top 5 Positive and Negative calls by AudioSent
st.subheader('List Top 5 Positive and Negative Calls Based on Audio')
st.write('For Intervention and Prevention')
st.write('Most Positive Calls:')
temp = data.sort_values(by ='TextSentNum' , ascending=False)
st.write(temp[['CallNo', 'TextSentNum']].head())
st.write('Most Negative Calls:')
temp = data.sort_values(by ='TextSentNum', ascending=True)
st.write(temp[['CallNo', 'TextSentNum']].head())

## Select a call, listen to it and read the transcript, and make notes
st.subheader('Select a Call to Review')
selected_call = st.text_input("Call Number?", 0)
st.write(selected_call)

## Play an audio file
st.write('Import Audio')
path = 'C:/Users/hongh/Documents/GitHub/Speech-Sentiment-Analysis-/StreamlitDashboard/input/audio/'

## Because we do not have the actual calls, play a simulated positive or negative call
if data['TextSentNum'][int(selected_call)] < 0:
    audio = 'MissedPickups.wav'
else:
    audio = 'DriverAboveBeyond.wav'

audio_file = open(path+audio, 'rb')
audio_bytes = audio_file.read()
st.audio(audio_bytes, format='audio/wav')

## Transcribe the audio file and write to the interface
def transcribe(path, audio_file):
    r = sr.Recognizer()
    thisFile = sr.AudioFile(path + audio_file)
    with thisFile as source:
        audio_data = r.record(source)
        try:
            text = r.recognize_google(audio_data)
            return text
        except sr.UnknownValueError as u:
            return u
        except sr.RequestError as e:
            return "Could not request results from Google Cloud Speech Recognition service; {0}".format(e)

if st.button('What did she say?'):
    sentence = transcribe(path, audio)
    st.write(sentence)
else:
    st.write('')

## The user can enter notes based on having given attention to a specific call.
notes = st.text_area("Enter Follow-up Notes", '')
st.write(notes)
## Example: "Check with operations, then contact Acme to follow-up on missed pickups"

## Plots of call sentiment and emotion
st.subheader('Call Sentiment by Product')
st.table(data.groupby("Product")['TextSentNum'].mean().reset_index()\
.round(2).sort_values("TextSentNum", ascending=False)\
.assign(MeanTextSentiment=lambda x: x.pop("TextSentNum").apply(lambda y: "%.2f" % y)))

st.subheader('Call Sentiment by Consignee')
st.table(data.groupby("Consignee")['TextSentNum'].mean().reset_index()\
.round(2).sort_values("TextSentNum", ascending=False)\
.assign(MeanTextSentiment=lambda x: x.pop("TextSentNum").apply(lambda y: "%.2f" % y)))

## Call transcript
st.sidebar.subheader("Show transcript of random call by audio emotion")
call_emotion = st.sidebar.radio('AudioSent', ('Positive', 'Neutral', 'Negative'))
st.sidebar.markdown(data.query("AudioSent == @call_emotion")[["Transcript"]].iat[4,0])

## Word clouds by text sentiment
st.sidebar.header("Frequent Words by Emotion")
a = st.sidebar.radio('AudioSent', ('Positive', 'Neutral', 'Negative'), key='wc')
if not st.sidebar.checkbox("Close", False, key='3'):
    st.subheader('Word Cloud for %s Emotion' % (a))
    df = data[data['AudioSent']=='Positive']  ## 'Positive' should be a
    words = ' '.join(df['Words'])
    processed_words = ' '.join([word for word in words.split()])
    wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white', width=800, height=640).generate(processed_words)
    plt.imshow(wordcloud)
    plt.xticks([])
    plt.yticks([])
    st.pyplot()

## Map of sentiment or emotion by month during Covid
st.sidebar.subheader("Months into Covid")
month = st.sidebar.slider("Month", 0, 6)
modified_data = data[data['Month'] == month]

if not st.sidebar.checkbox("Close", False, key='1'):
    st.subheader("How does Sentiment Vary by Time and Place?")
    st.markdown("%i calls between February and June 2020" % (len(modified_data)))
    st.map(data)
