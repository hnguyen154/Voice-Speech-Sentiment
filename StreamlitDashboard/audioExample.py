import streamlit as st
import azure.cognitiveservices.speech as s
import os
import IPython
import speech_recognition as sr

st.write("""
# IMPORT AUDIO
""")


path = 'C:/Users/hongh/Documents/GitHub/YouShowItWeKnowIt/Transcription/5793_9812_bundle_archive/cv-valid-train/converted/'
audio = 'sample-000533.wav'

audio_file = open(path+audio, 'rb')
audio_bytes = audio_file.read()
st.audio(audio_bytes, format='audio/wax')

    
def transcribe(path, audio_file):
    r = sr.Recognizer()
    thisFile = sr.AudioFile(path+audio_file)
    with thisFile as source:
        audio_data = r.record(source)
        try:
            text = r.recognize_google(audio_data)
            return text
        except sr.UnknownValueError as u:
            return u
        except sr.RequestError as e:
            return "Could not request results from Google Cloud Speech Recognition service; {0}".format(e)  


if st.button('What did he say?'):
    sentence = transcribe(path, audio)
    st.write(sentence)
else:
    st.write('')
    

from streamlit import caching
caching.clear_cache()
    
    