import streamlit as st
import numpy as np
import random
import nltk
from nltk.stem.lancaster import LancasterStemmer
import tensorflow as tf
from tensorflow.python.util.nest import is_sequence

import tflearn

import pickle
import json

nltk.download('punkt')
stemmer = LancasterStemmer()

# Load the pickled data
with open("data.pickle", "rb") as f:
    words, labels, training, output = pickle.load(f)

# Load the intents data from JSON file
with open("intents.json", encoding="utf-8") as file:
    data = json.load(file)

# Load the trained model
net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)
model.load("model.tflearn")

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return np.array(bag)


def chat(input_text):
    results = model.predict([bag_of_words(input_text, words)])
    results_index = np.argmax(results)
    tag = labels[results_index]

    for tg in data["intents"]:
        if tg['tag'] == tag:
            responses = tg['responses']

            # Make sure responses list exists before using it
            if responses:
                return random.choice(responses)
            else:
                return "Sorry, I don't understand that."

    # If no matching tag is found, return an error message
    return "Sorry, I don't understand that."

# Initialize an empty list to store chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Streamlit UI
st.title("Chat Bot")

# Text input for user message
input_text = st.text_input("You:")

# Button to send message
if st.button("Send"):
    response = chat(input_text)
    st.session_state.chat_history.append(("You:", input_text))
    st.session_state.chat_history.append(("Bot:", response))
    # Clear input text after sending message
    input_text = ""

# Display chat history
chat_history_container = st.empty()
chat_history_container.text_area("Chat History:", value='\n'.join([f"{sender} {message}" for sender, message in st.session_state.chat_history]), height=200)
