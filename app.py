import streamlit as st
import random
import json
import pickle
import numpy as np
import nltk
from keras.models import load_model
from nltk.stem import WordNetLemmatizer

# Load your chatbot model and other data
lemmatizer = WordNetLemmatizer()
intents = json.loads(open("intents.json").read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbotmodel.h5')

# Function to clean up sentences
def clean_up_sentences(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

# Function to get the bag of words
def bagw(sentence):
    sentence_words = clean_up_sentences(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

# Function to predict the intent
def predict_class(sentence):
    bow = bagw(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

# Function to get a response
def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    result = ""
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

def apply_custom_styles():
    st.markdown("""
        <style>
        .bot-response {
            background-color: #e0f7fa;  /* Light blue background */
            padding: 10px 20px;        /* Padding around text */
            border-radius: 15px;       /* Rounded corners for bubble appearance */
            display: inline-block;     /* Necessary for padding to work */
        }
        </style>
    """, unsafe_allow_html=True)


apply_custom_styles()  # Apply the custom styles


# Streamlit app with design enhancements
# ==========================================
# TITLE
# ==========================================
st.title("Chatbot Application 🤖")

# ==========================================
# SUBTITLE WITH DESIGN ELEMENTS
# ==========================================
st.markdown(
    """
    Welcome to the Chatbot Application.
    ---
    """
)

# ==========================================
# INITIALIZE CHAT HISTORY
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = []

# ==========================================
# REACT TO USER INPUT
# ==========================================
if user_input := st.chat_input("You: "):

    # Predict the intent of the user's message
    ints = predict_class(user_input)
    response1 = get_response(ints, intents)

    # ------------------------------------------
    # DISPLAY USER MESSAGE IN CHAT MESSAGE CONTAINER
    # ------------------------------------------
    st.write("---")
    st.session_state.messages.append({"role": "user", "content": user_input})

    # ------------------------------------------
    # PROCESS THE BOT'S RESPONSE
    # ------------------------------------------
    response = f"Bot: {response1}"  

    # ------------------------------------------
    # DISPLAY CHAT MESSAGES FROM HISTORY ON APP RERUN
    # ------------------------------------------
    for message in st.session_state.messages:
       with st.empty():
            if message["role"] == "user":
                st.markdown(f"**You🧑‍💻**: {message['content']}")
            elif message["role"] == "Bot":
                st.markdown(f"**🤖 Bot**: {message['content']}")

    # ------------------------------------------
    # DISPLAY BOT RESPONSE IN CHAT MESSAGE CONTAINER
    # ------------------------------------------
    with st.empty():
        st.markdown(response)

    # Add the bot's response to the chat history
    st.session_state.messages.append({"role": "Bot", "content": response1})

# ==========================================
# FOOTER
# ==========================================
st.write("---")
st.markdown("Thank you for using our Chatbot! 🙌")




