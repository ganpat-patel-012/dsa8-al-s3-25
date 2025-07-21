import streamlit as st

st.set_page_config(page_title="The Truth Detectives", layout="wide")

st.markdown("""<h1 style='text-align: center; color: #FF5733;'>🕵️‍♂️ The Truth Detectives 🕵️‍♀️</h1>""", unsafe_allow_html=True)

st.markdown("""<h3 style='text-align: center; color: #4CAF50;'>Unmasking fake news with NLP, deep learning, and real-time fact-checking! 📜🔍</h3>""", unsafe_allow_html=True)

st.markdown("""
Welcome, 🕵️‍♂️ Have you ever wondered if that viral news headline is true or just clickbait? Worry no more! 
We're here to investigate, analyze, and classify news statements using our cutting-edge ensemble of LSTM, GRU, and TextCNN models, powered by the LIAR dataset. With real-time web scraping and LLM verification, we uncover the truth behind the headlines!

Use the sidebar to input a news statement, check past investigations (previous predictions), and reveal the secrets of misinformation!

💡 **Disclaimer:** We take no responsibility if you start questioning every news article you read!
""", unsafe_allow_html=True)

st.markdown("""<h2 style='text-align: center; color: #3498DB;'>Meet The Truth Detectives</h2>""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: grey;'>Made with ❤️ and a whole lot of coffee ☕ by The Truth Detectives</p>", unsafe_allow_html=True)

# run - streamlit run app.py