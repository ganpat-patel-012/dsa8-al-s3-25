import streamlit as st
import requests
from configFiles.config import API_URL
import importlib

st.set_page_config(page_title="Fake News Detection", layout="wide")

# --- Sidebar: Show login/register if not authenticated ---
if not st.session_state.get("authenticated", False):
    if st.sidebar.button("Login/Register"):
        try:
            st.switch_page("pages/profile.py")
        except Exception:
            st.experimental_set_query_params(page="profile")

# --- Sidebar: Only show logout if authenticated ---
if st.session_state.get("authenticated", False):
    st.sidebar.write(f"Logged in as: {st.session_state['username']}")
    if st.sidebar.button("Logout"):
        st.session_state.pop("access_token", None)
        st.session_state.pop("username", None)
        st.session_state["authenticated"] = False
        st.rerun()

# --- Always show Home page content ---
st.markdown("""<h1 style='text-align: center; color: #FF5733;'>📰 Fake News Detection 📰</h1>""", unsafe_allow_html=True)
st.markdown("""<h3 style='text-align: center; color: #4CAF50;'>Unmasking fake news with NLP, deep learning, and real-time fact-checking! 📜🔍</h3>""", unsafe_allow_html=True)

st.markdown("""
Welcome, 🕵️‍♂️!  
Ever wondered if that viral news headline is true or just clickbait?  
**This platform helps you investigate, analyze, and classify news statements using an ensemble of LSTM, GRU, and TextCNN models, powered by the LIAR dataset.**  
With real-time web scraping and LLM-based justification, we uncover the truth behind the headlines!

---

### 🔑 **Features**
- **Easy Login/Register:** Use the sidebar to access your account.
- **Smart Predictions:** Input a news statement and get verdicts from multiple deep learning models.
- **Fact-Checking:** Real-time web scraping and LLMs provide justifications for each prediction.
- **Transparent History:**  
    - View your past predictions and their details.
    - **Missing or empty data is always shown as _No Data_ for clarity.**
    - **Multiple justifications per prediction are clearly displayed, with expandable sections for details.**

---

### 🚦 **How It Works**
1. **Data Input:** Paste or type a news statement.
2. **Model Ensemble:** LSTM, GRU, and TextCNN each provide a verdict.
3. **Fact‑Check:** Real-time web scraping and LLMs generate justifications.
4. **Results:** See the verdict (True/False) with confidence scores and detailed justifications.

---

### 👨‍💻 **Meet The Truth Detectives**
1. **Ganpat Patel:** Dockerization, web scraping, data cleaning, and bug fixes
2. **Thirumurugan Kumar:** Streamlit app and FastAPI (all features)
3. **Vishal Gouli:** RAG and LLM integration
            
---

<p style='text-align: center; color: grey;'>Made with ❤️ and a whole lot of coffee ☕ by The Truth Detectives</p>
            

            
""", unsafe_allow_html=True)