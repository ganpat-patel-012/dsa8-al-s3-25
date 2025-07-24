# Fake News Detection Platform

## Overview
This project is a full-stack Fake News Detection platform that leverages deep learning (LSTM, GRU, TextCNN), NLP, and real-time web evidence scraping to classify news statements as true or false. It features a user-friendly Streamlit web app, a FastAPI backend, and a PostgreSQL database for storing predictions and user feedback.

## Features & Techniques

### 🧑‍💻 User Experience
- **Streamlit Frontend:**  
  - Modern, interactive UI for login, registration, prediction, and history.
  - Tabbed prediction interface: “Statement Only” (quick check) and “Full Input” (detailed metadata).
  - Feedback system for users to flag and comment on predictions.
  - All tables display “No Data” for missing/empty values for clarity.

### 🔒 Authentication & Profiles
- **User Authentication:**  
  - Secure registration and login using FastAPI and JWT.
  - Session and cookie management for persistent login.
  - User-specific prediction history and feedback.

### 🧠 Model Ensemble & Prediction
- **Deep Learning Models:**  
  - LSTM, GRU, and TextCNN models trained on the LIAR dataset.
  - Each model provides a probability and binary verdict (True/False).
  - **Ensemble Approach:** Combines model outputs for a robust final prediction.
- **Prediction Storage:**  
  - All predictions, probabilities, and flags are stored in PostgreSQL for traceability.

### 🌐 Web Evidence & Fact-Checking
- **Automated Web Scraping:**  
  - Uses BeautifulSoup and googlesearch-python to gather web content relevant to the input statement.
  - Summarizes scraped content using LLMs (via LangChain + Groq) or fallback heuristics.
  - Evaluates evidence relevance with a BERT classifier.
  - Stores all web evidence, summaries, and relevance scores in the database.

### 🦾 Retrieval-Augmented Generation (RAG) & LLMs
- **RAG Pipeline:**  
  - FAISS vector database built from the LIAR dataset for fast retrieval of similar claims.
  - For each prediction, retrieves relevant examples and metadata to provide context.
  - LLM (via LangChain + Groq) generates two justifications: one from dataset context, one from web evidence.
  - LLM classifies each justification as “True” or “False.”
  - All justifications and LLM verdicts are stored and shown in the UI.

### 🗃️ Database & Backend
- **FastAPI Backend:**  
  - RESTful API for predictions, user management, feedback, and history.
  - Handles all model inference, evidence processing, and RAG logic.
- **PostgreSQL Database:**  
  - Stores users, predictions, feedback, web evidence, and RAG results.
  - SQL DDLs provided for easy setup.

### 🐳 Containerization & Deployment
- **Docker & docker-compose:**  
  - One-command setup for backend, frontend, and database.
  - Ensures reproducibility and easy deployment on any platform.

### 📦 Libraries & Technologies
- **Core:** Streamlit, FastAPI, PostgreSQL, Docker, docker-compose
- **ML/DL:** TensorFlow, PyTorch, Transformers, NLTK, scikit-learn
- **NLP & LLMs:** LangChain, Groq, HuggingFace, BERT
- **Web Scraping:** BeautifulSoup, googlesearch-python
- **Vector DB:** FAISS
- **Other:** pandas, numpy, SQLAlchemy, passlib, python-jose

## Project Structure
```
dsa8-al-s3-25/
├── configFiles/         # Config, API, DB, and ML logic
├── ddl/                # SQL table definitions
├── liar_dataset/       # LIAR dataset (train/valid/test)
├── models/             # Saved ML models
├── pages/              # Streamlit UI pages
├── NBs/                # Jupyter notebooks
├── main.py             # Streamlit entry point
├── requirements.txt    # Python dependencies
├── Dockerfile          # Docker build file
├── docker-compose.yml  # Multi-service orchestration
```

## Setup Instructions

### 1. Clone the Repository
```bash
git clone <repo-url>
cd dsa8-al-s3-25
```

### 2. Install Dependencies
#### Local (Python 3.8+ recommended)
```bash
pip install -r requirements.txt
```

#### Docker (Recommended)
```bash
docker-compose up --build
```
This will start the FastAPI backend, PostgreSQL database, and Streamlit frontend.

### 3. Prepare the Database
- Run the SQL scripts in `ddl/` to create the required tables, or let the backend handle it on first run.

### 4. Download/Place Models
- Place your trained LSTM, GRU, and TextCNN models in the `models/` directory as expected by the backend.

### 5. Run the App
#### Streamlit (Frontend)
```bash
streamlit run main.py
```
#### FastAPI (Backend)
```bash
uvicorn configFiles.fastAPI:app --reload
```

## Usage
- Register or login via the sidebar.
- Enter a news statement and related metadata, or use the "Fill with Random Test Data" button.
- Click "Predict" to get model results and web evidence.
- View model probabilities, ensemble prediction, and web evidence summaries.
- Submit feedback on predictions.

## Dataset
- Uses the [LIAR dataset](https://www.cs.ucsb.edu/~william/data/liar_dataset.zip) for training and testing.
- Place `train.tsv`, `valid.tsv`, and `test.tsv` in the `liar_dataset/` folder.

## API Endpoints
- `/predict/all` : Get predictions from all models (POST)
- `/register` : Register a new user (POST)
- `/login` : User login (POST)
- `/past-predictions` : Get prediction history (GET)

## Environment Variables
- Set database and API URLs in `configFiles/config.py` as needed.

## Contribution
Pull requests and issues are welcome! Please open an issue to discuss major changes.

## Contact
- Project by The Truth Detectives
- For questions, open an issue or contact the maintainer.

## Meet The Truth Detectives
1. **Ganpat Patel:** Dockerization, web scraping, data cleaning, and bug fixes
2. **Thirumurugan Kumar:** Streamlit app and FastAPI (all features)
3. **Vishal Gouli:** RAG and LLM integration

---
*Made with ❤️ and a whole lot of coffee ☕ by The Truth Detectives* 