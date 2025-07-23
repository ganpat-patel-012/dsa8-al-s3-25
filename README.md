# Fake News Detection Platform

## Overview
This project is a full-stack Fake News Detection platform that leverages deep learning (LSTM, GRU, TextCNN), NLP, and real-time web evidence scraping to classify news statements as true or false. It features a user-friendly Streamlit web app, a FastAPI backend, and a PostgreSQL database for storing predictions and user feedback.

## Features
- **User Authentication:** Register, login, and manage your profile.
- **Prediction Interface:** Input news statements and receive predictions from an ensemble of LSTM, GRU, and TextCNN models.
- **Web Evidence Scraping:** Scrape and summarize web content to support or refute statements.
- **Prediction History:** View and provide feedback on past predictions.
- **Feedback System:** Submit feedback on model predictions to improve the system.
- **Random Test Data:** Autofill the prediction form with random samples from the LIAR dataset for quick testing.

## Tech Stack
- **Frontend:** Streamlit
- **Backend:** FastAPI
- **Database:** PostgreSQL
- **ML/DL:** TensorFlow, PyTorch, Transformers, NLTK
- **Web Scraping:** BeautifulSoup, googlesearch-python
- **Containerization:** Docker, docker-compose

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

---
*Made with ❤️ and a whole lot of coffee ☕ by The Truth Detectives* 