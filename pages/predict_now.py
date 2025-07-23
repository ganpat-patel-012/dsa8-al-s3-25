import streamlit as st
import pandas as pd
from configFiles.makePrediction import get_prediction_all
from configFiles.dbCode import insert_prediction, insert_feedback
from datetime import datetime
import psycopg2
from configFiles.config import DB_CONFIG
from configFiles.web_data_scrap import scrape_web_evidence
import time
import random
from streamlit_cookies_manager import EncryptedCookieManager

cookies = EncryptedCookieManager(
    prefix="dsa8_",
    password="a-very-secret-password"
)
if not cookies.ready():
    st.stop()

# Restore session state from cookies if available
if cookies.get("authenticated") == "True":
    st.session_state["access_token"] = cookies.get("access_token")
    st.session_state["username"] = cookies.get("username")
    st.session_state["authenticated"] = True
    if cookies.get("user_id"):
        st.session_state["user_id"] = int(cookies.get("user_id"))
 
def show():
    # Sidebar login/register button if not authenticated
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

    if "authenticated" not in st.session_state or not st.session_state["authenticated"]:
        st.warning("You must be logged in to access this page.")
        return

    st.title(" Fake News Detection")

    # --- Random Test Data Button ---
    if st.button("🎲 Fill with Random Test Data"):
        test_df = pd.read_csv("liar_dataset/test.tsv", sep="\t", header=None)
        # Map columns based on observed structure
        # 0: id, 1: label, 2: statement, 3: subject, 4: speaker, 5: speakers_job_title, 6: location, 7: party, 13: context
        row = test_df.sample(1).iloc[0]
        st.session_state["statement"] = row[2]
        st.session_state["subject"] = row[3]
        st.session_state["speaker"] = row[4]
        st.session_state["speakers_job_title"] = row[5] if pd.notnull(row[5]) else ""
        st.session_state["location"] = row[6] if pd.notnull(row[6]) else "Unknown"
        st.session_state["party"] = row[7] if pd.notnull(row[7]) else "None"
        st.session_state["context"] = row[13] if pd.notnull(row[13]) else ""
        st.session_state["_random_label"] = row[1]  # Save the label to session state
        st.rerun()

    # Show the target flag value if a random row was loaded
    if "_random_label" in st.session_state:
        label = str(st.session_state["_random_label"]).strip().lower()
        if label in ["true", "mostly-true"]:
            display_label = "True"
        else:
            display_label = "False"
        st.info(f"Target label for this random sample: {display_label}")

    st.subheader("🔍 Make your Prediction")
    col1, col2 = st.columns(2)
    with col1:
        statement = st.text_input("Statement", placeholder="Please write the statement", key="statement")
        speaker = st.text_input("Speaker", placeholder="Please write the speaker's name", key="speaker")
        location = st.selectbox("Location", ["Unknown","Alabama","Alaska","Arizona","Arkansas","California","Colorado","Colorado ","Connecticut","Delaware","District of Columbia","Florida","Georgia","Illinois","Illinois ","Indiana","Iowa","Kansas","Kentucky","Louisiana","Maryland","Massachusetts","Michigan","Minnesota","Missouri","Nevada","New Hampshire","New Jersey","New Mexico","New York","North Carolina","Ohio","Oregon","Pennsylvania","Rhode Island","South Carolina","South Dakota","Tennessee","Texas","Utah","Vermont","Virginia","Washington","Washington D.C.","Washington, D.C.","Washington, D.C. ","West Virginia","Wisconsin","Wisconsin "], key="location")
        context = st.text_input("Context", placeholder="Please write the context", key="context")
    with col2:
        subject = st.text_input("Subject", placeholder="Please write the subject", key="subject")
        speakers_job_title = st.text_input("Speakers Job Title", placeholder="Please write the speaker's job title", key="speakers_job_title")
        party = st.selectbox("Party", ["None","Activist","Business-leader","Columnist","Constitution-party","County-commissioner","Democrat","Government-body","Independent","Journalist","Libertarian","Newsmaker","Organization","Republican","State-official","Talk-show-host"], key="party")
    
    feedback_state = st.session_state.get('feedback_state', {})
    if 'last_p_id' not in st.session_state:
        st.session_state['last_p_id'] = None
    if 'prediction_made' not in st.session_state:
        st.session_state['prediction_made'] = False

    if st.button("🚀 Predict"):
        # Ensure all fields are non-empty strings
        statement_val = statement if statement else ""
        subject_val = subject if subject else ""
        speaker_val = speaker if speaker else ""
        speakers_job_title_val = speakers_job_title if speakers_job_title else ""
        location_val = location if location else ""
        party_val = party if party else ""
        context_val = context if context else ""

        payload = {
            "statement": statement_val, "subject": subject_val,
            "speaker": speaker_val, "speakers_job_title": speakers_job_title_val,
            "location": location_val, "party": party_val,
            "context": context_val
        }

        result = get_prediction_all(payload)

        if isinstance(result, dict):
            # Extract probabilities
            probability_lstm = result.get("probability_lstm", 0.0)
            probability_gru = result.get("probability_gru", 0.0)
            probability_textcnn = result.get("probability_textcnn", 0.0)

        threshold = 0.6

        # Show raw probabilities
        st.subheader("Model Probabilities")
        st.write("LSTM:", probability_lstm)
        st.write("GRU:", probability_gru)
        st.write("TEXTCNN:", probability_textcnn)

        # Calculate ensemble
        ensemble_prediction = (probability_lstm + probability_gru + probability_textcnn) / 3
        st.write("Ensemble Prediction Probability:", ensemble_prediction)

        # Prepare prediction flags
        predictions = {
            "Model": ["LSTM", "GRU", "TEXTCNN", "Ensemble"],
            "Probability": [
                round(probability_lstm, 4),
                round(probability_gru, 4),
                round(probability_textcnn, 4),
                round(ensemble_prediction, 4)
            ],
            "Prediction (>= 0.6)": [
                probability_lstm >= threshold,
                probability_gru >= threshold,
                probability_textcnn >= threshold,
                ensemble_prediction >= threshold
            ]
        }

        # Show in a table
        df = pd.DataFrame(predictions)
        st.subheader("Prediction Results")
        st.table(df)

        # --- Web Evidence Section ---
        with st.spinner("Scraping web evidence for the statement... It can take up to 5 minutes... Please wait!"):
            evidence_df = scrape_web_evidence(statement)
        st.subheader("Web Evidence Results")
        if not evidence_df.empty:
            st.dataframe(evidence_df)
            # Only consider relevance_score where evidence_summary is valid
            if 'relevance_score' in evidence_df.columns:
                valid_mask = (
                    evidence_df['evidence_summary'].notna() &
                    (evidence_df['evidence_summary'] != "No relevant evidence summary found")
                )
                valid_scores = evidence_df.loc[valid_mask, 'relevance_score'].dropna()
                if not valid_scores.empty:
                    probability_web = valid_scores.mean()
                    st.write(f"Web Evidence Relevance Score (avg): {probability_web:.4f}")
                    st.write(f"Web Data flag: {probability_web >= threshold}")
                else:
                    st.info("No relevant web evidence found for this statement.")
        else:
            st.info("No web evidence found for this statement.")


        # result_data = {**payload, "predicted_price": predicted_price, "prediction_source": "WebApp", "prediction_type": "Single"}
        result_data = {"p_statements": statement, "p_subjects": subject, "p_speakers": speaker ,"p_speakers_job_title": speakers_job_title,"p_locations":location,"p_party":party,"p_context":context,"p_probability_lstm":round(probability_lstm, 4),"p_probability_gru":round(probability_gru, 4),"p_probability_textcnn":round(probability_textcnn, 4),"p_ensemble_probability":round(ensemble_prediction, 4),"p_flag_lstm":probability_lstm >= threshold,"p_flag_gru":probability_gru >= threshold,"p_flag_textcnn":probability_textcnn >= threshold,"p_ensemble_flag":ensemble_prediction >= threshold}
        # Add user id to result_data
        user_id = st.session_state.get('user_id')
        if not user_id:
            # Optionally, fetch user_id from username if not present
            conn = psycopg2.connect(**DB_CONFIG)
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM users WHERE username=%s", (st.session_state['username'],))
            row = cursor.fetchone()
            user_id = row[0] if row else None
            conn.close()
            st.session_state['user_id'] = user_id
        result_data["p_user_id"] = user_id
        p_id = insert_prediction(result_data)
        if isinstance(p_id, int):
            st.session_state['last_p_id'] = p_id
            st.session_state['prediction_made'] = True
            st.success(f"✅ Prediction saved to database! (ID: {p_id})")
            # Save web evidence to web_data_scrap
            if 'evidence_df' in locals() and not evidence_df.empty:
                from configFiles.dbCode import insert_web_data_scrap
                web_data_rows = []
                for _, row in evidence_df.iterrows():
                    web_data_rows.append({
                        'p_id': p_id,
                        'statement': row.get('statement', ''),
                        'url': row.get('url', ''),
                        'scraped_content': row.get('scraped_content', ''),
                        'evidence_summary': row.get('evidence_summary', ''),
                        'relevance_score': row.get('relevance_score', None)
                    })
                msg = insert_web_data_scrap(web_data_rows)
                st.info(msg)
        else:
            st.session_state['last_p_id'] = None
            st.session_state['prediction_made'] = False
            st.error(p_id)


        
    # Feedback form after prediction
    if st.session_state.get('prediction_made') and st.session_state.get('last_p_id'):
        st.subheader("📝 Submit Feedback for this Prediction")
        feedback_flag = st.selectbox("Feedback Flag", ["True", "False"], key="feedback_flag")
        feedback_comment = st.text_area("Comment", key="feedback_comment")
        if st.button("Submit Feedback"):
            f_p_id = st.session_state['last_p_id']
            f_statemnt = statement
            f_flag = feedback_flag
            f_comment = feedback_comment
            feedback_msg = insert_feedback(f_p_id, f_statemnt, f_flag, f_comment)
            st.success(feedback_msg)
            st.session_state['prediction_made'] = False
            st.session_state['last_p_id'] = None
 
if __name__ == "__main__":
    show()