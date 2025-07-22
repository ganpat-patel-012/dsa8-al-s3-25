import streamlit as st
import pandas as pd
from configFiles.makePrediction import get_prediction_all
from configFiles.dbCode import insert_prediction, insert_feedback
from datetime import datetime
 
def show():
    st.title(" Fake News Detection")
 
    st.subheader("🔍 Make your Prediction")
    col1, col2 = st.columns(2)
    with col1:
        statement = st.text_input("Statement", placeholder="Please write the statement")
        speaker = st.text_input("Speaker", placeholder="Please write the speaker's name")
        location = st.selectbox("Location", ["Unknown","Alabama","Alaska","Arizona","Arkansas","California","Colorado","Colorado ","Connecticut","Delaware","District of Columbia","Florida","Georgia","Illinois","Illinois ","Indiana","Iowa","Kansas","Kentucky","Louisiana","Maryland","Massachusetts","Michigan","Minnesota","Missouri","Nevada","New Hampshire","New Jersey","New Mexico","New York","North Carolina","Ohio","Oregon","Pennsylvania","Rhode Island","South Carolina","South Dakota","Tennessee","Texas","Utah","Vermont","Virginia","Washington","Washington D.C.","Washington, D.C.","Washington, D.C. ","West Virginia","Wisconsin","Wisconsin "])
        context = st.text_input("Context", placeholder="Please write the context")
    with col2:
        subject = st.text_input("Subject", placeholder="Please write the subject")
        speakers_job_title = st.text_input("Speakers Job Title", placeholder="Please write the speaker's job title")
        party = st.selectbox("Party", ["None","Activist","Business-leader","Columnist","Constitution-party","County-commissioner","Democrat","Government-body","Independent","Journalist","Libertarian","Newsmaker","Organization","Republican","State-official","Talk-show-host"])
    
    feedback_state = st.session_state.get('feedback_state', {})
    if 'last_p_id' not in st.session_state:
        st.session_state['last_p_id'] = None
    if 'prediction_made' not in st.session_state:
        st.session_state['prediction_made'] = False

    if st.button("🚀 Predict"):
            if not statement.strip() or not subject.strip() or not speaker.strip() or not speakers_job_title.strip() or not context.strip():
                st.error("❗ Please fill in all required fields before making a prediction.")
            else:
                payload = {
                    "statement": statement, "subject": subject,
                    "speaker": speaker, "speakers_job_title": speakers_job_title,
                    "location": location, "party": party,
                    "context": context
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
                import pandas as pd
                df = pd.DataFrame(predictions)
                st.subheader("Prediction Results")
                st.table(df)


                # result_data = {**payload, "predicted_price": predicted_price, "prediction_source": "WebApp", "prediction_type": "Single"}
                result_data = {"p_statements": statement, "p_subjects": subject, "p_speakers": speaker ,"p_speakers_job_title": speakers_job_title,"p_locations":location,"p_party":party,"p_context":context,"p_probability_lstm":round(probability_lstm, 4),"p_probability_gru":round(probability_gru, 4),"p_probability_textcnn":round(probability_textcnn, 4),"p_ensemble_probability":round(ensemble_prediction, 4),"p_flag_lstm":probability_lstm >= threshold,"p_flag_gru":probability_gru >= threshold,"p_flag_textcnn":probability_textcnn >= threshold,"p_ensemble_flag":ensemble_prediction >= threshold}
                p_id = insert_prediction(result_data)
                if isinstance(p_id, int):
                    st.session_state['last_p_id'] = p_id
                    st.session_state['prediction_made'] = True
                    st.success(f"✅ Prediction saved to database! (ID: {p_id})")
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