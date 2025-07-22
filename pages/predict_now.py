import streamlit as st
import pandas as pd
from configFiles.makePrediction import get_prediction_all
# from configFiles.dbCode import insert_prediction
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
                # msg = insert_prediction(result_data)
                # st.success(msg)
                # st.dataframe(pd.DataFrame([result_data]), use_container_width=True)
 
if __name__ == "__main__":
    show()