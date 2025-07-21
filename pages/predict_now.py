import streamlit as st
import pandas as pd
# from configFiles.makePrediction import get_prediction
# from configFiles.dbCode import insert_prediction
from datetime import datetime
 
def show():
    st.title(" Fake News Detection")
 
    st.subheader("🔍 Make your Prediction")
    col1, col2 = st.columns(2)
    with col1:
        statement = st.text_input("Statement", placeholder="Please write the statement")
        speaker = st.text_input("Speaker", placeholder="Please write the speaker's name")
        location = st.selectbox("Location", ["Alabama","Alaska","Arizona","Arkansas","California","Colorado","Colorado ","Connecticut","Delaware","District of Columbia","Florida","Georgia","Illinois","Illinois ","Indiana","Iowa","Kansas","Kentucky","Louisiana","Maryland","Massachusetts","Michigan","Minnesota","Missouri","Nevada","New Hampshire","New Jersey","New Mexico","New York","North Carolina","Ohio","Oregon","Pennsylvania","Rhode Island","South Carolina","South Dakota","Tennessee","Texas","Unknown","Utah","Vermont","Virginia","Washington","Washington D.C.","Washington, D.C.","Washington, D.C. ","West Virginia","Wisconsin","Wisconsin "])
        context = st.text_input("Context", placeholder="Please write the context")
    with col2:
        subject = st.text_input("Subject", placeholder="Please write the subject")
        speakers_job_title = st.text_input("Speakers Job Title", placeholder="Please write the speaker's job title")
        party = st.selectbox("Party", ["Activist","Business-leader","Columnist","Constitution-party","County-commissioner","Democrat","Government-body","Independent","Journalist","Libertarian","Newsmaker","None","Organization","Republican","State-official","Talk-show-host"])
    if st.button("🚀 Predict"):
            payload = {
                "statement": statement, "subject": subject,
                "speaker": speaker, "speakers_job_title": speakers_job_title,
                "location": location, "party": party,
                "context": context
            }
            # predicted_price = get_prediction(payload)
            # st.write("Predicted Price (₹)", predicted_price)
            # result_data = {**payload, "predicted_price": predicted_price, "prediction_source": "WebApp", "prediction_type": "Single"}
            # msg = insert_prediction(result_data)
            # st.success(msg)
            # st.dataframe(pd.DataFrame([result_data]), use_container_width=True)
 
if __name__ == "__main__":
    show()