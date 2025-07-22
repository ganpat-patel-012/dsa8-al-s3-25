import streamlit as st
import pandas as pd
from configFiles.config import API_URL
import requests

def show():
    st.title("📊 Past Predictions")

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("📅 Start Date")
    with col2:
        end_date = st.date_input("📅 End Date")


    if st.button("🔄 Fetch Predictions"):
        try:
            response = requests.get(f"{API_URL}/past-predictions", params={
                "start_date": start_date,
                "end_date": end_date,
            })

            if response.status_code == 200:
                data = response.json()

                # ✅ Ensure the response is a list of dicts
                if not isinstance(data, list) or len(data) == 0:
                    st.warning("⚠️ No prediction history found for the selected filters.")
                    return

                df = pd.DataFrame(data)

                # ✅ Set 'id' as index if it exists
                if "id" in df.columns:
                    df = df.set_index("id")

                st.subheader("✅ Prediction Results")

                # Simulate modal state
                if 'show_modal' not in st.session_state:
                    st.session_state['show_modal'] = False
                if 'modal_feedbacks' not in st.session_state:
                    st.session_state['modal_feedbacks'] = []
                if 'modal_pred_id' not in st.session_state:
                    st.session_state['modal_pred_id'] = None

                # Display table with View Feedback buttons
                for idx, row in df.iterrows():
                    st.write(f"---\n**Prediction ID:** {row.get('p_id', idx)}")
                    st.write(f"**Statement:** {row.get('p_statements', '')}")
                    st.write(f"**Speaker:** {row.get('p_speakers', '')}")
                    st.write(f"**Prediction Time:** {row.get('prediction_time', '')}")
                    feedbacks = row.get('feedbacks', [])
                    # Button to view feedback
                    if st.button(f"View Feedback for Prediction {row.get('p_id', idx)}", key=f"viewfb_{row.get('p_id', idx)}"):
                        st.session_state['show_modal'] = True
                        st.session_state['modal_feedbacks'] = feedbacks
                        st.session_state['modal_pred_id'] = row.get('p_id', idx)

                # Simulated modal popup
                if st.session_state.get('show_modal', False):
                    st.markdown("""
                        <style>
                        .modal-bg {
                            position: fixed;
                            top: 0; left: 0; width: 100vw; height: 100vh;
                            background: rgba(0,0,0,0.5);
                            z-index: 9999;
                        }
                        .modal-content {
                            position: fixed;
                            top: 50%; left: 50%; transform: translate(-50%, -50%);
                            background: white; padding: 2em; border-radius: 10px;
                            z-index: 10000; min-width: 300px; max-width: 90vw;
                        }
                        </style>
                        <div class='modal-bg'></div>
                        <div class='modal-content'>
                    """, unsafe_allow_html=True)
                    st.markdown(f"### Feedback for Prediction ID: {st.session_state['modal_pred_id']}")
                    feedbacks = st.session_state.get('modal_feedbacks', [])
                    import ast
                    if isinstance(feedbacks, str):
                        try:
                            feedbacks = ast.literal_eval(feedbacks)
                        except Exception:
                            feedbacks = []
                    if feedbacks and isinstance(feedbacks, list) and len(feedbacks) > 0:
                        for fb in feedbacks:
                            st.write(f"- **Flag:** {fb.get('f_flag', '')} | **Comment:** {fb.get('f_comment', '')}")
                    else:
                        st.write("No feedback for this prediction.")
                    if st.button("Close", key="close_modal"):
                        st.session_state['show_modal'] = False
                        st.session_state['modal_feedbacks'] = []
                        st.session_state['modal_pred_id'] = None
                    st.markdown("</div>", unsafe_allow_html=True)

            else:
                st.error(f"❌ Failed to fetch data: {response.status_code}")

        except Exception as e:
            st.error(f"❌ Error fetching data: {str(e)}")

show()
