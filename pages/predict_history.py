import streamlit as st
import pandas as pd
from configFiles.config import API_URL
import requests
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

                df = df.fillna("N/A")

                # ✅ Set 'id' as index if it exists
                if "id" in df.columns:
                    df = df.set_index("id")

                st.subheader("✅ Prediction Results")
                st.dataframe(df, use_container_width=True)

            else:
                st.error(f"❌ Failed to fetch data: {response.status_code}")

        except Exception as e:
            st.error(f"❌ Error fetching data: {str(e)}")

show()