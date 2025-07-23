import streamlit as st
import requests
from configFiles.config import API_URL
import re
import time
from streamlit_cookies_manager import EncryptedCookieManager

# Initialize cookies manager
cookies = EncryptedCookieManager(
    prefix="dsa8_",  # Change as needed
    password="a-very-secret-password"  # Use a strong password in production
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


def register_user(username, password):
    response = requests.post(f"{API_URL}/register", json={"username": username, "password": password})
    return response

def login_user(username, password):
    response = requests.post(f"{API_URL}/login", json={"username": username, "password": password})
    return response

def is_valid_email(email):
    # Simple regex for email validation
    return re.match(r"^[\w\.-]+@[\w\.-]+\.\w+$", email) is not None

def is_strong_password(password):
    # At least 8 chars, 1 uppercase, 1 lowercase, 1 special char, 1 number
    if len(password) < 8:
        return False
    if not re.search(r"[A-Z]", password):
        return False
    if not re.search(r"[a-z]", password):
        return False
    if not re.search(r"[0-9]", password):
        return False
    if not re.search(r"[^A-Za-z0-9]", password):
        return False
    return True

def show():
    if st.session_state.get("authenticated"):
        st.title("Profile")
        st.write(f"**Username:** {st.session_state.get('username', 'Unknown')}")
        # --- Show all predictions made by the user ---
        import psycopg2
        import pandas as pd
        from configFiles.config import DB_CONFIG
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
            if user_id:
                cookies["user_id"] = str(user_id)
                cookies.save()
        if user_id:
            try:
                conn = psycopg2.connect(**DB_CONFIG)
                query = """
                    SELECT p_id, p_statements, p_subjects, p_speakers, p_speakers_job_title, p_locations, p_party, p_context, p_probability_lstm, p_probability_gru, p_probability_textcnn, p_ensemble_probability, p_flag_lstm, p_flag_gru, p_flag_textcnn, p_ensemble_flag, prediction_time
                    FROM predictions
                    WHERE p_user_id = %s
                    ORDER BY prediction_time DESC
                """
                df = pd.read_sql_query(query, conn, params=(user_id,))
                conn.close()
                if not df.empty:
                    st.subheader("Your Predictions History")
                    st.dataframe(df, use_container_width=True)
                else:
                    st.info("You have not made any predictions yet.")
            except Exception as e:
                st.error(f"Error fetching your predictions: {e}")
        if st.button("Logout"):
            for key in ["access_token", "username", "authenticated", "user_id"]:
                if key in st.session_state:
                    del st.session_state[key]
            # Clear cookies on logout
            for key in ["access_token", "username", "authenticated", "user_id"]:
                cookies[key] = ""
            cookies.save()
            st.success("Logged out successfully.")
            st.rerun()
    else:
        st.title("Profile")
        tab_login, tab_register = st.tabs(["Login", "Register"])

        with tab_register:
            st.header("Register")
            reg_username = st.text_input("Username (Email)", key="reg_username")
            reg_password = st.text_input("Password", type="password", key="reg_password")
            reg_submit = st.button("Register", key="reg_submit")
            if reg_submit:
                if not reg_username or not reg_password:
                    st.error("Please enter username and password.")
                elif not is_valid_email(reg_username):
                    st.error("Please enter a valid email address as username.")
                elif not is_strong_password(reg_password):
                    st.error("Password must be at least 8 characters long and include 1 uppercase, 1 lowercase, 1 number, and 1 special character.")
                else:
                    resp = register_user(reg_username, reg_password)
                    if resp.status_code == 200:
                        st.success("Registration successful! Please login.")
                        import time
                        time.sleep(2)
                        st.rerun()
                    else:
                        detail = resp.json().get('detail', resp.text)
                        if "already exists" in detail.lower():
                            st.error(f"Registration failed: {detail}")
                            if st.button("Go to Login"):
                                st.rerun()
                        else:
                            st.error(f"Registration failed: {detail}")

        with tab_login:
            st.header("Login")
            login_username = st.text_input("Username (Email)", key="login_username")
            login_password = st.text_input("Password", type="password", key="login_password")
            login_submit = st.button("Login", key="login_submit")
            if login_submit:
                if not login_username or not login_password:
                    st.error("Please enter username and password.")
                else:
                    resp = login_user(login_username, login_password)
                    if resp.status_code == 200:
                        token = resp.json()["access_token"]
                        st.session_state["access_token"] = token
                        st.session_state["username"] = login_username
                        st.session_state["authenticated"] = True
                        # Fetch user_id
                        import psycopg2
                        from configFiles.config import DB_CONFIG
                        conn = psycopg2.connect(**DB_CONFIG)
                        cursor = conn.cursor()
                        cursor.execute("SELECT id FROM users WHERE username=%s", (login_username,))
                        row = cursor.fetchone()
                        user_id = row[0] if row else None
                        conn.close()
                        st.session_state["user_id"] = user_id
                        # Save to cookies
                        cookies["access_token"] = token
                        cookies["username"] = login_username
                        cookies["authenticated"] = "True"
                        cookies["user_id"] = str(user_id) if user_id else ""
                        cookies.save()
                        st.success("Login successful! Please navigate to the main page or prediction page.")
                        st.info("You can now close this tab and open the main page or prediction page.")
                        st.rerun()
                    else:
                        st.error(f"Login failed: {resp.json().get('detail', resp.text)}")

# Ensure the UI is shown when this file is run directly by Streamlit
show() 