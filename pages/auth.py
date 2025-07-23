import streamlit as st
import requests
from configFiles.config import API_URL
import re

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
        if st.button("Logout"):
            for key in ["access_token", "username", "authenticated"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.success("Logged out successfully.")
            st.experimental_rerun()
    else:
        st.title("User Authentication")
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
                    else:
                        detail = resp.json().get('detail', resp.text)
                        if "already exists" in detail.lower():
                            st.error(f"Registration failed: {detail}")
                            if st.button("Go to Login"):
                                st.experimental_set_query_params(tab="Login")
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
                        st.success("Login successful! Please navigate to the main page or prediction page.")
                        st.info("You can now close this tab and open the main page or prediction page.")
                        st.experimental_rerun()
                    else:
                        st.error(f"Login failed: {resp.json().get('detail', resp.text)}")

# Ensure the UI is shown when this file is run directly by Streamlit
show() 