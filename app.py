# ================= IMPORTS =================

import streamlit as st
import pandas as pd
import joblib
import hashlib
import sqlite3
import os

from datetime import datetime


# ================= CONFIG =================

st.set_page_config(
    page_title="Insurance AI Platform",
    page_icon="💼",
    layout="centered"
)


# ================= MOBILE STYLE =================

st.markdown("""
<style>
.block-container {
    padding: 1rem;
    max-width: 100%;
}

button {
    width: 100%;
    border-radius: 8px;
}

input, select {
    font-size: 16px !important;
}
</style>
""", unsafe_allow_html=True)


# ================= PATH =================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ================= USER DATABASE =================

conn_user = sqlite3.connect("users.db", check_same_thread=False)
cur_user = conn_user.cursor()

cur_user.execute("""
CREATE TABLE IF NOT EXISTS users(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE,
    password TEXT
)
""")

conn_user.commit()


# ================= CLAIM DATABASE =================

conn = sqlite3.connect("memory.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS claims_history(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    age INTEGER,
    policy INTEGER,
    amount REAL,
    days INTEGER,
    pre INTEGER,
    decision INTEGER,
    created TEXT,
    username TEXT
)
""")

conn.commit()


# ================= LOAD MODELS =================

@st.cache_resource
def load_models():

    cost = joblib.load(os.path.join(BASE_DIR,"cost_model.joblib"))
    claim = joblib.load(os.path.join(BASE_DIR,"claim_model.joblib"))
    fraud = joblib.load(os.path.join(BASE_DIR,"fraud_model.joblib"))

    return cost, claim, fraud


cost_model, claim_model, fraud_model = load_models()


# ================= HELPERS =================

def hash_pass(p):
    return hashlib.sha256(p.encode()).hexdigest()


def check_user(u, p):

    cur_user.execute(
        "SELECT password FROM users WHERE username=?",
        (u,)
    )

    row = cur_user.fetchone()

    if row and row[0] == hash_pass(p):
        return True

    return False


def create_user(u, p):

    try:

        cur_user.execute(
            "INSERT INTO users VALUES(NULL,?,?)",
            (u, hash_pass(p))
        )

        conn_user.commit()
        return True

    except:
        return False


def detect_fraud(df):

    return fraud_model.predict_proba(df)[0][1]


import requests


HF_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"


def ask_ai(prompt):

    api_key = st.secrets["HF_API_KEY"]

    headers = {
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 200,
            "temperature": 0.7
        }
    }

    try:

        res = requests.post(
            f"https://api-inference.huggingface.co/models/{HF_MODEL}",
            headers=headers,
            json=payload,
            timeout=60
        )

        data = res.json()

        if isinstance(data, list):
            return data[0]["generated_text"]

        return "AI response unavailable."

    except Exception as e:

        return "AI Error: " + str(e)



# ================= SESSION =================

if "login" not in st.session_state:
    st.session_state.login = False

if "user" not in st.session_state:
    st.session_state.user = ""


# ================= AUTH UI =================

def login_ui():

    st.subheader("🔐 Login")

    u = st.text_input("Username")
    p = st.text_input("Password", type="password")

    if st.button("Login"):

        if check_user(u, p):

            st.session_state.login = True
            st.session_state.user = u

            st.success("Logged in")
            st.rerun()

        else:
            st.error("Invalid username or password")


def signup_ui():

    st.subheader("📝 Signup")

    u = st.text_input("New Username")
    p = st.text_input("New Password", type="password")

    if st.button("Register"):

        if len(u) < 3 or len(p) < 4:

            st.warning("Username/password too short")
            return

        if create_user(u, p):

            st.success("Account created. Login now.")

        else:

            st.error("Username already exists")


# ================= LOGIN PAGE =================

if not st.session_state.login:

    t1, t2 = st.tabs(["Login", "Signup"])

    with t1:
        login_ui()

    with t2:
        signup_ui()

    st.stop()


# ================= SIDEBAR =================

menu = st.sidebar.radio(
    "Menu",
    ["Cost", "Claim", "History", "Profile", "Assistant", "Logout"]
)


st.sidebar.success(f"User: {st.session_state.user}")


# ================= COST =================

if menu == "Cost":

    st.title("💰 Insurance Cost Predictor")

    with st.form("cost"):

        age = st.number_input("Age",18,100,25)
        bmi = st.number_input("BMI",10.0,50.0,22.0)
        children = st.number_input("Children",0,5,0)

        gender = st.selectbox("Gender",["male","female"])
        smoker = st.selectbox("Smoker",["yes","no"])
        region = st.selectbox(
            "Region",
            ["southeast","southwest","northeast","northwest"]
        )

        submit = st.form_submit_button("Predict")


    if submit:

        df = pd.DataFrame([[

            age,bmi,children,

            1 if gender=="male" else 0,
            1 if smoker=="yes" else 0,

            1 if region=="northwest" else 0,
            1 if region=="southeast" else 0,
            1 if region=="southwest" else 0

        ]])


        cost = cost_model.predict(df)[0]

        st.success(f"Estimated Cost: ₹ {cost:,.0f}")


# ================= CLAIM =================

elif menu == "Claim":

    st.title("🏥 Claim Analysis")

    with st.form("claim"):

        age = st.number_input("Age",18,100,25)
        policy = st.selectbox("Policy Type",[1,2,3])
        amount = st.number_input("Amount",0.0,step=1000.0)
        days = st.number_input("Hospital Days",0)
        pre = st.selectbox("Pre-existing",[0,1])

        submit = st.form_submit_button("Analyze")


    if submit:

        df = pd.DataFrame([[

            age,policy,amount,days,pre

        ]])


        decision = claim_model.predict(df)[0]

        fraud = detect_fraud(df)


        st.success("Approved" if decision else "Rejected")

        st.progress(fraud)

        st.write(f"Fraud Risk: {fraud*100:.1f}%")


        cursor.execute("""
        INSERT INTO claims_history
        VALUES(NULL,?,?,?,?,?,?,?,?)
        """,(

            age,policy,amount,days,pre,
            decision,
            datetime.now().strftime("%Y-%m-%d %H:%M"),
            st.session_state.user

        ))

        conn.commit()


# ================= HISTORY =================

elif menu == "History":

    st.title("📜 My History")

    rows = cursor.execute("""
        SELECT * FROM claims_history
        WHERE username=?
        ORDER BY id DESC
    """,(st.session_state.user,)).fetchall()


    if not rows:

        st.info("No history yet")

    else:

        df = pd.DataFrame(rows,columns=[
            "ID","Age","Policy","Amount",
            "Days","Pre","Decision","Date","User"
        ])

        st.dataframe(df,use_container_width=True)


# ================= PROFILE =================

elif menu == "Profile":

    st.title("👤 Dashboard")

    rows = cursor.execute("""
        SELECT decision, amount
        FROM claims_history
        WHERE username=?
    """,(st.session_state.user,)).fetchall()


    if not rows:

        st.info("No data yet")
        st.stop()


    df = pd.DataFrame(rows,columns=["Decision","Amount"])


    total = len(df)

    df["Decision"] = df["Decision"].astype(int)

    approved = df["Decision"].sum()
    rejected = total - approved


    avg = df["Amount"].mean()


    c1,c2,c3,c4 = st.columns(4)

    c1.metric("Total",total)
    c2.metric("Approved",approved)
    c3.metric("Rejected",rejected)
    c4.metric("Average",f"₹{avg:,.0f}")


    st.bar_chart(df["Decision"].value_counts())


# ================= ASSISTANT =================

elif menu == "Assistant":

    st.title("🤖 AI Assistant")

    q = st.text_area("Ask Question")

    if st.button("Send") and q:

        st.write(ask_ai(q))


# ================= LOGOUT =================

elif menu == "Logout":

    st.session_state.login = False
    st.session_state.user = ""

    st.success("Logged out")

    st.rerun()
