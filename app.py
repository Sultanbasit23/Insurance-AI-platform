# ================= IMPORTS =================

import streamlit as st
import pandas as pd
import joblib
import hashlib
import requests
import sqlite3
import shap
import matplotlib.pyplot as plt

from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import declarative_base, sessionmaker

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.colors import black, grey


# ================= MOBILE STYLE =================

st.markdown("""
<style>
.block-container {
    padding: 1rem;
    max-width: 100%;
}

button {
    width: 100%;
    border-radius: 10px;
}

input, select, textarea {
    font-size: 16px !important;
}
</style>
""", unsafe_allow_html=True)


# ================= CONFIG =================

st.set_page_config(
    page_title="Insurance AI Platform",
    page_icon="💼",
    layout="centered"
)


# ================= DATABASE (USERS) =================

engine = create_engine("sqlite:///users.db", echo=False)
Base = declarative_base()
Session = sessionmaker(bind=engine)
db = Session()


class User(Base):

    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True)
    password = Column(String(200))


Base.metadata.create_all(engine)


# ================= CLAIM DB =================

conn = sqlite3.connect("memory.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS claims_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    age INTEGER,
    policy_type INTEGER,
    claim_amount REAL,
    hospital_days INTEGER,
    pre_existing INTEGER,
    decision INTEGER,
    ai_response TEXT,
    created_at TEXT,
    username TEXT
)
""")

conn.commit()


# ================= MODELS =================

@st.cache_resource
def load_models():
    cost = joblib.load("cost_model.joblib")
    claim = joblib.load("claim_model.joblib")
    fraud = joblib.load("fraud_model.joblib")
    return cost, claim, fraud


cost_model, claim_model, fraud_model = load_models()


# ================= SHAP EXPLAINERS =================

@st.cache_resource
def load_explainers():

    cost, claim, fraud = load_models()

    # Background data for cost model
    cost_bg = pd.DataFrame([
        [25,22,0,1,1,0,1,0],
        [45,30,2,1,0,1,0,0],
        [35,25,1,0,0,0,1,0],
        [60,28,3,1,1,0,0,1]
    ], columns=[
        "age","bmi","children","sex_male","smoker_yes",
        "region_northwest","region_southeast","region_southwest"
    ])

    # Background data for claim/fraud model
    claim_bg = pd.DataFrame([
        [25,1,50000,3,0],
        [45,2,200000,10,1],
        [35,1,75000,5,0],
        [60,3,300000,15,1]
    ], columns=[
        "age","policy_type","claim_amount","hospital_days","pre_existing"
    ])

    cost_exp = shap.Explainer(cost, cost_bg)
    claim_exp = shap.Explainer(claim, claim_bg)
    fraud_exp = shap.Explainer(fraud, claim_bg)

    return cost_exp, claim_exp, fraud_exp


cost_explainer, claim_explainer, fraud_explainer = load_explainers()


# ================= HELPERS =================

def hash_pass(p):
    return hashlib.sha256(p.encode()).hexdigest()


def ask_ai(prompt):

    base = """
You are a professional insurance advisor.
Explain clearly and simply.
"""

    try:

        res = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model":"mistral",
                "prompt": base+prompt,
                "stream":False
            },
            timeout=120
        )

        return res.json()["response"]

    except:
        return "AI unavailable"


def detect_fraud(df):
    return fraud_model.predict_proba(df)[0][1]


# ================= SHAP HELPERS =================

def plot_waterfall(vals, name):

    plt.figure()

    if len(vals.shape)==3:
        shap.plots.waterfall(vals[0,:,1], show=False)
    else:
        shap.plots.waterfall(vals[0], show=False)

    plt.tight_layout()
    plt.savefig(name)
    plt.close()


def shap_text(vals, cols, title):

    if len(vals.shape)==3:
        values = vals[0,:,1].values
    else:
        values = vals[0].values

    impacts = list(zip(cols, values))
    impacts.sort(key=lambda x: abs(x[1]), reverse=True)

    text = title + "\n\n"

    for f,v in impacts[:3]:

        if v>0:
            text += f"• {f} increased impact\n"
        else:
            text += f"• {f} reduced impact\n"

    return text


# ================= PDF =================

def generate_pdf(data, filename):

    doc = SimpleDocTemplate(filename, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []


    elements.append(Paragraph("Insurance AI Report", styles["Title"]))
    elements.append(Spacer(1,20))


    elements.append(Paragraph(f"User: {data['user']}", styles["Normal"]))
    elements.append(Paragraph(f"Date: {data['date']}", styles["Normal"]))
    elements.append(Spacer(1,20))


    table = Table(data["table"], colWidths=[200,250])

    table.setStyle(TableStyle([
        ("GRID",(0,0),(-1,-1),1,black),
        ("BACKGROUND",(0,0),(-1,0),grey)
    ]))


    elements.append(table)
    elements.append(Spacer(1,20))


    elements.append(Paragraph("AI Explanation", styles["Heading2"]))
    elements.append(Spacer(1,10))
    elements.append(Paragraph(data["ai"], styles["Normal"]))


    if data["img"]:

        elements.append(Spacer(1,20))
        elements.append(Image(data["img"], width=400, height=300))


    doc.build(elements)

    return filename


# ================= SESSION =================

if "logged" not in st.session_state:
    st.session_state.logged=False

if "user" not in st.session_state:
    st.session_state.user=None


# ================= AUTH =================

def login():

    st.subheader("🔐 Login")

    u = st.text_input("Username", key="lu")
    p = st.text_input("Password", type="password", key="lp")

    if st.button("Login"):

        user = db.query(User).filter_by(username=u).first()

        if user and user.password==hash_pass(p):

            st.session_state.logged=True
            st.session_state.user=u
            st.rerun()

        else:
            st.error("Invalid login")


def signup():

    st.subheader("📝 Signup")

    u = st.text_input("Username", key="su")
    p = st.text_input("Password", type="password", key="sp")

    if st.button("Register"):

        if db.query(User).filter_by(username=u).first():

            st.error("User exists")

        else:

            db.add(User(username=u,password=hash_pass(p)))
            db.commit()

            st.success("Account created")


# ================= LOGIN =================

if not st.session_state.logged:

    t1,t2 = st.tabs(["Login","Signup"])

    with t1: login()
    with t2: signup()

    st.stop()


# ================= MENU =================

menu = st.sidebar.radio(
    "Navigation",
    ["Cost","Claim","History","Analytics","Assistant","Profile"]
)


# ================= COST =================

if menu=="Cost":

    st.title("💰 Cost Predictor")

    with st.form("cost_form"):

        age = st.number_input("Age",18,100,25)
        bmi = st.number_input("BMI",10.0,50.0,22.0)
        children = st.number_input("Children",0,5,0)

        gender = st.selectbox("Gender",["male","female"])
        smoker = st.selectbox("Smoker",["yes","no"])
        region = st.selectbox("Region",
            ["southeast","southwest","northeast","northwest"])

        submit_cost = st.form_submit_button("📊 Predict Cost")


    if submit_cost:

        df = pd.DataFrame([[

            age,bmi,children,
            1 if gender=="male" else 0,
            1 if smoker=="yes" else 0,
            1 if region=="northwest" else 0,
            1 if region=="southeast" else 0,
            1 if region=="southwest" else 0

        ]], columns=cost_bg.columns)


        cost = cost_model.predict(df)[0]

        st.success(f"₹ {cost:,.0f}")


        vals = cost_explainer(df)

        plot_waterfall(vals,"shap_cost.png")

        st.image("shap_cost.png")


        txt = shap_text(vals,df.columns,"Cost Factors")

        st.text(txt)


        ai = ask_ai(txt)

        st.write(ai)


        table = [
            ["Age",age],
            ["BMI",bmi],
            ["Children",children],
            ["Gender",gender],
            ["Smoker",smoker],
            ["Region",region],
            ["Cost",f"₹{cost:,.0f}"]
        ]


        if st.button("📄 Download Cost Report"):

            data = {

                "user": st.session_state.user,
                "date": datetime.now().strftime("%Y-%m-%d %H:%M"),

                "table": table,
                "ai": ai,
                "img": "shap_cost.png"
            }


            file = generate_pdf(data,"cost_report.pdf")

            with open(file,"rb") as f:

                st.download_button(
                    "Download",
                    f,
                    file_name=file,
                    mime="application/pdf"
                )


# ================= CLAIM =================

elif menu=="Claim":

    st.title("🏥 Claim Analysis")

    with st.form("claim_form"):

        age = st.number_input("Age",18,100,25)
        policy = st.selectbox("Policy",[1,2,3])
        amount = st.number_input("Amount",0.0,step=1000.0)
        days = st.number_input("Days",0)
        pre = st.selectbox("Pre-existing",[0,1])

        submit_claim = st.form_submit_button("📋 Analyze Claim")


    if submit_claim:

        df = pd.DataFrame([[

            age,policy,amount,days,pre

        ]], columns=claim_bg.columns)


        pred = claim_model.predict(df)[0]

        st.success("Approved" if pred else "Rejected")


        fraud = detect_fraud(df)

        st.progress(fraud)

        st.write(f"{fraud*100:.1f}% Risk")


        cvals = claim_explainer(df)
        fvals = fraud_explainer(df)


        plot_waterfall(cvals,"shap_claim.png")
        plot_waterfall(fvals,"shap_fraud.png")


        st.image("shap_claim.png")
        st.image("shap_fraud.png")


        txt = shap_text(cvals,df.columns,"Decision Factors")
        ftxt = shap_text(fvals,df.columns,"Fraud Factors")


        st.text(txt)
        st.text(ftxt)


        ai = ask_ai(txt+"\n"+ftxt)

        st.write(ai)


        cursor.execute("""
        INSERT INTO claims_history
        VALUES (NULL,?,?,?,?,?,?,?,?)
        """,(

            age,policy,amount,days,pre,
            pred,ai,
            datetime.now().strftime("%Y-%m-%d %H:%M"),
            st.session_state.user

        ))

        conn.commit()


        table = [
            ["Age",age],
            ["Policy",policy],
            ["Amount",amount],
            ["Days",days],
            ["Pre-existing",pre],
            ["Decision","Approved" if pred else "Rejected"],
            ["Fraud",f"{fraud*100:.1f}%"]
        ]


        if st.button("📄 Download Claim Report"):

            data = {

                "user": st.session_state.user,
                "date": datetime.now().strftime("%Y-%m-%d %H:%M"),

                "table": table,
                "ai": ai,
                "img": "shap_claim.png"
            }


            file = generate_pdf(data,"claim_report.pdf")

            with open(file,"rb") as f:

                st.download_button(
                    "Download",
                    f,
                    file_name=file,
                    mime="application/pdf"
                )


# ================= HISTORY =================

elif menu=="History":

    st.title("📜 History")

    rows = cursor.execute(
        "SELECT * FROM claims_history ORDER BY id DESC"
    ).fetchall()


    if rows:

        df = pd.DataFrame(rows,columns=[
            "ID","Age","Policy","Amount",
            "Days","Pre","Decision","AI","Date","User"
        ])

        st.dataframe(df)

    else:
        st.info("No data")


# ================= ANALYTICS =================

elif menu=="Analytics":

    st.title("📊 Analytics")

    rows = cursor.execute(
        "SELECT age,claim_amount FROM claims_history"
    ).fetchall()


    if rows:

        df = pd.DataFrame(rows,columns=["Age","Amount"])

        st.line_chart(df.set_index("Age"))

    else:
        st.info("No data")


# ================= ASSISTANT =================

elif menu=="Assistant":

    st.title("🤖 Assistant")

    q = st.text_area("Ask")

    if st.button("Send") and q:

        st.write(ask_ai(q))


# ================= PROFILE =================

elif menu=="Profile":

    st.title("👤 Dashboard")

    user = st.session_state.user


    rows = cursor.execute("""
        SELECT decision, claim_amount
        FROM claims_history
        WHERE username=?
    """,(user,)).fetchall()


    if not rows:

        st.info("No claims yet")
        st.stop()


    df = pd.DataFrame(rows,columns=["Decision","Amount"])


    total = len(df)

    approved = df["Decision"].sum()

    rejected = total-approved

    avg = df["Amount"].mean()


    c1,c2,c3,c4 = st.columns(4)

    c1.metric("Total",total)
    c2.metric("Approved",approved)
    c3.metric("Rejected",rejected)
    c4.metric("Avg",f"₹{avg:,.0f}")


    st.bar_chart(df["Decision"].value_counts())


    risk = rejected/total


    if risk<0.3:
        st.success("Low Risk")

    elif risk<0.6:
        st.warning("Medium Risk")

    else:
        st.error("High Risk")
