# streamlit/app.py

# RUN:
# cd streamlit
# streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

import joblib
import json

# -----------------------------------
# PAGE CONFIG
# -----------------------------------
st.set_page_config(
    page_title="FD Loan Risk Engine",
    page_icon="🏦",
    layout="wide"
)

# -----------------------------------
# LOAD MODELS
# -----------------------------------
@st.cache_resource
def load():

    return {

        'xgb':
            joblib.load("../models/xgb_model.pkl"),

        'lr':
            joblib.load("../models/lr_model.pkl"),

        'sc':
            joblib.load("../models/lr_scaler.pkl"),

        'lc':
            joblib.load("../models/le_city.pkl"),

        'le':
            joblib.load("../models/le_emp.pkl"),

        'lb':
            joblib.load("../models/le_bank.pkl"),

        'xm':
            json.load(open("../models/xgb_metrics.json")),

        'lm':
            json.load(open("../models/lr_metrics.json")),
    }

M = load()

FEATURES = M['xm']['features']

# -----------------------------------
# HEADER
# -----------------------------------
st.markdown(
    """
    <div style="
        background:linear-gradient(135deg,#1E3A5F,#065F46);
        padding:22px;
        border-radius:12px;
        margin-bottom:18px">

        <h1 style="color:white;margin:0">
            Slice FD-Backed Loan Risk Engine
        </h1>

        <p style="color:#6EE7B7;margin:0">
            LR Baseline + XGBoost + LTV Optimizer
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# -----------------------------------
# METRICS
# -----------------------------------
c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "LR AUC-ROC",
    M['lm']['auc_roc']
)

c2.metric(
    "XGB AUC-ROC",
    M['xm']['auc_roc']
)

c3.metric(
    "XGB Lift",
    f"+{(M['xm']['auc_roc'] - M['lm']['auc_roc']) * 100:.1f}%"
)

c4.metric(
    "Default Rate",
    "~3.8%"
)

st.markdown("---")

# -----------------------------------
# TABS
# -----------------------------------
tab1, tab2 = st.tabs([
    "Risk Assessment",
    "LTV Optimizer"
])

# -----------------------------------
# FEATURE GENERATOR
# -----------------------------------
def get_features(
    age,
    inc,
    cibil,
    emp,
    city,
    bank,
    fd_amt,
    fd_ten,
    fd_rate,
    fd_mat,
    ltv,
    loan_ten,
    exist_loans,
    missed,
    preclosure,
    rel_yrs,
    digital
):

    try:

        ec = int(M['le'].transform([emp])[0])

        cc = int(M['lc'].transform([city])[0])

        bc = int(M['lb'].transform([bank])[0])

    except:

        ec = cc = bc = 0

    # -----------------------------------
    # LOAN CALCULATIONS
    # -----------------------------------
    loan_amt = fd_amt * ltv

    loan_int = fd_rate + (
        2.5 if ltv > 0.80 else 1.0
    )

    er = loan_int / (12 * 100)

    emi = (
        loan_amt
        * er
        * (1 + er) ** loan_ten
        /
        ((1 + er) ** loan_ten - 1)
    )

    emi_inc = emi / max(inc, 1)

    # -----------------------------------
    # DATAFRAME
    # -----------------------------------
    return pd.DataFrame([{

        'age': age,
        'income_mo': inc,
        'cibil': cibil,

        'emp_enc': ec,
        'city_enc': cc,
        'bank_enc': bc,

        'fd_amount': fd_amt,
        'fd_tenure_mo': fd_ten,
        'fd_interest_rate': fd_rate,
        'fd_months_to_maturity': fd_mat,

        'loan_amt': loan_amt,
        'ltv': ltv,
        'loan_tenure_mo': loan_ten,
        'loan_interest': loan_int,

        'emi': emi,
        'emi_income': emi_inc,

        'num_existing_loans': exist_loans,
        'missed_pmts_hist': missed,
        'pre_closure_risk': preclosure,

        'relationship_yrs': rel_yrs,
        'digital_user': digital,

        'fd_coverage': fd_amt / (loan_amt + 1),

        'maturity_buffer':
            fd_mat - loan_ten,

        'interest_spread':
            loan_int - fd_rate,

        'ltv_band_num':
            min(int((ltv - 0.40) / 0.10), 4),

        'high_ltv':
            int(ltv > 0.80),

        'relationship_score':
            rel_yrs
            * digital
            * (1 - preclosure),

        'debt_burden':
            exist_loans * emi_inc,

        'fd_density':
            fd_amt / (fd_ten / 12 + 1),

    }])[FEATURES]

# ====================================================
# TAB 1 — RISK ASSESSMENT
# ====================================================
with tab1:

    c1, c2, c3 = st.columns(3)

    # -----------------------------------
    # COLUMN 1
    # -----------------------------------
    with c1:

        age_ = st.slider(
            "Age",
            18,
            80,
            42
        )

        cibil_ = st.slider(
            "CIBIL",
            300,
            900,
            720
        )

        inc_ = st.number_input(
            "Monthly Income",
            5000,
            500000,
            85000,
            5000
        )

        emp_ = st.selectbox(
            "Employment",
            [
                "Salaried",
                "Self-Employed",
                "Retired",
                "Business"
            ]
        )

    # -----------------------------------
    # COLUMN 2
    # -----------------------------------
    with c2:

        fd_amt = st.number_input(
            "FD Amount (INR)",
            10000,
            5000000,
            500000,
            10000
        )

        fd_ten = st.selectbox(
            "FD Tenure (months)",
            [12, 24, 36, 48, 60, 84]
        )

        fd_rt = st.slider(
            "FD Interest Rate (%)",
            5.0,
            9.5,
            7.5,
            0.1
        )

        fd_mat = st.slider(
            "Months to FD Maturity",
            3,
            84,
            24
        )

    # -----------------------------------
    # COLUMN 3
    # -----------------------------------
    with c3:

        ltv_ = st.slider(
            "LTV Ratio",
            0.40,
            0.95,
            0.70,
            0.01
        )

        l_ten = st.selectbox(
            "Loan Tenure (months)",
            [12, 24, 36, 48]
        )

        miss_ = st.number_input(
            "Missed Payment History",
            0,
            10,
            0
        )

        rel_ = st.slider(
            "Relationship Years",
            0,
            25,
            5
        )

    # -----------------------------------
    # BUTTON
    # -----------------------------------
    if st.button(
        "ASSESS RISK",
        type="primary",
        use_container_width=True
    ):

        X = get_features(
            age_,
            inc_,
            cibil_,
            emp_,
            "Tier1",
            "HDFC",
            fd_amt,
            fd_ten,
            fd_rt,
            fd_mat,
            ltv_,
            l_ten,
            1,
            miss_,
            0,
            rel_,
            1
        )

        prob = float(
            M['xgb'].predict_proba(X)[0][1]
        )

        # -----------------------------------
        # RISK LOGIC
        # -----------------------------------
        risk = (
            "HIGH"
            if prob > 0.07
            else "MEDIUM"
            if prob > 0.03
            else "LOW"
        )

        dec = (
            "REJECT"
            if prob > 0.07
            else "REVIEW"
            if prob > 0.03
            else "APPROVE"
        )

        col_c = (
            "red"
            if dec == "REJECT"
            else "orange"
            if dec == "REVIEW"
            else "green"
        )

        c_a, c_b = st.columns(2)

        # -----------------------------------
        # RESULT TEXT
        # -----------------------------------
        with c_a:

            st.markdown(
                f"<h2 style='color:{col_c}'>{dec} — {risk}</h2>",
                unsafe_allow_html=True
            )

            st.metric(
                "Default Probability",
                f"{prob * 100:.3f}%"
            )

            st.metric(
                "LTV",
                f"{ltv_:.2f}"
            )

            loan_amt_show = int(
                fd_amt * ltv_ / 1000
            ) * 1000

            st.metric(
                "Loan Amount",
                f"Rs {loan_amt_show:,}"
            )

        # -----------------------------------
        # GAUGE CHART
        # -----------------------------------
        with c_b:

            fig = go.Figure(

                go.Indicator(
                    mode="gauge+number",

                    value=prob * 100,

                    gauge={
                        "axis": {
                            "range": [0, 15]
                        },

                        "bar": {
                            "color": col_c
                        },

                        "steps": [
                            {
                                "range": [0, 3],
                                "color": "#D1FAE5"
                            },
                            {
                                "range": [3, 7],
                                "color": "#FEF9C3"
                            },
                            {
                                "range": [7, 15],
                                "color": "#FEE2E2"
                            }
                        ]
                    },

                    title={
                        "text":
                        "Default Probability (%)"
                    }
                )
            )

            fig.update_layout(height=280)

            st.plotly_chart(
                fig,
                use_container_width=True
            )

# ====================================================
# TAB 2 — LTV OPTIMIZER
# ====================================================
with tab2:

    st.subheader(
        "Find Maximum Safe LTV for Customer"
    )

    c1, c2 = st.columns(2)

    # -----------------------------------
    # COLUMN 1
    # -----------------------------------
    with c1:

        age2 = st.slider(
            "Age ",
            18,
            80,
            42,
            key="a2"
        )

        cibil2 = st.slider(
            "CIBIL ",
            300,
            900,
            720,
            key="c2"
        )

        inc2 = st.number_input(
            "Monthly Income ",
            5000,
            500000,
            85000,
            key="i2"
        )

    # -----------------------------------
    # COLUMN 2
    # -----------------------------------
    with c2:

        fd2 = st.number_input(
            "FD Amount ",
            10000,
            5000000,
            500000,
            key="f2"
        )

        fd_rt2 = st.slider(
            "FD Rate (%)",
            5.0,
            9.5,
            7.5,
            0.1,
            key="fr2"
        )

        miss2 = st.number_input(
            "Missed Payments ",
            0,
            10,
            0,
            key="m2"
        )

    # -----------------------------------
    # BUTTON
    # -----------------------------------
    if st.button(
        "FIND MAX SAFE LTV",
        type="primary",
        use_container_width=True
    ):

        best_ltv = 0.50

        for ltv_t in np.arange(
            0.50,
            0.95,
            0.01
        ):

            X = get_features(
                age2,
                inc2,
                cibil2,
                "Salaried",
                "Tier1",
                "HDFC",
                fd2,
                36,
                fd_rt2,
                24,
                ltv_t,
                24,
                1,
                miss2,
                0,
                5,
                1
            )

            p = float(
                M['xgb'].predict_proba(X)[0][1]
            )

            if p <= 0.05:
                best_ltv = ltv_t
            else:
                break

        max_loan = int(
            fd2 * best_ltv / 1000
        ) * 1000

        st.success(
            f"Maximum Safe LTV: {best_ltv:.2f} | "
            f"Max Loan: Rs {max_loan:,}"
        )

        st.info(
            "At this LTV, predicted default probability <= 5%"
        )