# api/fd_api.py

# RUN:
# cd api
# uvicorn fd_api:app --reload --port 8010

# TEST:
# http://localhost:8010/docs

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import joblib
import json
import time

import numpy as np
import pandas as pd

# -----------------------------------
# FASTAPI APP
# -----------------------------------
app = FastAPI(
    title="Slice FD-Backed Loan Risk API",
    version="1.0.0"
)

# -----------------------------------
# CORS
# -----------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# -----------------------------------
# LOAD MODELS
# -----------------------------------
xgb_model = joblib.load("../models/xgb_model.pkl")

lr_model = joblib.load("../models/lr_model.pkl")

lr_scaler = joblib.load("../models/lr_scaler.pkl")

# -----------------------------------
# LOAD LABEL ENCODERS
# -----------------------------------
le_emp = joblib.load("../models/le_emp.pkl")

le_city = joblib.load("../models/le_city.pkl")

le_bank = joblib.load("../models/le_bank.pkl")

# -----------------------------------
# LOAD METRICS
# -----------------------------------
xgb_m = json.load(
    open("../models/xgb_metrics.json")
)

FEATURES = xgb_m["features"]

# -----------------------------------
# REQUEST SCHEMA
# -----------------------------------
class FDLoanRequest(BaseModel):

    age: int = Field(..., ge=18, le=80, example=42)

    income_mo: float = Field(..., ge=5000, example=85000)

    cibil: int = Field(
        720,
        ge=300,
        le=900,
        example=720
    )

    employment: str = Field(
        "Salaried",
        example="Salaried"
    )

    city_tier: str = Field(
        "Tier1",
        example="Tier1"
    )

    fd_amount: float = Field(
        ...,
        ge=10000,
        example=500000
    )

    fd_tenure_mo: int = Field(
        36,
        ge=6,
        le=120,
        example=36
    )

    fd_interest_rate: float = Field(
        7.5,
        ge=5.0,
        le=9.5,
        example=7.5
    )

    fd_bank: str = Field(
        "HDFC",
        example="HDFC"
    )

    fd_months_to_maturity: int = Field(
        24,
        ge=3,
        le=120,
        example=24
    )

    ltv: float = Field(
        0.70,
        ge=0.40,
        le=0.95,
        example=0.70
    )

    loan_tenure_mo: int = Field(
        24,
        ge=6,
        le=60,
        example=24
    )

    num_existing_loans: int = Field(
        1,
        ge=0,
        example=1
    )

    missed_pmts_hist: int = Field(
        0,
        ge=0,
        le=10,
        example=0
    )

    pre_closure_risk: int = Field(
        0,
        ge=0,
        le=1,
        example=0
    )

    relationship_yrs: int = Field(
        5,
        ge=0,
        example=5
    )

    digital_user: int = Field(
        1,
        ge=0,
        le=1,
        example=1
    )

# -----------------------------------
# FEATURE ENGINEERING FUNCTION
# -----------------------------------
def build_features(req: FDLoanRequest) -> pd.DataFrame:

    try:

        emp_enc = int(
            le_emp.transform([req.employment])[0]
        )

        city_enc = int(
            le_city.transform([req.city_tier])[0]
        )

        bank_enc = int(
            le_bank.transform([req.fd_bank])[0]
        )

    except ValueError as e:

        raise HTTPException(
            status_code=400,
            detail=f"Invalid value: {e}"
        )

    # -----------------------------------
    # LOAN CALCULATIONS
    # -----------------------------------
    loan_amt = req.fd_amount * req.ltv

    loan_interest = (
        req.fd_interest_rate
        + (2.5 if req.ltv > 0.80 else 1.0)
    )

    emi_rate = loan_interest / (12 * 100)

    emi = (
        loan_amt
        * emi_rate
        * (1 + emi_rate) ** req.loan_tenure_mo
        /
        ((1 + emi_rate) ** req.loan_tenure_mo - 1)
    )

    emi_income = emi / max(req.income_mo, 1)

    # -----------------------------------
    # FEATURE DATAFRAME
    # -----------------------------------
    return pd.DataFrame([{

        'age': req.age,
        'income_mo': req.income_mo,
        'cibil': req.cibil,

        'emp_enc': emp_enc,
        'city_enc': city_enc,
        'bank_enc': bank_enc,

        'fd_amount': req.fd_amount,
        'fd_tenure_mo': req.fd_tenure_mo,
        'fd_interest_rate': req.fd_interest_rate,
        'fd_months_to_maturity': req.fd_months_to_maturity,

        'loan_amt': loan_amt,
        'ltv': req.ltv,
        'loan_tenure_mo': req.loan_tenure_mo,
        'loan_interest': loan_interest,

        'emi': emi,
        'emi_income': emi_income,

        'num_existing_loans': req.num_existing_loans,
        'missed_pmts_hist': req.missed_pmts_hist,
        'pre_closure_risk': req.pre_closure_risk,

        'relationship_yrs': req.relationship_yrs,
        'digital_user': req.digital_user,

        'fd_coverage': req.fd_amount / (loan_amt + 1),

        'maturity_buffer':
            req.fd_months_to_maturity
            - req.loan_tenure_mo,

        'interest_spread':
            loan_interest
            - req.fd_interest_rate,

        'ltv_band_num':
            min(int((req.ltv - 0.40) / 0.10), 4),

        'high_ltv':
            int(req.ltv > 0.80),

        'relationship_score':
            req.relationship_yrs
            * req.digital_user
            * (1 - req.pre_closure_risk),

        'debt_burden':
            req.num_existing_loans
            * emi_income,

        'fd_density':
            req.fd_amount
            / (req.fd_tenure_mo / 12 + 1),

    }])[FEATURES]

# -----------------------------------
# ROOT ENDPOINT
# -----------------------------------
@app.get("/")
def root():

    return {
        "status": "running",
        "models": [
            "logistic_regression",
            "xgboost"
        ],
        "xgb_auc": xgb_m["auc_roc"],
        "lr_auc": xgb_m["lr_auc"]
    }

# -----------------------------------
# RISK PREDICTION ENDPOINT
# -----------------------------------
@app.post("/predict/risk")
def predict_risk(req: FDLoanRequest):

    start = time.time()

    feats = build_features(req)

    prob = float(
        xgb_model.predict_proba(feats)[0][1]
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

    decision = (
        "REJECT"
        if prob > 0.07
        else "REVIEW"
        if prob > 0.03
        else "APPROVE"
    )

    # -----------------------------------
    # FLAGS
    # -----------------------------------
    flags = []

    if req.ltv > 0.85:
        flags.append(
            f"High LTV ({req.ltv:.2f})"
        )

    if req.cibil < 650:
        flags.append(
            f"Low CIBIL ({req.cibil})"
        )

    if req.missed_pmts_hist > 0:
        flags.append(
            f"Past delinquency ({req.missed_pmts_hist})"
        )

    if req.pre_closure_risk:
        flags.append(
            "Pre-closure risk flagged"
        )

    return {

        "default_probability":
            round(prob, 4),

        "risk_level":
            risk,

        "decision":
            decision,

        "risk_flags":
            flags,

        "latency_ms":
            round(
                (time.time() - start) * 1000,
                2
            )
    }

# -----------------------------------
# LTV OPTIMIZER ENDPOINT
# -----------------------------------
@app.post("/optimize/ltv")
def optimize_ltv(req: FDLoanRequest):

    best_ltv = 0.50

    for ltv_test in np.arange(
        0.50,
        0.95,
        0.01
    ):

        test_req = req.copy()

        test_req.ltv = round(
            ltv_test,
            2
        )

        feats = build_features(test_req)

        prob = float(
            xgb_model.predict_proba(feats)[0][1]
        )

        if prob <= 0.05:
            best_ltv = ltv_test
        else:
            break

    max_loan = int(
        req.fd_amount
        * best_ltv
        / 1000
    ) * 1000

    return {

        "max_safe_ltv":
            round(best_ltv, 2),

        "max_loan_amount":
            max_loan,

        "fd_amount":
            req.fd_amount,

        "risk_threshold":
            "5%"
    }