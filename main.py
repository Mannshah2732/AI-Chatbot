import flask
from flask import Flask, request, jsonify, render_template
import json
import requests
import random
import dotenv
import os
import mysql.connector
import pymysql
import datetime
from dotenv import load_dotenv

from agent import create_agent

app = Flask(__name__)

load_dotenv()

# DB Config
MYSQL_USER = os.getenv("DB_USER")
MYSQL_PASSWORD = os.getenv("DB_PASSWORD")
MYSQL_HOST = os.getenv("DB_HOST")
MYSQL_DB = os.getenv("DB_NAME")

MAKE_WEBHOOK_URL = os.getenv("MAKE_WEBHOOK_URL")
MAKE_WEBHOOK_SECRET = os.getenv("MAKE_WEBHOOK_SECRET")
CLIENT_ID = os.getenv("CLIENT_ID", "robeck_dental")

def get_db():
    conn = mysql.connector.connect(
        host=MYSQL_HOST,
        port=3306,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DB,
    )
    cursor = conn.cursor(dictionary=True)
    return conn, cursor

def log_chat(chat_id, client_id, user_message, ai_message, is_lead=0, intent_tag=None, confidence_score=None):
    """
    Inserts a single chat turn into robeck_dental_chat_logs.
    Returns the inserted row id.
    """
    sql = """
        INSERT INTO robeck_dental_chat_logs
            (chat_id, client_id, user_message, ai_message, is_lead, intent_tag, confidence_score)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """
    params = (chat_id, client_id, user_message, ai_message, int(is_lead), intent_tag, confidence_score)

    conn, cursor = None, None
    try:
        conn, cursor = get_db()
        cursor.execute(sql, params)
        conn.commit()
        return cursor.lastrowid
    finally:
        if cursor: cursor.close()
        if conn: conn.close()

import re

# ---- Phone normalization ----
DEFAULT_COUNTRY_CODE = os.getenv("DEFAULT_COUNTRY_CODE", "")  # e.g. "+91" or "+1" (leave empty to not prepend)
PHONE_CANDIDATE_RE = re.compile(r'(\+?\d[\d\-\s\(\)]{6,}\d)')  # matches + and 7-20 digits with separators

def normalize_phone(value: str) -> str | None:
    """Return E.164-like string: keep leading + if present, strip spaces/dashes/()"""
    if not value:
        return None
    m = PHONE_CANDIDATE_RE.search(value)
    if not m:
        return None
    raw = m.group(1)
    has_plus = raw.strip().startswith('+')
    digits = ''.join(ch for ch in raw if ch.isdigit())
    if not digits:
        return None
    if has_plus:
        return '+' + digits
    # No plus: optionally prepend your default country code
    if DEFAULT_COUNTRY_CODE:
        return DEFAULT_COUNTRY_CODE + digits
    return digits


def detect_intent_and_lead(text: str):
    # very basic examples; customize as you like
    lead_patterns = [r'\bappointment\b', r'\bbook\b', r'\bcall me\b', r'\bphone\b', r'\bconsultation\b']
    implant_patterns = [r'\bimplant', r'\bmissing tooth', r'\bdental implant']

    is_lead = any(re.search(p, text, re.I) for p in lead_patterns)
    intent = None
    if any(re.search(p, text, re.I) for p in implant_patterns):
        intent = "implant_interest"
    return intent, int(is_lead)

# ---------- Make.com ----------
def send_to_make(payload: dict):
    if not MAKE_WEBHOOK_URL:
        return False, "MAKE_WEBHOOK_URL not set"
    headers = {"Content-Type": "application/json"}
    if MAKE_WEBHOOK_SECRET:
        headers["X-Webhook-Secret"] = MAKE_WEBHOOK_SECRET
    try:
        r = requests.post(MAKE_WEBHOOK_URL, json=payload, headers=headers, timeout=10)
        r.raise_for_status()
        return True, r.text
    except Exception as e:
        return False, str(e)

def _normalize_leads(lead_payload):
    """Accept dict or list; return a dict or None."""
    if not lead_payload:
        return None
    if isinstance(lead_payload, dict):
        return lead_payload
    if isinstance(lead_payload, list):
        # pick the first dict-like item
        for item in lead_payload:
            if isinstance(item, dict):
                return item
    return None


# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Chatbot UI route
@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')

def send_to_make(payload: dict):
    try:
        r = requests.post(MAKE_WEBHOOK_URL, json=payload, timeout=8)
        r.raise_for_status()
        return True, r.text
    except Exception as e:
        return False, str(e)


# Chatbot response route
@app.route('/get_response', methods=['POST'])
def get_response():
    user_input = request.json.get("message", "").strip()
    chat_id = (request.json or {}).get("chat_id") or "robeck-dental"
    client_id = "robeck_dental"

    if not user_input:
        return jsonify({"response": "Please enter a valid message."})

    agent = create_agent()

    try:
        raw = agent.invoke(
            {"input": user_input},
            {"configurable": {"session_id": chat_id}}
        )

        # 1) Normalize agent output to `out`
        out = raw["output"] if isinstance(raw, dict) and "output" in raw else raw

        final_message = None
        lead_payload = None

        # 2) If dict already (e.g., {"type":"text","text":"..."})
        if isinstance(out, dict):
            if out.get("type") == "text" and "text" in out:
                final_message = out["text"]
            elif out.get("type") == "leads" and "leads" in out:
                lead_payload = out["leads"]
                final_message = out.get("message") or "Thanks! I’ve recorded your details."
            else:
                # Unknown dict → return as-is
                return jsonify(out)

        # 3) If string, try JSON first, else treat as plain text
        elif isinstance(out, str):
            s = out.strip()
            if s.startswith("{") and s.endswith("}"):
                try:
                    j = json.loads(s)
                    if isinstance(j, dict) and j.get("type") == "text" and "text" in j:
                        final_message = j["text"]
                    elif isinstance(j, dict) and j.get("type") == "leads" and "leads" in j:
                        lead_payload = j["leads"]
                        final_message = j.get("message") or "Thanks! I’ve recorded your details."
                    else:
                        return jsonify(j)
                except json.JSONDecodeError:
                    final_message = s
            else:
                final_message = s
        else:
            final_message = str(out)

        # 4) If we got leads, send to Make and (optionally) mark as lead
#        is_lead = 0
#        if lead_payload:
#            payload = {
#                "timestamp": datetime.datetime.now().isoformat() + "Z",
#                "client_id": client_id,
#                "chat_id": chat_id,
#                "name": lead_payload.get("name"),
#                "phone": lead_payload.get("phone"),
#                "email": lead_payload.get("email"),
#                "consent": bool(lead_payload.get("consent", True)),
#                "source": "chatbot",
#                "latest_user_message": user_input,
#                "latest_ai_message": final_message,
#            }
#            send_to_make(payload)
#            is_lead = 1

        is_lead = 0
        if lead_payload:
            leads = _normalize_leads(lead_payload)
            raw_phone = leads.get("phone") if leads else None
            phone = normalize_phone(raw_phone) or normalize_phone(user_input)  # fallback: extract from the user message

            if leads:  # only proceed if we have a dict
                payload = {
                    "timestamp": datetime.datetime.now().isoformat() + "Z",
                    "client_id": client_id,
                    "chat_id": chat_id,
                    "name": leads.get("name"),
                    "phone": phone,
                    "email": leads.get("email"),
                    "consent": bool(leads.get("consent", True)),
                    "source": "chatbot",
                    "latest_user_message": user_input,
                    "latest_ai_message": final_message,
                }
                send_to_make(payload)
                is_lead = 1

        # 5) Log to DB
        intent_tag, heuristic_lead = detect_intent_and_lead(user_input)
        log_id = log_chat(
            chat_id=chat_id,
            client_id=client_id,
            user_message=user_input,
            ai_message=final_message,
            is_lead=is_lead or heuristic_lead,
            intent_tag=intent_tag,
            confidence_score=None
        )

        return jsonify({"response": final_message, "log_id": log_id, "chat_id": chat_id})

    except Exception as e:
        return jsonify({"response": f"Error: {str(e)}", "chat_id": chat_id})
if __name__ == '__main__':
    app.run(debug=True)