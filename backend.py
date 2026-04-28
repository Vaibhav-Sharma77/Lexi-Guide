"""
Lexi-Guide Backend API (FULL FIXED VERSION)
"""

import os
import logging
import re
import json
from datetime import datetime
from typing import Optional, Literal

from dotenv import load_dotenv
import google.generativeai as genai

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

import uvicorn

# ─────────────────────────────────────────────
# ENV + LOGGING
# ─────────────────────────────────────────────
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# FASTAPI INIT
# ─────────────────────────────────────────────
app = FastAPI(
    title="Lexi-Guide API",
    description="AI-powered legal contract analysis for MSMEs",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# GEMINI SETUP (FIXED)
# ─────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("❌ GEMINI_API_KEY missing")

genai.configure(api_key=GEMINI_API_KEY)

try:
    # ✅ Stable model
    model = genai.GenerativeModel("gemini-1.5-flash")
    logger.info("✅ Gemini initialized")
except Exception as e:
    logger.error(f"❌ Gemini init failed: {e}")
    model = None

# ─────────────────────────────────────────────
# REQUEST MODEL
# ─────────────────────────────────────────────
class ContractAnalysisRequest(BaseModel):
    contract_text: str = Field(..., min_length=50)
    user_role: Literal["student", "freelancer", "client", "startup", "vendor", "legal"] = "student"
    country: str = "India"
    user_id: Optional[str] = None

    @field_validator("country")
    @classmethod
    def validate_country(cls, v):
        if len(v.strip()) < 2:
            raise ValueError("Country too short")
        return v.strip().title()

# ─────────────────────────────────────────────
# HEALTH MODEL
# ─────────────────────────────────────────────
class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str

# ─────────────────────────────────────────────
# PROMPT TEMPLATE (YOUR ORIGINAL)
# ─────────────────────────────────────────────
MASTER_PROMPT_TEMPLATE = """
You are Lexi-Guide AI.

Analyze contract for {user_role} in {country}.

Return ONLY JSON:

{{
  "legal_safety_index": {{
    "score": 0-100,
    "justification": ""
  }},
  "clauses": [
    {{
      "clause_title": "",
      "original_text": "",
      "risk_level": "",
      "impact_first_explanation": "",
      "safer_suggestion": ""
    }}
  ]
}}

Contract:
{contract_text}
"""

# ─────────────────────────────────────────────
# ROLE CONTEXT
# ─────────────────────────────────────────────
def get_role_context(role: str):
    return {
        "freelancer": "Focus on payments, IP, risks",
        "startup": "Focus on scalability and liability",
        "client": "Focus on quality and deliverables"
    }.get(role, "General analysis")

# ─────────────────────────────────────────────
# COUNTRY CONTEXT
# ─────────────────────────────────────────────
def get_country_context(country: str):
    if country == "India":
        return "Indian Contract Act applies"
    return "General contract law applies"

# ─────────────────────────────────────────────
# GEMINI RESPONSE EXTRACT
# ─────────────────────────────────────────────
def extract_text(response):
    try:
        if hasattr(response, "text") and response.text:
            return response.text

        if hasattr(response, "candidates"):
            return response.candidates[0].content.parts[0].text
    except Exception as e:
        logger.error(f"Extract error: {e}")

    return None

# ─────────────────────────────────────────────
# JSON CLEANER (CRITICAL FIX)
# ─────────────────────────────────────────────
def parse_json(text: str):
    try:
        text = re.sub(r"```json", "", text)
        text = re.sub(r"```", "", text)

        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError("No JSON found")

        clean = match.group()
        return json.loads(clean)

    except Exception as e:
        logger.error(f"JSON ERROR: {e}")
        logger.error(f"RAW:\n{text}")
        raise HTTPException(500, "Invalid AI response")

# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────
@app.get("/", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        version="1.0.0"
    )

@app.post("/analyze")
async def analyze_contract(request: ContractAnalysisRequest):

    try:
        if len(request.contract_text.strip()) < 50:
            raise HTTPException(400, "Contract too short")

        logger.info(f"Analyzing for {request.user_role}")

        role_context = get_role_context(request.user_role)
        country_context = get_country_context(request.country)

        prompt = MASTER_PROMPT_TEMPLATE.format(
            user_role=request.user_role,
            country=request.country,
            contract_text=request.contract_text
        )

        if model is None:
            return get_mock_response()

        # ✅ Gemini call FIXED
        response = model.generate_content(
            prompt,
            generation_config={
                "response_mime_type": "application/json"
            }
        )

        text = extract_text(response)

        if not text:
            raise HTTPException(500, "Empty AI response")

        logger.info(f"AI RESPONSE:\n{text}")

        result = parse_json(text)

        # validate structure
        if "legal_safety_index" not in result or "clauses" not in result:
            raise HTTPException(500, "Invalid structure")

        return result

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"ERROR: {e}")
        raise HTTPException(500, "Contract analysis failed")

# ─────────────────────────────────────────────
# MOCK (fallback)
# ─────────────────────────────────────────────
def get_mock_response():
    return {
        "legal_safety_index": {
            "score": 60,
            "justification": "Moderate risk contract"
        },
        "clauses": [
            {
                "clause_title": "Payment",
                "original_text": "Payment after 90 days",
                "risk_level": "High",
                "impact_first_explanation": "This means delayed cash flow",
                "safer_suggestion": "Reduce to 30 days"
            }
        ]
    }

# ─────────────────────────────────────────────
# GLOBAL ERROR HANDLER
# ─────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_handler(request: Request, exc: Exception):
    logger.error(f"GLOBAL ERROR: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"}
    )

# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("backend:app", host="0.0.0.0", port=port)
