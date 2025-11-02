"""
Lexi-Guide Backend API
A FastAPI server that analyzes legal contracts using Google Gemini AI
"""

import os
import logging
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Literal
from datetime import datetime

from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Lexi-Guide API",
    description="AI-powered legal contract analysis for Indian MSMEs",
    version="1.0.0"
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development - configure properly for production
    allow_credentials=False,  # Set to False when allow_origins is "*"
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Initialize Gemini AI - API key loaded from environment
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY not found in environment variables!")
    logger.error("Please check your .env file or environment configuration.")
    raise ValueError("GEMINI_API_KEY is required but not found in environment")

try:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-2.0-flash')  # Updated to correct model name
    logger.info("Gemini AI initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Gemini AI: {e}")
    model = None

# Request/Response models
class ContractAnalysisRequest(BaseModel):
    contract_text: str = Field(..., min_length=50, description="The contract text to analyze")
    user_role: Literal["student", "freelancer", "client", "startup", "vendor", "legal"] = Field(
        default="student", 
        description="User's business role"
    )
    country: str = Field(default="India", description="Country/jurisdiction for legal analysis")
    user_id: Optional[str] = Field(default=None, description="Optional user identifier")
    
    @field_validator('user_role')
    @classmethod
    def validate_user_role(cls, v):
        valid_roles = ["student", "freelancer", "client", "startup", "vendor", "legal"]
        if v not in valid_roles:
            raise ValueError(f"Invalid user role. Must be one of: {', '.join(valid_roles)}")
        return v
    
    @field_validator('country')
    @classmethod
    def validate_country(cls, v):
        if len(v.strip()) < 2:
            raise ValueError("Country name must be at least 2 characters long")
        return v.strip().title()

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str

# Master AI Prompt Template
MASTER_PROMPT_TEMPLATE = """
**SYSTEM PROMPT:**
You are "Lexi-Guide," an expert AI legal analyst specialized in Indian contract law for Micro, Small, and Medium Enterprises (MSMEs). Your mission is to help non-lawyers understand legal documents by identifying risks, simplifying complex language, and providing actionable, safer alternatives. You are empathetic, clear, and your tone is that of a trustworthy mentor.

**USER PROMPT:**
Analyze the following legal document for a user who is a **{user_role}** in **{country}**.

**Role-Specific Context:**
{role_context}

**Country-Specific Legal Context:**
{country_context}

**Contract Text:**
```
{contract_text}
```

**Instructions:**
Respond ONLY with a valid JSON object. Do not include any text, markdown formatting, or explanations before or after the JSON. The JSON object must have the following structure:

{{
  "legal_safety_index": {{
    "score": <An integer from 0 (extremely risky) to 100 (very safe)>,
    "justification": "A brief, 1-2 sentence explanation for the score, mentioning the 2-3 biggest risk factors for a {user_role} in {country}."
  }},
  "clauses": [
    {{
      "clause_title": "<The title or a short, descriptive summary of the clause, e.g., '5. Payment Terms'>",
      "original_text": "<The exact text of the clause from the document>",
      "risk_level": "<'High', 'Medium', 'Low', or 'None'>",
      "impact_first_explanation": "Explain what this means for the {user_role} in {country}, in simple terms. Start with 'This means...'. Use local legal context where relevant.",
      "safer_suggestion": "<If risk is High or Medium, suggest a concrete, well-worded alternative clause that reduces the {user_role}'s risk under {country} law. If risk is Low or None, state 'No suggestion needed. This clause is well-balanced under {country} law.'>"
    }}
  ]
}}
"""

def get_role_context(user_role: str) -> str:
    """Get role-specific context for the AI prompt"""
    role_contexts = {
        "student": "As a student, focus on learning-oriented contracts like internships, academic projects, or part-time work. Emphasize educational value, fair compensation for student work, flexible scheduling for studies, and protection against exploitative terms that might interfere with academic progress.",
        "freelancer": "As a freelancer, focus on project-based work agreements, payment terms, intellectual property rights, scope creep protection, and client relationship management. Emphasize clear deliverables, milestone payments, and termination clauses that protect independent contractor status.",
        "client": "As a client engaging freelancers or service providers, focus on service quality assurance, timely deliverables, confidentiality protection, and value for money. Emphasize clear expectations, performance standards, and remedies for non-compliance while maintaining fair terms.",
        "startup": "As a startup, focus on cost-effective agreements, scalability clauses, intellectual property protection, and investor-friendly terms. Emphasize flexibility for growth, reasonable liability limits, and terms that don't hinder future funding or partnerships.",
        "vendor": "As a vendor or service provider, focus on sustainable business terms, payment security, scope of work clarity, and long-term relationship building. Emphasize fair pricing, reasonable liability limits, and terms that support business growth and client retention.",
        "legal": "As a legal professional, focus on comprehensive risk assessment, regulatory compliance, enforceability under applicable law, and professional liability considerations. Emphasize thorough analysis of all clauses, potential legal challenges, and industry best practices."
    }
    return role_contexts.get(user_role, "General business perspective focusing on balanced terms, risk mitigation, and legal compliance.")

def get_country_context(country: str) -> str:
    """Get country-specific legal context for the AI prompt"""
    country_contexts = {
        "India": "India follows a mixed legal system with common law principles and civil law influences. Key considerations include the Indian Contract Act 1872, specific labor laws, GST implications, and jurisdiction-specific regulations. Courts favor written agreements and specific performance remedies.",
        "United States": "The US follows common law with state-specific variations. Key considerations include at-will employment, strong intellectual property protections, extensive litigation culture, and state-specific contract laws. Arbitration clauses are generally enforceable.",
        "United Kingdom": "The UK follows common law with emphasis on contractual freedom and precedent. Key considerations include unfair contract terms regulations, strong consumer protections, and Brexit-related commercial implications.",
        "Germany": "Germany follows civil law with strict contractual formalities. Key considerations include BGB (Civil Code), strong employee protections, detailed consumer rights, and preference for written agreements with specific terms.",
        "France": "France follows civil law with emphasis on good faith and contractual balance. Key considerations include Code Civil, strong consumer protections, mandatory employee benefits, and strict formality requirements.",
        "Canada": "Canada follows common law (except Quebec - civil law) with federal and provincial jurisdiction variations. Key considerations include strong consumer protections, bilingual requirements in some provinces, and specific employment standards.",
        "Australia": "Australia follows common law with consumer protection emphasis. Key considerations include Australian Consumer Law, unfair contract terms provisions, and state-specific variations in commercial law.",
        "Japan": "Japan follows civil law with emphasis on consensus and long-term relationships. Key considerations include detailed written agreements, specific performance preferences, and cultural importance of maintaining business relationships.",
        "Singapore": "Singapore follows common law with efficient dispute resolution. Key considerations include strong contract enforcement, international arbitration preferences, and business-friendly regulations.",
        "UAE": "UAE follows civil law with Islamic law influences. Key considerations include Sharia-compliant terms, specific requirements for foreign businesses, and emphasis on local jurisdiction clauses.",
        "China": "China follows civil law with socialist characteristics. Key considerations include specific foreign investment regulations, emphasis on local dispute resolution, and government approval requirements for certain contracts.",
        "Brazil": "Brazil follows civil law with detailed consumer protections. Key considerations include strong employee rights, specific tax implications, and preference for local jurisdiction and Portuguese language terms."
    }
    
    # If country not in predefined list, provide general guidance
    if country not in country_contexts:
        return f"Analyzing under {country} law. Consider local contract law principles, consumer protection regulations, employment standards, and jurisdiction-specific requirements. When in doubt, recommend consulting local legal counsel for {country}-specific advice."
    
    return country_contexts[country]
    """Get role-specific context for the AI prompt"""
    role_contexts = {
        "student": "You are analyzing this for a law student or academic researcher. Focus on educational value, learning opportunities, and understanding legal concepts. Highlight both good and problematic clauses for learning purposes.",
        "freelancer": "You are analyzing this for an independent contractor or freelancer. Focus on payment terms, intellectual property rights, liability clauses, and termination conditions that could impact their business.",
        "client": "You are analyzing this for a business or company hiring services. Focus on deliverables, timeline commitments, quality standards, and protecting the client's interests.",
        "startup": "You are analyzing this for a startup or small business. Focus on scalability, cash flow impact, liability exposure, and terms that could affect business growth and operations.",
        "vendor": "You are analyzing this for a supplier or vendor providing goods/services. Focus on payment terms, delivery requirements, liability limitations, and terms that protect the vendor's interests.",
        "legal": "You are analyzing this for a legal professional. Provide detailed analysis of legal implications, precedent considerations, enforceability issues, and professional liability concerns."
    }
    return role_contexts.get(user_role, role_contexts["freelancer"])

@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        version="1.0.0"
    )

@app.post("/analyze")
async def analyze_contract(request: ContractAnalysisRequest):
    """
    Analyze a legal contract and return structured insights
    """
    try:
        # Input validation
        if len(request.contract_text.strip()) < 50:
            raise HTTPException(
                status_code=400, 
                detail="Contract text too short. Please provide a complete contract."
            )
        
        # Log analysis request (without sensitive data)
        logger.info(f"Contract analysis requested for role: {request.user_role} in country: {request.country}")
        
        # Prepare the prompt with role-specific and country-specific context
        role_context = get_role_context(request.user_role)
        country_context = get_country_context(request.country)
        prompt = MASTER_PROMPT_TEMPLATE.format(
            user_role=request.user_role,
            country=request.country,
            role_context=role_context,
            country_context=country_context,
            contract_text=request.contract_text.strip()
        )
        
        # Call Gemini AI
        if model is None:
            # Return mock data for development
            logger.warning("Gemini AI not available, returning mock data")
            return get_mock_analysis_response()
        
        try:
            response = model.generate_content(prompt)
            
            if not response.text:
                raise HTTPException(
                    status_code=500,
                    detail="AI service returned empty response"
                )
            
            # Parse and validate JSON response
            import json
            try:
                analysis_result = json.loads(response.text.strip())
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON from AI: {e}")
                # Try to extract JSON from response if wrapped in markdown
                text = response.text.strip()
                if text.startswith("```json"):
                    text = text[7:]
                if text.endswith("```"):
                    text = text[:-3]
                analysis_result = json.loads(text.strip())
            
            # Validate required fields
            required_fields = ["legal_safety_index", "clauses"]
            for field in required_fields:
                if field not in analysis_result:
                    raise ValueError(f"Missing required field: {field}")
            
            logger.info("Contract analysis completed successfully")
            return analysis_result
            
        except Exception as ai_error:
            logger.error(f"AI analysis failed: {ai_error}")
            raise HTTPException(
                status_code=500,
                detail="Contract analysis failed. Please try again."
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in contract analysis: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error occurred"
        )

def get_mock_analysis_response():
    """Mock response for development/testing"""
    return {
        "legal_safety_index": {
            "score": 45,
            "justification": "Several high-risk clauses including unlimited liability and unclear payment terms. Immediate legal review recommended."
        },
        "clauses": [
            {
                "clause_title": "1. Payment Terms",
                "original_text": "Payment shall be made within 90 days of invoice date. Late fees of 5% per month apply.",
                "risk_level": "High",
                "impact_first_explanation": "This means you won't get paid for 3 months after completing work, and if they're even later, you'll face hefty penalties that could exceed your project fee.",
                "safer_suggestion": "Payment shall be made within 30 days of invoice date. Late fees of 1.5% per month shall apply to Client for overdue payments."
            },
            {
                "clause_title": "2. Liability Clause",
                "original_text": "Contractor assumes full liability for any damages arising from this agreement.",
                "risk_level": "High", 
                "impact_first_explanation": "This means if anything goes wrong (even beyond your control), you're personally responsible for all costs and damages, which could bankrupt your business.",
                "safer_suggestion": "Contractor's liability shall be limited to the total contract value. Both parties shall maintain appropriate professional indemnity insurance."
            },
            {
                "clause_title": "3. Intellectual Property",
                "original_text": "All work product and intellectual property created shall belong to the Client.",
                "risk_level": "Medium",
                "impact_first_explanation": "This means you can't reuse any code, designs, or methods you create for this project in future work, limiting your ability to build on your expertise.",
                "safer_suggestion": "Client receives exclusive license to use work product. Contractor retains rights to general methodologies and pre-existing intellectual property."
            }
        ]
    }

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred"}
    )

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "backend:app",
        host="0.0.0.0",
        port=port,
        reload=False,  # Disabled reload to prevent crashes during testing
        log_level="info"
    )
