import json
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field, ValidationError

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_ENABLED = os.getenv("GEMINI_ENABLED", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


class CureAdvice(BaseModel):
    summary: str = Field(min_length=1)
    treatment_options: list[str] = Field(min_length=1, max_length=8)
    prevention_steps: list[str] = Field(min_length=1, max_length=8)


def get_cure_advice(disease: str) -> CureAdvice:
    if not GEMINI_ENABLED:
        raise RuntimeError("Gemini cure guidance is disabled.")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_gemini_api_key_here":
        raise RuntimeError("GEMINI_API_KEY is not configured. Add it to the .env file.")

    prompt = f"""You are a careful agricultural plant-disease advisor.
Provide practical, general guidance for the plant disease below.
Disease: {disease}

Return ONLY valid JSON with exactly these keys:
- summary: one concise explanation of the disease and urgency
- treatment_options: an array of 2 to 5 actionable treatment options
- prevention_steps: an array of 2 to 5 actionable prevention steps

Use plain language. Mention that product labels and local agricultural guidance must be followed.
Do not claim certainty from the disease name alone, and do not recommend unsafe chemical mixtures.
"""

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.2,
            ),
        )
        return CureAdvice.model_validate(json.loads(response.text))
    except (json.JSONDecodeError, ValidationError, TypeError) as error:
        raise RuntimeError("Gemini returned an invalid cure response.") from error
    except Exception as error:
        raise RuntimeError("Unable to generate cure guidance right now.") from error
