import os
import json
import requests
from typing import Dict, Any, List

def enrich_with_llm(col_name: str, sample_values: List[Any], stats: Dict[str, Any]) -> Dict[str, Any]:
    """Layer 3: Use LLM to resolve ambiguity for low-confidence columns."""
    
    if os.getenv("ENABLE_LLM_LAYER", "false").lower() != "true":
        return {}

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return {}

    prompt = f"""
    Analyze this tabular data column:
    Column Name: {col_name}
    Sample Values: {sample_values}
    Stats: {json.dumps(stats)}

    Return ONLY a valid JSON with:
    "role": one of ["target", "id", "numeric", "categorical", "text", "datetime", "drop"],
    "transform": one of ["log", "standard", "ohe", "ordinal", "tfidf", "domain_extract", "cyclical", "passthrough", "drop"],
    "confidence": float (0-1)
    """

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [{"role": "user", "content": prompt}],
                "response_format": {"type": "json_object"}
            },
            timeout=5
        )
        data = response.json()
        llm_res = json.loads(data['choices'][0]['message']['content'])
        llm_res["layer_source"] = 3
        return llm_res
    except Exception as e:
        print(f"LLM Layer Error: {e}")
        return {}
