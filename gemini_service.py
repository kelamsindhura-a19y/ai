"""
Gemini Service Module
Handles communication with Google Gemini API using the official google-genai SDK.
Features secure credential management, rate-limit resilience, and educational demo fallback.
"""

import os
import time
from typing import Dict, Any, Optional

try:
    from google import genai
    from google.genai import errors as genai_errors
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    genai_errors = None

from prompts import (
    build_zero_shot_prompt,
    build_few_shot_prompt,
    build_chain_of_thought_prompt
)
from config import DEFAULT_MODEL


def resolve_api_key(explicit_key: Optional[str] = None) -> Optional[str]:
    """
    Resolves the Gemini API key securely from:
    1. Explicit key passed from UI input
    2. Environment variable 'GEMINI_API_KEY'
    3. Streamlit secrets (if available in runtime)
    """
    if explicit_key and explicit_key.strip():
        return explicit_key.strip()
    
    env_key = os.getenv("GEMINI_API_KEY")
    if env_key and env_key.strip() and env_key.strip() != "your_gemini_api_key_here":
        return env_key.strip()
    
    # Try streamlit secrets safely
    try:
        import streamlit as st
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    
    return None


def call_gemini(
    prompt: str,
    api_key: Optional[str] = None,
    model_name: str = DEFAULT_MODEL,
    temperature: float = 0.2
) -> Dict[str, Any]:
    """
    Sends a single prompt to Google Gemini API.
    Measures real execution time and token consumption.
    Returns dictionary with text, latency, token counts, and error details.
    """
    resolved_key = resolve_api_key(api_key)
    
    # If no key is configured or SDK missing, return demo simulation
    if not resolved_key or not GENAI_AVAILABLE:
        return _generate_demo_response(prompt, is_offline_fallback=True)
    
    start_time = time.perf_counter()
    try:
        client = genai.Client(api_key=resolved_key)
        
        # Generation configuration
        config = {
            "temperature": temperature,
            "max_output_tokens": 1024,
        }
        
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=config
        )
        latency = round(time.perf_counter() - start_time, 3)
        
        # Extract token usage metadata if available
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            meta = response.usage_metadata
            prompt_tokens = getattr(meta, "prompt_token_count", 0) or 0
            completion_tokens = getattr(meta, "candidates_token_count", 0) or 0
            total_tokens = getattr(meta, "total_token_count", 0) or (prompt_tokens + completion_tokens)
        else:
            # Approximate heuristic (1 token ~ 4 chars) if metadata is omitted
            prompt_tokens = max(1, len(prompt) // 4)
            completion_tokens = max(1, len(response.text or "") // 4)
            total_tokens = prompt_tokens + completion_tokens

        return {
            "success": True,
            "response": response.text if response.text else "No response generated.",
            "latency": latency,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "is_demo": False,
            "error": None
        }

    except Exception as e:
        latency = round(time.perf_counter() - start_time, 3)
        err_msg = str(e)
        
        friendly_error = _format_error_message(err_msg)
        return {
            "success": False,
            "response": f"[API Error Encountered: {friendly_error}]",
            "latency": latency,
            "prompt_tokens": max(1, len(prompt) // 4),
            "completion_tokens": 0,
            "total_tokens": max(1, len(prompt) // 4),
            "is_demo": False,
            "error": friendly_error
        }


def _format_error_message(raw_error: str) -> str:
    """Translates raw API errors into actionable, student-friendly explanations."""
    err_lower = raw_error.lower()
    if "api_key_invalid" in err_lower or "invalid api key" in err_lower or "400" in err_lower:
        return "Invalid Gemini API Key. Please verify your API key in the sidebar or .env file."
    elif "429" in err_lower or "quota" in err_lower or "resource_exhausted" in err_lower:
        return "API Quota / Rate Limit reached (HTTP 429). Please wait a moment or check your free tier limits."
    elif "connection" in err_lower or "timeout" in err_lower or "getaddrinfo" in err_lower:
        return "Network connection issue. Please check your internet connection or proxy settings."
    return f"API Exception: {raw_error[:150]}"


def _generate_demo_response(prompt: str, is_offline_fallback: bool = True) -> Dict[str, Any]:
    """
    Generates educational, simulated responses when no API key is supplied.
    Clearly marked with demo banners to maintain academic integrity.
    """
    time.sleep(0.35)  # Simulate network latency
    
    # Identify prompt type from structure
    is_cot = "step-by-step" in prompt.lower() or "reasoning:" in prompt.lower()
    is_few_shot = "example 1:" in prompt.lower()
    
    # Infer question topic
    lower_prompt = prompt.lower()
    
    if "australia" in lower_prompt:
        if is_cot:
            reply = (
                "1. Given question: What is the capital city of Australia?\n"
                "2. Sydney and Melbourne are often mistaken as the capital due to their size.\n"
                "3. In 1908, Canberra was selected as the compromise capital city between Sydney and Melbourne.\n"
                "Final Answer: Canberra"
            )
            lat, tok = 0.85, 95
        elif is_few_shot:
            reply = "The capital city of Australia is Canberra."
            lat, tok = 0.52, 68
        else:
            reply = "Canberra."
            lat, tok = 0.38, 25
    elif "train" in lower_prompt or "speed" in lower_prompt or "240" in lower_prompt:
        if is_cot:
            reply = (
                "1. Formula for average speed: Speed = Distance / Time.\n"
                "2. Distance = 240 kilometers; Time = 3 hours.\n"
                "3. Calculation: 240 / 3 = 80.\n"
                "Final Answer: 80 km/h"
            )
            lat, tok = 0.92, 110
        elif is_few_shot:
            reply = "Speed = 240 km / 3 h = 80 km/h."
            lat, tok = 0.55, 75
        else:
            reply = "80 km/h"
            lat, tok = 0.40, 30
    elif "roses" in lower_prompt or "logic" in lower_prompt or "flower" in lower_prompt:
        if is_cot:
            reply = (
                "1. Premise 1: All roses are flowers (Roses ⊆ Flowers).\n"
                "2. Premise 2: Some flowers fade quickly (There exists x in Flowers that fades quickly).\n"
                "3. Logical deduction: The subset of flowers that fade quickly may or may not overlap with roses.\n"
                "4. Therefore, it is an invalid categorical syllogism to claim all roses fade quickly.\n"
                "Final Answer: No"
            )
            lat, tok = 1.05, 128
        elif is_few_shot:
            reply = "No, we cannot conclude that all roses fade quickly because only some flowers do."
            lat, tok = 0.62, 82
        else:
            reply = "No."
            lat, tok = 0.41, 28
    elif "powerhouse" in lower_prompt or "cell" in lower_prompt:
        if is_cot:
            reply = (
                "1. Cellular biology question: identify the organelle that produces cellular ATP.\n"
                "2. Mitochondria perform cellular respiration, generating chemical energy in the form of ATP.\n"
                "3. Hence, they are traditionally designated the 'powerhouse of the cell'.\n"
                "Final Answer: Mitochondria"
            )
            lat, tok = 0.88, 102
        elif is_few_shot:
            reply = "The powerhouse of the cell is the Mitochondria."
            lat, tok = 0.50, 70
        else:
            reply = "Mitochondria."
            lat, tok = 0.35, 24
    elif "list" in lower_prompt or "len" in lower_prompt or "python" in lower_prompt:
        if is_cot:
            reply = (
                "1. We need the built-in Python function to count elements in a sequence.\n"
                "2. In Python, the standard built-in function is len().\n"
                "3. Example: len([1, 2, 3]) evaluates to 3.\n"
                "Final Answer: len()"
            )
            lat, tok = 0.82, 92
        elif is_few_shot:
            reply = "In Python, the built-in function is `len()`."
            lat, tok = 0.48, 64
        else:
            reply = "len()"
            lat, tok = 0.34, 22
    else:
        # Generic demo response
        if is_cot:
            reply = (
                "1. Problem Analysis: Deconstructing the target question and core attributes.\n"
                "2. Step-by-Step Logic: Evaluating the underlying domain principles.\n"
                "3. Synthesis: Combining deductive steps into a cohesive deduction.\n"
                "Final Answer: Synthesized solution based on step-by-step reasoning."
            )
            lat, tok = 0.95, 105
        elif is_few_shot:
            reply = "Following the provided exemplar format, here is the direct, structured answer."
            lat, tok = 0.58, 72
        else:
            reply = "Concise direct answer to the input query."
            lat, tok = 0.39, 29

    return {
        "success": True,
        "response": f"[DEMO MODE - Simulated Output]\n{reply}",
        "latency": lat,
        "prompt_tokens": max(1, len(prompt) // 4),
        "completion_tokens": tok,
        "total_tokens": (max(1, len(prompt) // 4)) + tok,
        "is_demo": True,
        "error": None
    }


def execute_all_strategies(
    question: str,
    category: str,
    api_key: Optional[str] = None,
    model_name: str = DEFAULT_MODEL,
    temperature: float = 0.2
) -> Dict[str, Dict[str, Any]]:
    """
    Executes all three prompting strategies (Zero-Shot, Few-Shot, CoT)
    for the exact same user question and category under identical parameters.
    """
    prompts = {
        "Zero-Shot": build_zero_shot_prompt(question),
        "Few-Shot": build_few_shot_prompt(question, category),
        "Chain-of-Thought": build_chain_of_thought_prompt(question),
    }
    
    results = {}
    for technique_name, prompt_text in prompts.items():
        res = call_gemini(
            prompt=prompt_text,
            api_key=api_key,
            model_name=model_name,
            temperature=temperature
        )
        res["prompt_text"] = prompt_text
        results[technique_name] = res
        
    return results
