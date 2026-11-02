"""
Evaluation Module for Prompt Engineering Comparison
Implements objective, reproducible evaluation scoring rubrics for:
1. Correctness (ground-truth reference comparison)
2. Relevance (topical alignment and keyword coverage)
3. Explanation Quality (reasoning depth, structure, clarity)
4. Overall Aggregate Performance Score
"""

import re
from typing import Dict, Any, List, Optional
from config import WEIGHTS


def normalize_text(text: str) -> str:
    """Normalizes string for robust comparison by removing punctuation and lowercasing."""
    if not text:
        return ""
    # Strip markdown markers, symbols, punctuation
    cleaned = re.sub(r"[^\w\s\.]", " ", text.lower())
    return " ".join(cleaned.split())


def evaluate_correctness(
    response_text: str,
    reference_answer: Optional[str] = None,
    keywords: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Evaluates factual correctness against known reference answer.
    Does not guess or hallucinate scores. Uses structured matching:
    - Exact/Normalized substring match
    - Number/Entity equality
    - Keyword density
    """
    if not response_text or "[API Error" in response_text:
        return {"score": 0.0, "status": "Failed / Error", "details": "Response unavailable or errored."}
    
    # Strip demo header if present for fair content evaluation
    clean_response = response_text.replace("[DEMO MODE - Simulated Output]", "").strip()
    norm_resp = normalize_text(clean_response)
    
    # If no reference answer is provided, evaluate via keyword coverage
    if not reference_answer or reference_answer.strip() == "":
        if keywords:
            matched_kw = [kw for kw in keywords if normalize_text(kw) in norm_resp]
            kw_ratio = len(matched_kw) / max(1, len(keywords))
            score = round(kw_ratio * 100.0, 1)
            status = "High Relevance" if score >= 75 else ("Moderate" if score >= 40 else "Needs Review")
            return {
                "score": score,
                "status": status,
                "details": f"Evaluated via keywords ({len(matched_kw)}/{len(keywords)} found)."
            }
        return {"score": 70.0, "status": "Ungraded", "details": "Open-ended question without reference answer."}
    
    norm_ref = normalize_text(reference_answer)
    ref_terms = [t for t in norm_ref.split() if len(t) > 2]
    
    # 1. Exact or normalized containment
    if norm_ref in norm_resp:
        return {"score": 100.0, "status": "Correct", "details": "Direct match with ground truth reference answer."}
    
    # 2. Check for boolean/single-word critical answers (e.g. "No", "Daughter", "Canberra")
    if len(reference_answer.strip().split()) <= 2:
        # Check word boundary regex
        ref_word = re.escape(reference_answer.strip().lower())
        if re.search(rf"\b{ref_word}\b", norm_resp):
            return {"score": 100.0, "status": "Correct", "details": "Exact core answer entity detected."}
    
    # 3. Check for numerical values (e.g., 80 km/h, 10, 120, 154, 1945)
    numbers_in_ref = re.findall(r"\b\d+(?:\.\d+)?\b", reference_answer)
    numbers_in_resp = re.findall(r"\b\d+(?:\.\d+)?\b", clean_response)
    
    if numbers_in_ref:
        matched_numbers = [n for n in numbers_in_ref if n in numbers_in_resp]
        if len(matched_numbers) == len(numbers_in_ref):
            return {"score": 100.0, "status": "Correct", "details": f"All critical numerical values matched: {matched_numbers}"}
        elif len(matched_numbers) > 0:
            return {"score": 60.0, "status": "Partially Correct", "details": f"Partial numerical match: {matched_numbers}"}
    
    # 4. Keyword and key phrase overlap
    if keywords:
        matched_kw = [kw for kw in keywords if normalize_text(kw) in norm_resp]
        kw_ratio = len(matched_kw) / max(1, len(keywords))
        if kw_ratio >= 0.7:
            return {"score": 85.0, "status": "Mostly Correct", "details": f"Strong conceptual match ({len(matched_kw)}/{len(keywords)} keywords)."}
        elif kw_ratio >= 0.4:
            return {"score": 55.0, "status": "Partially Correct", "details": f"Moderate conceptual match ({len(matched_kw)}/{len(keywords)} keywords)."}
    
    # 5. Overlap ratio of reference words
    if ref_terms:
        matches = [t for t in ref_terms if t in norm_resp]
        overlap_ratio = len(matches) / len(ref_terms)
        if overlap_ratio >= 0.75:
            return {"score": 80.0, "status": "Mostly Correct", "details": f"High lexical overlap ({int(overlap_ratio*100)}%)."}
        elif overlap_ratio >= 0.4:
            return {"score": 50.0, "status": "Partially Correct", "details": f"Moderate lexical overlap ({int(overlap_ratio*100)}%)."}
            
    return {"score": 15.0, "status": "Incorrect", "details": "Reference answer concepts not detected in response."}


def evaluate_relevance(response_text: str, question: str, keywords: Optional[List[str]] = None) -> float:
    """
    Evaluates response relevance by measuring query alignment and lack of off-topic drift.
    Range: 0.0 - 100.0
    """
    if not response_text or "[API Error" in response_text:
        return 0.0
        
    norm_resp = normalize_text(response_text)
    norm_q = normalize_text(question)
    
    # Stopwords to filter out
    stopwords = {"what", "is", "the", "of", "in", "to", "a", "an", "and", "how", "why", "who", "which", "are"}
    q_tokens = [w for w in norm_q.split() if w not in stopwords and len(w) > 2]
    
    if not q_tokens:
        return 75.0
        
    q_matches = sum(1 for w in q_tokens if w in norm_resp)
    q_coverage = q_matches / len(q_tokens)
    
    kw_coverage = 0.5
    if keywords:
        kw_matches = sum(1 for kw in keywords if normalize_text(kw) in norm_resp)
        kw_coverage = kw_matches / max(1, len(keywords))
        
    # Combined relevance
    raw_score = (0.5 * q_coverage + 0.5 * kw_coverage) * 100.0
    # Floor to reasonable baseline if response is substantive
    score = min(100.0, max(25.0, round(raw_score, 1)))
    return score


def evaluate_explanation_quality(response_text: str, technique: str) -> float:
    """
    Evaluates explanation depth, structural reasoning, and presentation clarity.
    Standardized rubric:
    - Base score based on clarity (30 pts)
    - Logical connectives: 'because', 'therefore', 'since', 'step' (+20 pts)
    - Structural markers: lists, numbers, bullet points (+20 pts)
    - Conclusive clarity: 'Final Answer' or clear closing statement (+15 pts)
    - Code block or formal notation if applicable (+15 pts)
    """
    if not response_text or "[API Error" in response_text:
        return 0.0
        
    clean_text = response_text.replace("[DEMO MODE - Simulated Output]", "").strip()
    words = clean_text.split()
    word_count = len(words)
    
    score = 30.0  # Base score for coherent output
    
    # Check for logical connectors
    connectors = ["because", "therefore", "since", "hence", "as a result", "due to", "thus", "steps", "reasoning"]
    found_connectors = sum(1 for c in connectors if c in clean_text.lower())
    score += min(20.0, found_connectors * 5.0)
    
    # Structural hierarchy (numbered lists, steps, bullets)
    has_numbered_steps = bool(re.search(r"(?:^|\n)\s*(?:\d+[\.\)]|step\s*\d+:)", clean_text, re.IGNORECASE))
    has_bullets = bool(re.search(r"(?:^|\n)\s*[-*•]", clean_text))
    if has_numbered_steps:
        score += 20.0
    elif has_bullets:
        score += 12.0
        
    # Conclusive clarity ('Final Answer', 'Conclusion', bolded summary)
    if "final answer" in clean_text.lower() or "in conclusion" in clean_text.lower():
        score += 15.0
    elif word_count > 10 and clean_text.endswith("."):
        score += 8.0
        
    # Code formatting check
    if "```" in clean_text or "`" in clean_text:
        score += 15.0
        
    # Length sanity adjustments
    if word_count < 4 and technique != "Zero-Shot":
        score -= 20.0  # Overly brief for CoT or Few-Shot
    elif word_count > 400:
        score -= 10.0  # Overly verbose / potential rambling
        
    return min(100.0, max(10.0, round(score, 1)))


def evaluate_technique_response(
    technique: str,
    response_dict: Dict[str, Any],
    question: str,
    reference_answer: Optional[str] = None,
    keywords: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Aggregates all evaluation dimensions into a coherent report for a single technique.
    """
    text = response_dict.get("response", "")
    
    corr_result = evaluate_correctness(text, reference_answer, keywords)
    correctness = corr_result["score"]
    status = corr_result["status"]
    
    relevance = evaluate_relevance(text, question, keywords)
    explanation = evaluate_explanation_quality(text, technique)
    
    # Calculate weighted composite overall score
    overall = round(
        (correctness * WEIGHTS["correctness"]) +
        (relevance * WEIGHTS["relevance"]) +
        (explanation * WEIGHTS["explanation"]),
        1
    )
    
    return {
        "technique": technique,
        "correctness": correctness,
        "status": status,
        "correctness_details": corr_result["details"],
        "relevance": relevance,
        "explanation": explanation,
        "overall_score": overall,
        "latency": response_dict.get("latency", 0.0),
        "prompt_tokens": response_dict.get("prompt_tokens", 0),
        "completion_tokens": response_dict.get("completion_tokens", 0),
        "total_tokens": response_dict.get("total_tokens", 0),
        "is_demo": response_dict.get("is_demo", False),
        "error": response_dict.get("error", None)
    }
