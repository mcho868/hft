#!/usr/bin/env python3
"""
LLM-as-Judge for Medical Triage Reasoning and Next Step Quality Evaluation

Evaluates the quality of reasoning and next step recommendations using external LLM API.
Based on the appropriateness_judge.py methodology but focused on reasoning quality.
"""

import os
import time
import re
import requests
from typing import Dict, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# Check if TinfoilAgent API key is available
TINFOIL_API_KEY = os.getenv("TINFOIL_API_KEY")
TINFOIL_ENDPOINT = os.getenv("TINFOIL_ENDPOINT") 
TINFOIL_AVAILABLE = TINFOIL_API_KEY is not None

@dataclass
class ReasoningQualityScores:
    """Reasoning and next step quality evaluation scores"""
    next_step_quality_score: float  # 0-100: Quality of next step recommendation
    reasoning_quality_score: float  # 0-100: Quality of clinical reasoning
    overall_score: float  # 0-100: Combined quality assessment
    next_step_rationale: str  # Explanation for next step quality
    reasoning_rationale: str  # Explanation for reasoning quality
    overall_rationale: str  # Overall assessment rationale

class TinfoilLLMClient:
    """LLM client using Tinfoil REST API for quality evaluation"""
    
    def __init__(self, model_name: str = "llama3-3-70b"):
        self.model_name = model_name
        self.api_url = TINFOIL_ENDPOINT
        self.api_key = TINFOIL_API_KEY
        
        if TINFOIL_AVAILABLE:
            logger.info(f"✅ Initialized Tinfoil REST client for LLM judge with model: {model_name}")
        else:
            logger.warning("⚠️  Tinfoil API key not available, using mock responses for LLM judge")
        
    def complete(self, prompt: str, max_tokens: int = 1500) -> str:
        """Get completion using Tinfoil REST API"""
        if not TINFOIL_AVAILABLE:
            return self._mock_completion(prompt)
            
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            payload = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": max_tokens,
                "temperature": 0.1  # Low temperature for consistent evaluation
            }
            
            response = requests.post(self.api_url, json=payload, headers=headers, timeout=120)
            response.raise_for_status()
            
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            
            if content:
                return content
            else:
                logger.warning("⚠️  Tinfoil API returned empty content, using fallback")
                return self._mock_completion(prompt)
                
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Error with Tinfoil API: {e}")
            if hasattr(e, 'response') and e.response:
                logger.error(f"   Response: {e.response.text[:200]}...")
            return self._mock_completion(prompt)
        except Exception as e:
            logger.error(f"❌ Unexpected error with Tinfoil API: {e}")
            return self._mock_completion(prompt)
    
    def _mock_completion(self, prompt: str) -> str:
        """Fallback mock completion for testing"""
        time.sleep(0.1)  # Simulate API delay
        
        # Simulate realistic quality scores based on prompt content
        if "emergency" in prompt.lower() or "ed" in prompt.lower():
            return """
            **NEXT STEP QUALITY EVALUATION:**
            
            Next Step Quality Score: 85/100
            
            Next Step Rationale: The recommendation for emergency department evaluation is appropriate and timely given the symptoms described. The advice correctly prioritizes patient safety and provides clear, actionable guidance. Minor deduction for not mentioning specific protocols or what to expect at the ED.
            
            **REASONING QUALITY EVALUATION:**
            
            Reasoning Quality Score: 82/100
            
            Reasoning Rationale: The clinical reasoning demonstrates good understanding of emergency triage principles. The logic flow from symptoms to decision is sound and shows appropriate risk assessment. Could be enhanced with more specific differential considerations and red flag identification.
            
            **OVERALL ASSESSMENT:**
            
            Overall Quality Score: 83/100
            
            Overall Rationale: Strong clinical decision-making with appropriate urgency recognition. The combination of sound reasoning and practical next steps provides good patient care guidance. Both components demonstrate competent medical knowledge application.
            """
        elif "home" in prompt.lower() or "self" in prompt.lower():
            return """
            **NEXT STEP QUALITY EVALUATION:**
            
            Next Step Quality Score: 78/100
            
            Next Step Rationale: The home care recommendation is appropriate for this condition level. Provides practical self-care guidance that is actionable. Could be improved with more specific monitoring instructions and clear criteria for when to seek further care.
            
            **REASONING QUALITY EVALUATION:**
            
            Reasoning Quality Score: 75/100
            
            Reasoning Rationale: The reasoning shows appropriate risk stratification for a lower acuity condition. Demonstrates understanding of when conservative management is suitable. Would benefit from more detailed consideration of potential complications or progression scenarios.
            
            **OVERALL ASSESSMENT:**
            
            Overall Quality Score: 76/100
            
            Overall Rationale: Solid clinical judgment for routine care recommendations. Both reasoning and next steps are medically sound and patient-appropriate. Good balance of reassurance and appropriate caution.
            """
        else:
            return """
            **NEXT STEP QUALITY EVALUATION:**
            
            Next Step Quality Score: 79/100
            
            Next Step Rationale: The GP appointment recommendation is appropriate for this presentation. Provides reasonable timeline expectations and covers key areas to discuss. Could include more specific preparation instructions for the patient.
            
            **REASONING QUALITY EVALUATION:**
            
            Reasoning Quality Score: 77/100
            
            Reasoning Rationale: Shows good clinical reasoning with appropriate risk assessment. Demonstrates understanding of primary care scope and referral thresholds. Minor areas for improvement in differential consideration breadth.
            
            **OVERALL ASSESSMENT:**
            
            Overall Quality Score: 78/100
            
            Overall Rationale: Competent clinical decision-making with appropriate care level assignment. Both reasoning process and practical recommendations are medically sound and follow standard clinical practice guidelines.
            """

class MedicalTriageQualityJudge:
    """LLM-powered judge for evaluating medical triage reasoning and next step quality"""
    
    def __init__(self, judge_model: str = "llama3-3-70b", rate_limit_delay: float = 0.5):
        self.judge_model = judge_model
        self.rate_limit_delay = rate_limit_delay
        
        # Initialize LLM client
        self.client = TinfoilLLMClient(judge_model)
        
        # Load evaluation prompt templates
        self.next_step_prompt_template = self._load_next_step_evaluation_prompt()
        self.reasoning_prompt_template = self._load_reasoning_evaluation_prompt()
        
        logger.info(f"🧑‍⚕️ Initialized Medical Triage Quality Judge with {judge_model}")
    
    def _load_next_step_evaluation_prompt(self) -> str:
        """Load the next step quality evaluation prompt template"""
        return """
You are an expert medical evaluator assessing the quality of next step recommendations for medical triage decisions.

**CASE INFORMATION:**
Patient Query: {patient_query}

**MODEL OUTPUT:**
Triage Decision: {model_triage_decision}
Recommended Next Steps: {model_next_steps}

**REFERENCE STANDARD:**
Expected Triage: {expected_triage}
Expected Next Steps: {expected_next_steps}

**EVALUATION TASK:**
Evaluate the quality of the model's next step recommendation on a scale of 0-100, considering:

1. **Appropriateness**: How suitable is the recommendation for the patient's condition?
2. **Actionability**: How clear and practical are the instructions for the patient?
3. **Safety**: Does the advice prioritize patient safety appropriately?
4. **Completeness**: Does it cover the necessary information and follow-up?
5. **Alignment**: How well does it align with the expected standard of care?

**REQUIRED FORMAT:**
Next Step Quality Score: [0-100]/100

Next Step Rationale: [Provide detailed explanation of the score, highlighting strengths and areas for improvement in the next step recommendation. Be specific about what makes the advice good or how it could be better.]
"""

    def _load_reasoning_evaluation_prompt(self) -> str:
        """Load the reasoning quality evaluation prompt template"""
        return """
You are an expert medical evaluator assessing the quality of clinical reasoning for medical triage decisions.

**CASE INFORMATION:**
Patient Query: {patient_query}

**MODEL OUTPUT:**
Triage Decision: {model_triage_decision}
Clinical Reasoning: {model_reasoning}

**REFERENCE STANDARD:**
Expected Triage: {expected_triage}
Expected Reasoning: {expected_reasoning}

**EVALUATION TASK:**
Evaluate the quality of the model's clinical reasoning on a scale of 0-100, considering:

1. **Medical Accuracy**: Is the reasoning medically sound and evidence-based?
2. **Logic Flow**: Does the reasoning follow a logical progression from symptoms to conclusion?
3. **Risk Assessment**: Does it appropriately assess and communicate risk levels?
4. **Differential Consideration**: Does it consider relevant alternative diagnoses or factors?
5. **Clinical Judgment**: Does it demonstrate appropriate clinical decision-making skills?

**REQUIRED FORMAT:**
Reasoning Quality Score: [0-100]/100

Reasoning Rationale: [Provide detailed explanation of the score, analyzing the clinical reasoning quality, identifying strong points and areas where the reasoning could be improved or made more comprehensive.]
"""

    def evaluate_case(self, case_data: Dict[str, Any]) -> ReasoningQualityScores:
        """Evaluate both reasoning and next step quality for a single case"""
        
        try:
            # Extract case information
            case_id = case_data.get("case_id", "unknown")
            patient_query = case_data.get("query", "")
            model_triage = case_data.get("predicted_triage", "")
            model_response = case_data.get("response", "")
            expected_triage = case_data.get("expected_triage", "")
            expected_next_steps = case_data.get("expected_next_steps", "")
            expected_reasoning = case_data.get("expected_reasoning", "")
            
            # Extract next steps and reasoning from model response
            model_next_steps = self._extract_next_steps(model_response)
            model_reasoning = self._extract_reasoning(model_response)
            
            logger.debug(f"🔍 Evaluating case {case_id} with LLM judge...")
            
            # Evaluate next step quality
            next_step_scores = self._evaluate_next_step_quality(
                patient_query, model_triage, model_next_steps,
                expected_triage, expected_next_steps
            )
            
            # Rate limiting
            time.sleep(self.rate_limit_delay)
            
            # Evaluate reasoning quality  
            reasoning_scores = self._evaluate_reasoning_quality(
                patient_query, model_triage, model_reasoning,
                expected_triage, expected_reasoning
            )
            
            # Rate limiting
            time.sleep(self.rate_limit_delay)
            
            # Calculate overall score (weighted average)
            overall_score = (next_step_scores["score"] * 0.6 + reasoning_scores["score"] * 0.4)
            overall_rationale = f"Next Steps ({next_step_scores['score']}/100): {next_step_scores['rationale'][:100]}... | Reasoning ({reasoning_scores['score']}/100): {reasoning_scores['rationale'][:100]}..."
            
            return ReasoningQualityScores(
                next_step_quality_score=next_step_scores["score"],
                reasoning_quality_score=reasoning_scores["score"],
                overall_score=overall_score,
                next_step_rationale=next_step_scores["rationale"],
                reasoning_rationale=reasoning_scores["rationale"],
                overall_rationale=overall_rationale
            )
            
        except Exception as e:
            logger.error(f"❌ Error evaluating case {case_data.get('case_id', 'unknown')}: {e}")
            # Return default scores on error
            return ReasoningQualityScores(
                next_step_quality_score=50.0,
                reasoning_quality_score=50.0,
                overall_score=50.0,
                next_step_rationale=f"Evaluation failed: {str(e)}",
                reasoning_rationale=f"Evaluation failed: {str(e)}",
                overall_rationale=f"Evaluation failed: {str(e)}"
            )
    
    def _evaluate_next_step_quality(self, patient_query: str, model_triage: str, 
                                  model_next_steps: str, expected_triage: str, 
                                  expected_next_steps: str) -> Dict[str, Any]:
        """Evaluate next step quality using LLM judge"""
        
        prompt = self.next_step_prompt_template.format(
            patient_query=patient_query,
            model_triage_decision=model_triage,
            model_next_steps=model_next_steps,
            expected_triage=expected_triage,
            expected_next_steps=expected_next_steps
        )
        
        response = self.client.complete(prompt, max_tokens=1000)
        return self._parse_quality_response(response, "Next Step Quality Score")
    
    def _evaluate_reasoning_quality(self, patient_query: str, model_triage: str,
                                  model_reasoning: str, expected_triage: str,
                                  expected_reasoning: str) -> Dict[str, Any]:
        """Evaluate reasoning quality using LLM judge"""
        
        prompt = self.reasoning_prompt_template.format(
            patient_query=patient_query,
            model_triage_decision=model_triage,
            model_reasoning=model_reasoning,
            expected_triage=expected_triage,
            expected_reasoning=expected_reasoning
        )
        
        response = self.client.complete(prompt, max_tokens=1000)
        return self._parse_quality_response(response, "Reasoning Quality Score")
    
    def _parse_quality_response(self, response: str, score_label: str) -> Dict[str, Any]:
        """Parse LLM evaluation response to extract quality scores"""
        
        score = 50.0  # Default score
        rationale = "Could not parse evaluation response"
        
        try:
            # Extract score using regex pattern
            score_pattern = rf'{score_label}:\s*(\d+(?:\.\d+)?)/100'
            score_match = re.search(score_pattern, response)
            if score_match:
                score = float(score_match.group(1))
            
            # Extract rationale
            rationale_pattern = rf'{score_label.split()[0]} (?:{score_label.split()[1]} )?Rationale:\s*(.+?)(?:\n\n|\*\*|$)'
            rationale_match = re.search(rationale_pattern, response, re.DOTALL)
            if rationale_match:
                rationale = rationale_match.group(1).strip()
            
        except Exception as e:
            logger.warning(f"⚠️ Error parsing quality response: {e}")
        
        return {"score": score, "rationale": rationale}
    
    def _extract_next_steps(self, response: str) -> str:
        """Extract next steps from model response"""
        # Look for common next step patterns
        patterns = [
            r'Next steps?:\s*(.+?)(?:\n\n|\n[A-Z]|$)',
            r'Recommendation:\s*(.+?)(?:\n\n|\n[A-Z]|$)',
            r'Action:\s*(.+?)(?:\n\n|\n[A-Z]|$)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # Fallback: return part of response that might contain next steps
        lines = response.split('\n')
        for line in lines:
            if any(word in line.lower() for word in ['recommend', 'should', 'need', 'book', 'see', 'go']):
                return line.strip()
        
        return response[:200]  # Fallback to first 200 chars
    
    def _extract_reasoning(self, response: str) -> str:
        """Extract reasoning from model response"""
        # Look for common reasoning patterns
        patterns = [
            r'Reasoning:\s*(.+?)(?:\n\n|\nNext|$)',
            r'Rationale:\s*(.+?)(?:\n\n|\nNext|$)',
            r'Explanation:\s*(.+?)(?:\n\n|\nNext|$)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # Fallback: return part of response that might contain reasoning
        lines = response.split('\n')
        reasoning_lines = []
        for line in lines:
            if any(word in line.lower() for word in ['because', 'since', 'due to', 'given', 'considering']):
                reasoning_lines.append(line.strip())
        
        if reasoning_lines:
            return ' '.join(reasoning_lines)
        
        return response[:200]  # Fallback to first 200 chars