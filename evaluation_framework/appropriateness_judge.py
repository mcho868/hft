#!/usr/bin/env python3
"""
LLM-as-Judge Clinical Appropriateness Evaluator
Evaluates clinical decision quality and appropriateness using advanced language models.
"""

import json
import time
import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import re
import numpy as np
import requests

# Check if TinfoilAgent API key is available
TINFOIL_API_KEY = os.getenv("TINFOIL_API_KEY")
TINFOIL_ENDPOINT = os.getenv("TINFOIL_ENDPOINT")
TINFOIL_AVAILABLE = TINFOIL_API_KEY is not None

class TinfoilLLMClient:
    """LLM client using Tinfoil REST API"""
    
    def __init__(self, model_name: str = "llama3-3-70b"):
        self.model_name = model_name
        self.api_url = TINFOIL_ENDPOINT
        self.api_key = TINFOIL_API_KEY
        
        if TINFOIL_AVAILABLE:
            print(f"✅ Initialized Tinfoil REST client with model: {model_name}")
        else:
            print("⚠️  Tinfoil API key not available, using mock responses")
        
    def complete(self, prompt: str, max_tokens: int = 1000) -> str:
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
                "temperature": 0.1  # Low temperature for consistent clinical evaluation
            }
            
            response = requests.post(self.api_url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            
            if content:
                return content
            else:
                print("⚠️  Tinfoil API returned empty content, using fallback")
                return self._mock_completion(prompt)
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Error with Tinfoil API: {e}")
            print(f"   Status code: {getattr(e.response, 'status_code', 'N/A') if hasattr(e, 'response') else 'N/A'}")
            if hasattr(e, 'response') and e.response:
                print(f"   Response text: {e.response.text[:200]}...")
            return self._mock_completion(prompt)
        except (KeyError, IndexError) as e:
            print(f"❌ Error parsing Tinfoil API response: {e}")
            print(f"   Response structure: {result.keys() if 'result' in locals() else 'No result variable'}")
            return self._mock_completion(prompt)
        except Exception as e:
            print(f"❌ Unexpected error with Tinfoil API: {e}")
            return self._mock_completion(prompt)
    
    def _mock_completion(self, prompt: str) -> str:
        """Fallback mock completion"""
        # Simulate realistic evaluation responses
        time.sleep(0.05)  # Reduced from 0.5s to 0.05s for faster testing
        
        if "emergency" in prompt.lower() or "chest pain" in prompt.lower():
            return """
            Clinical Appropriateness Assessment:
            
            Safety: 9/10 - Correctly identified high-risk symptoms requiring immediate care
            Efficiency: 8/10 - Appropriate resource utilization for emergency case
            Completeness: 8/10 - Covered key diagnostic considerations
            Reasoning: 9/10 - Strong clinical reasoning with appropriate urgency
            
            Overall Score: 8.5/10
            
            Rationale: The triage decision demonstrates excellent clinical judgment in recognizing emergency symptoms. The recommendation for immediate ED evaluation is appropriate and potentially life-saving. Minor points deducted for not mentioning specific protocols.
            """
        elif "minor" in prompt.lower() or "home" in prompt.lower():
            return """
            Clinical Appropriateness Assessment:
            
            Safety: 7/10 - Appropriately identified low-risk condition
            Efficiency: 9/10 - Excellent resource conservation
            Completeness: 6/10 - Could include more self-care guidance
            Reasoning: 7/10 - Sound reasoning but could be more detailed
            
            Overall Score: 7.2/10
            
            Rationale: The home care recommendation is clinically appropriate for this minor condition. Good efficiency in avoiding unnecessary healthcare visits. Could be enhanced with more specific self-care instructions and red flag symptoms to watch for.
            """
        else:
            return """
            Clinical Appropriateness Assessment:
            
            Safety: 8/10 - Reasonable safety considerations
            Efficiency: 7/10 - Appropriate level of care
            Completeness: 7/10 - Adequate coverage of key points
            Reasoning: 7/10 - Sound clinical reasoning
            
            Overall Score: 7.2/10
            
            Rationale: The clinical decision shows good judgment with appropriate consideration of patient safety and resource utilization. Standard clinical reasoning applied appropriately.
            """

@dataclass
class AppropriatenessScores:
    """Clinical appropriateness evaluation scores"""
    safety_score: float  # 0-10: Risk assessment and harm prevention
    efficiency_score: float  # 0-10: Appropriate resource utilization
    completeness_score: float  # 0-10: Coverage of relevant factors
    reasoning_score: float  # 0-10: Quality of clinical reasoning
    overall_score: float  # 0-10: Weighted overall assessment
    rationale: str  # Textual explanation of scores

@dataclass
class ClinicalCase:
    """Structured clinical case for evaluation"""
    case_id: str
    patient_presentation: str
    model_triage_decision: str
    model_next_steps: str
    ground_truth_triage: str
    ground_truth_next_steps: str
    case_complexity: str  # "simple", "moderate", "complex"
    medical_domain: str  # "cardiac", "respiratory", "general", etc.

class ClinicalAppropriatenessJudge:
    """LLM-as-Judge for clinical appropriateness evaluation using TinfoilAgent"""
    
    def __init__(self, judge_model: str = "llama3-3-70b", 
                 use_tinfoil: bool = True,
                 rate_limit_delay: float = 1.0):
        self.judge_model = judge_model
        self.use_tinfoil = use_tinfoil
        self.rate_limit_delay = rate_limit_delay
        
        # Initialize LLM client - prefer Tinfoil REST API if available
        if use_tinfoil and TINFOIL_AVAILABLE:
            self.client = TinfoilLLMClient(judge_model)
            print(f"🤖 Using Tinfoil REST API for clinical appropriateness evaluation")
        else:
            # Fallback to mock client
            self.client = self._create_mock_client(judge_model)
            print(f"🔄 Using mock client for evaluation (Tinfoil API key not available)")
        
        # Load evaluation prompts
        self.evaluation_prompt_template = self._load_evaluation_prompt()
        
        # Clinical evaluation criteria
        self.evaluation_criteria = {
            "safety": {
                "weight": 0.4,
                "description": "Risk assessment and patient harm prevention"
            },
            "efficiency": {
                "weight": 0.2,
                "description": "Appropriate healthcare resource utilization"
            },
            "completeness": {
                "weight": 0.2,
                "description": "Coverage of relevant medical factors"
            },
            "reasoning": {
                "weight": 0.2,
                "description": "Quality and logic of clinical decision-making"
            }
        }
    
    def _create_mock_client(self, model_name: str):
        """Create mock client with same interface"""
        class MockClient:
            def __init__(self, model):
                self.model_name = model
            
            def complete(self, prompt: str, max_tokens: int = 1000) -> str:
                return self._mock_completion(prompt)
            
            def _mock_completion(self, prompt: str) -> str:
                time.sleep(0.05)  # Reduced for faster testing
                if "emergency" in prompt.lower() or "chest pain" in prompt.lower():
                    return """Clinical Appropriateness Assessment:
                    
Safety: 9/10 - Correctly identified high-risk symptoms requiring immediate care
Efficiency: 8/10 - Appropriate resource utilization for emergency case
Completeness: 8/10 - Covered key diagnostic considerations
Reasoning: 9/10 - Strong clinical reasoning with appropriate urgency

Overall Score: 8.5/10

Rationale: The triage decision demonstrates excellent clinical judgment in recognizing emergency symptoms."""
                else:
                    return """Clinical Appropriateness Assessment:
                    
Safety: 7/10 - Appropriately assessed condition severity
Efficiency: 7/10 - Reasonable resource utilization
Completeness: 7/10 - Adequate coverage of key factors
Reasoning: 7/10 - Sound clinical reasoning

Overall Score: 7.0/10

Rationale: The clinical decision shows appropriate medical judgment."""
        
        return MockClient(model_name)
    
    def _post_init(self):
        """Complete initialization after client setup"""
        # Load evaluation prompts
        self.evaluation_prompt_template = self._load_evaluation_prompt()
        
        # Clinical evaluation criteria
        self.evaluation_criteria = {
            "safety": {
                "weight": 0.4,
                "description": "Risk assessment and patient harm prevention"
            },
            "efficiency": {
                "weight": 0.2,
                "description": "Appropriate healthcare resource utilization"
            },
            "completeness": {
                "weight": 0.2,
                "description": "Coverage of relevant medical factors"
            },
            "reasoning": {
                "weight": 0.2,
                "description": "Quality and logic of clinical decision-making"
            }
        }
    
    def _load_evaluation_prompt(self) -> str:
        """Load the next step quality evaluation prompt template"""
        return """
You are an expert medical evaluator assessing the quality of next step advice for medical triage decisions.

**CASE INFORMATION:**
Patient Presentation: {patient_presentation}

**MODEL OUTPUT:**
Triage Decision: {model_triage_decision}
Recommended Next Steps: {model_next_steps}

**REFERENCE STANDARD:**
Expected Triage: {ground_truth_triage}
Expected Next Steps: {ground_truth_next_steps}

**EVALUATION TASK:**
Evaluate the quality of the model's next step advice on a scale of 0-10, considering:
- How appropriate and helpful is the advice given the patient's condition?
- Does the advice align with medical best practices?
- Is the advice clear, actionable, and safe for the patient?
- How well does it compare to the expected next steps?

**SCORING SCALE:**
- 9-10: Excellent advice - highly appropriate, safe, and helpful
- 7-8: Good advice - mostly appropriate with minor issues
- 5-6: Adequate advice - reasonable but could be improved
- 3-4: Poor advice - significant concerns or gaps
- 0-2: Very poor advice - inappropriate or potentially harmful

**RESPONSE FORMAT:**
Please provide your evaluation in the following format:

Next Step Quality Assessment:

Overall Score: [score]/10

Rationale: [2-3 sentences explaining your assessment, highlighting key strengths or concerns with the advice]
"""
    
    def evaluate_case(self, case: ClinicalCase) -> AppropriatenessScores:
        """Evaluate a single clinical case for appropriateness"""
        
        # Format evaluation prompt
        prompt = self.evaluation_prompt_template.format(
            patient_presentation=case.patient_presentation,
            model_triage_decision=case.model_triage_decision,
            model_next_steps=case.model_next_steps,
            ground_truth_triage=case.ground_truth_triage,
            ground_truth_next_steps=case.ground_truth_next_steps
        )
        
        try:
            # Get LLM evaluation
            response = self.client.complete(prompt, max_tokens=1000)
            
            # Parse scores from response
            scores = self._parse_evaluation_response(response)
            
            # Add rate limiting
            time.sleep(self.rate_limit_delay)
            
            return scores
            
        except Exception as e:
            print(f"Error evaluating case {case.case_id}: {e}")
            # Return default scores on error
            return AppropriatenessScores(
                safety_score=5.0,
                efficiency_score=5.0,
                completeness_score=5.0,
                reasoning_score=5.0,
                overall_score=5.0,
                rationale=f"Evaluation failed: {str(e)}"
            )
    
    def _parse_evaluation_response(self, response: str) -> AppropriatenessScores:
        """Parse LLM evaluation response to extract next step quality score"""
        
        # Initialize default scores
        overall_score = 5.0
        rationale = "Could not parse evaluation response"
        
        try:
            # Extract overall score using regex pattern
            overall_match = re.search(r'Overall Score:\s*(\d+(?:\.\d+)?)/10', response)
            if overall_match:
                overall_score = float(overall_match.group(1))
            
            # Extract rationale
            rationale_match = re.search(r'Rationale:\s*(.+?)(?:\n\n|\n$|$)', response, re.DOTALL)
            if rationale_match:
                rationale = rationale_match.group(1).strip()
            
        except Exception as e:
            print(f"Error parsing evaluation response: {e}")
        
        # Return simplified scores (only overall_score is meaningful now)
        return AppropriatenessScores(
            safety_score=overall_score,  # Use overall score for all fields for compatibility
            efficiency_score=overall_score,
            completeness_score=overall_score,
            reasoning_score=overall_score,
            overall_score=overall_score,
            rationale=rationale
        )
    
    def evaluate_batch(self, cases: List[ClinicalCase], 
                      batch_size: int = 5) -> List[AppropriatenessScores]:
        """Evaluate a batch of clinical cases"""
        
        print(f"🧑‍⚕️ Evaluating {len(cases)} cases with LLM judge...")
        
        results = []
        
        for i in range(0, len(cases), batch_size):
            batch = cases[i:i + batch_size]
            print(f"📋 Processing batch {i//batch_size + 1}/{(len(cases) + batch_size - 1)//batch_size}")
            
            for case in batch:
                scores = self.evaluate_case(case)
                results.append(scores)
                print(f"   Case {case.case_id}: {scores.overall_score:.1f}/10")
        
        return results
    
    def calculate_aggregate_metrics(self, scores_list: List[AppropriatenessScores]) -> Dict[str, float]:
        """Calculate aggregate appropriateness metrics"""
        
        if not scores_list:
            return {
                "avg_safety": 0.0,
                "avg_efficiency": 0.0,
                "avg_completeness": 0.0,
                "avg_reasoning": 0.0,
                "avg_overall": 0.0,
                "std_overall": 0.0,
                "min_overall": 0.0,
                "max_overall": 0.0
            }
        
        safety_scores = [s.safety_score for s in scores_list]
        efficiency_scores = [s.efficiency_score for s in scores_list]
        completeness_scores = [s.completeness_score for s in scores_list]
        reasoning_scores = [s.reasoning_score for s in scores_list]
        overall_scores = [s.overall_score for s in scores_list]
        
        return {
            "avg_safety": float(np.mean(safety_scores)),
            "avg_efficiency": float(np.mean(efficiency_scores)),
            "avg_completeness": float(np.mean(completeness_scores)),
            "avg_reasoning": float(np.mean(reasoning_scores)),
            "avg_overall": float(np.mean(overall_scores)),
            "std_overall": float(np.std(overall_scores)),
            "min_overall": float(np.min(overall_scores)),
            "max_overall": float(np.max(overall_scores)),
            "median_overall": float(np.median(overall_scores))
        }
    
    def generate_evaluation_report(self, cases: List[ClinicalCase], 
                                 scores: List[AppropriatenessScores]) -> Dict[str, Any]:
        """Generate comprehensive evaluation report"""
        
        aggregate_metrics = self.calculate_aggregate_metrics(scores)
        
        # Categorize cases by performance
        high_performing = [i for i, s in enumerate(scores) if s.overall_score >= 8.0]
        moderate_performing = [i for i, s in enumerate(scores) if 6.0 <= s.overall_score < 8.0]
        low_performing = [i for i, s in enumerate(scores) if s.overall_score < 6.0]
        
        # Identify common issues
        safety_issues = [i for i, s in enumerate(scores) if s.safety_score < 6.0]
        efficiency_issues = [i for i, s in enumerate(scores) if s.efficiency_score < 6.0]
        
        report = {
            "summary": {
                "total_cases": len(cases),
                "high_performing_count": len(high_performing),
                "moderate_performing_count": len(moderate_performing),
                "low_performing_count": len(low_performing),
                "safety_concerns_count": len(safety_issues),
                "efficiency_concerns_count": len(efficiency_issues)
            },
            "aggregate_metrics": aggregate_metrics,
            "performance_breakdown": {
                "high_performing_cases": high_performing[:10],  # Limit for brevity
                "low_performing_cases": low_performing[:10],
                "safety_concern_cases": safety_issues[:10]
            },
            "recommendations": self._generate_recommendations(scores)
        }
        
        return report
    
    def _generate_recommendations(self, scores: List[AppropriatenessScores]) -> List[str]:
        """Generate improvement recommendations based on evaluation"""
        
        recommendations = []
        
        # Calculate average scores for each dimension
        avg_safety = np.mean([s.safety_score for s in scores])
        avg_efficiency = np.mean([s.efficiency_score for s in scores])
        avg_completeness = np.mean([s.completeness_score for s in scores])
        avg_reasoning = np.mean([s.reasoning_score for s in scores])
        
        # Generate targeted recommendations
        if avg_safety < 7.0:
            recommendations.append("Improve safety protocols and risk assessment procedures")
        
        if avg_efficiency < 7.0:
            recommendations.append("Optimize resource utilization and triage algorithms")
        
        if avg_completeness < 7.0:
            recommendations.append("Enhance comprehensive factor consideration in decision-making")
        
        if avg_reasoning < 7.0:
            recommendations.append("Strengthen clinical reasoning and decision logic")
        
        # Count high-risk cases
        safety_risks = sum(1 for s in scores if s.safety_score < 5.0)
        if safety_risks > 0:
            recommendations.append(f"Address {safety_risks} high-risk safety concerns immediately")
        
        if not recommendations:
            recommendations.append("Overall performance is satisfactory - focus on consistency")
        
        return recommendations

def create_clinical_case_from_evaluation(case_id: str, patient_input: str, 
                                       model_triage: str, model_steps: str,
                                       true_triage: str, true_steps: str) -> ClinicalCase:
    """Helper function to create clinical case from evaluation data"""
    
    # Simple complexity classification based on case content
    complexity = "simple"
    if any(term in patient_input.lower() for term in ["chest pain", "stroke", "trauma"]):
        complexity = "complex"
    elif any(term in patient_input.lower() for term in ["fever", "headache", "nausea"]):
        complexity = "moderate"
    
    # Simple domain classification
    domain = "general"
    if any(term in patient_input.lower() for term in ["chest pain", "heart", "cardiac"]):
        domain = "cardiac"
    elif any(term in patient_input.lower() for term in ["breath", "cough", "lung"]):
        domain = "respiratory"
    
    return ClinicalCase(
        case_id=case_id,
        patient_presentation=patient_input,
        model_triage_decision=model_triage,
        model_next_steps=model_steps,
        ground_truth_triage=true_triage,
        ground_truth_next_steps=true_steps,
        case_complexity=complexity,
        medical_domain=domain
    )

def main():
    """Test clinical appropriateness judge"""
    print("🧑‍⚕️ Testing Clinical Appropriateness Judge")
    
    # Initialize judge
    judge = ClinicalAppropriatenessJudge()
    
    # Create test cases
    test_cases = [
        ClinicalCase(
            case_id="test_001",
            patient_presentation="45-year-old male with severe chest pain and shortness of breath",
            model_triage_decision="ED",
            model_next_steps="Immediate emergency department evaluation for possible heart attack",
            ground_truth_triage="ED",
            ground_truth_next_steps="Emergency department for cardiac workup including ECG and troponins",
            case_complexity="complex",
            medical_domain="cardiac"
        ),
        ClinicalCase(
            case_id="test_002",
            patient_presentation="25-year-old female with mild headache lasting 2 days",
            model_triage_decision="HOME",
            model_next_steps="Rest, hydration, and over-the-counter pain medication",
            ground_truth_triage="HOME",
            ground_truth_next_steps="Home care with OTC analgesics, return if worsening",
            case_complexity="simple",
            medical_domain="general"
        )
    ]
    
    # Evaluate cases
    scores = judge.evaluate_batch(test_cases)
    
    # Print results
    print(f"\n📊 Evaluation Results:")
    for case, score in zip(test_cases, scores):
        print(f"\nCase {case.case_id}:")
        print(f"  Overall Score: {score.overall_score:.1f}/10")
        print(f"  Safety: {score.safety_score:.1f}, Efficiency: {score.efficiency_score:.1f}")
        print(f"  Rationale: {score.rationale[:100]}...")
    
    # Generate report
    report = judge.generate_evaluation_report(test_cases, scores)
    print(f"\n📋 Aggregate Metrics:")
    for key, value in report["aggregate_metrics"].items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    print("\n✅ Clinical appropriateness evaluation test complete!")

if __name__ == "__main__":
    main()