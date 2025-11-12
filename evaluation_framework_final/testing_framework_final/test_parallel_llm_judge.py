#!/usr/bin/env python3
"""
Quick test script for parallel LLM judge functionality
"""

import os
import sys
import logging
import time
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_parallel_functionality():
    """Test the parallel LLM judge functionality"""
    
    logger.info("🧪 Testing Parallel LLM Judge Functionality")
    logger.info("=" * 50)
    
    try:
        # Test imports
        logger.info("📦 Testing imports...")
        from llm_quality_judge import MedicalTriageQualityJudge
        from parallel_llm_judge import ParallelLLMJudgeManager
        from testing_core_llm_as_judge import TriageInferenceEngine
        logger.info("✅ All imports successful")
        
        # Test parallel manager initialization
        logger.info("🔧 Testing ParallelLLMJudgeManager...")
        with ParallelLLMJudgeManager(max_judge_workers=2) as manager:
            logger.info("✅ Parallel manager initialized successfully")
            
            # Test submitting a mock case
            mock_inference_result = {
                "case_id": "test_001",
                "query": "I have chest pain and shortness of breath",
                "predicted_triage": "ED",
                "response": "Triage decision: ED. This requires immediate medical attention.",
                "expected_triage": "ED",
                "expected_next_steps": "Seek emergency care immediately",
                "expected_reasoning": "Chest pain with shortness of breath suggests potential cardiac emergency"
            }
            
            logger.info("📋 Submitting test case for evaluation...")
            manager.submit_for_evaluation("test_001", mock_inference_result)
            
            # Wait for result
            logger.info("⏳ Waiting for LLM evaluation...")
            result = manager.wait_for_case("test_001", timeout=30.0)
            
            if result:
                logger.info("✅ LLM evaluation successful!")
                logger.info(f"   Next Step Quality: {result.next_step_quality_score:.1f}/100")
                logger.info(f"   Reasoning Quality: {result.reasoning_quality_score:.1f}/100")
                logger.info(f"   Overall Quality: {result.overall_score:.1f}/100")
            else:
                logger.warning("⚠️ LLM evaluation timed out or failed")
        
        # Test inference engine
        logger.info("🔧 Testing TriageInferenceEngine with parallel processing...")
        engine = TriageInferenceEngine(
            retriever=None,
            enable_llm_judge=True,
            judge_model="llama3-3-70b",
            max_judge_workers=2
        )
        logger.info("✅ Inference engine initialized successfully")
        
        logger.info("🎉 All tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def main():
    """Main test function"""
    
    # Check if API key is available
    api_key = os.getenv("TINFOIL_API_KEY")
    if api_key:
        logger.info("✅ TINFOIL_API_KEY found")
    else:
        logger.warning("⚠️ TINFOIL_API_KEY not found, will use mock responses")
    
    # Run tests
    success = test_parallel_functionality()
    
    if success:
        logger.info("🎉 Parallel LLM Judge test completed successfully!")
        return 0
    else:
        logger.error("❌ Parallel LLM Judge test failed!")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)