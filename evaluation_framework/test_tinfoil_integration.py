#!/usr/bin/env python3
"""
Test TinfoilAgent integration with the LLM-as-judge system.
"""

import sys
from pathlib import Path
from appropriateness_judge import ClinicalAppropriatenessJudge, create_clinical_case_from_evaluation

def test_tinfoil_agent():
    """Test TinfoilAgent integration"""
    print("🧪 Testing TinfoilAgent Integration")
    print("="*50)
    
    try:
        # Test TinfoilAgent initialization
        print("1. Testing TinfoilAgent initialization...")
        judge = ClinicalAppropriatenessJudge(
            judge_model="llama3-3-70b",
            use_tinfoil=True,
            rate_limit_delay=1.0
        )
        
        print(f"✅ Judge initialized with model: {judge.judge_model}")
        print(f"   Client type: {type(judge.client).__name__}")
        
        # Test direct API call
        print("\n2. Testing direct TinfoilAgent API call...")
        test_prompt = "Evaluate the clinical appropriateness of prescribing antibiotics for a viral upper respiratory infection."
        
        try:
            response = judge.client.complete(test_prompt, max_tokens=500)
            print(f"✅ API Response received ({len(response)} characters)")
            print(f"   Response preview: {response[:200]}...")
            
            if "Clinical Appropriateness" in response or len(response) > 50:
                print("✅ Response appears to be from real API")
            else:
                print("⚠️  Response appears to be from fallback mock")
                
        except Exception as e:
            print(f"❌ Error testing API call: {e}")
        
        # Test clinical case evaluation
        print("\n3. Testing clinical case evaluation...")
        
        test_case = create_clinical_case_from_evaluation(
            case_id="tinfoil_test_001",
            patient_input="45-year-old male presents with severe chest pain radiating to left arm, diaphoresis, and shortness of breath lasting 30 minutes",
            model_triage="ED",
            model_steps="Immediate emergency department evaluation. Obtain ECG, chest X-ray, and cardiac biomarkers. Initiate cardiac monitoring.",
            true_triage="ED", 
            true_steps="Emergency department for acute coronary syndrome workup including ECG, troponins, and cardiology consultation"
        )
        
        try:
            appropriateness_scores = judge.evaluate_case(test_case)
            
            print(f"✅ Case evaluation completed")
            print(f"   Overall Score: {appropriateness_scores.overall_score:.1f}/10")
            print(f"   Safety Score: {appropriateness_scores.safety_score:.1f}/10")
            print(f"   Rationale: {appropriateness_scores.rationale[:100]}...")
            
            return True
            
        except Exception as e:
            print(f"❌ Error in case evaluation: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Error initializing TinfoilAgent: {e}")
        return False

def test_environment_setup():
    """Test environment and API key setup"""
    print("\n🔧 Testing Environment Setup")
    print("="*50)
    
    try:
        # Test TinfoilAgent import
        tinfoil_path = Path(__file__).parent.parent / "mlx_models"
        sys.path.insert(0, str(tinfoil_path))
        
        from tinfoilAgent import TinfoilAgent
        print("✅ TinfoilAgent imported successfully")
        
        # Test environment variables
        import os
        api_key = os.getenv("TINFOIL_API_KEY")
        endpoint = os.getenv("TINFOIL_ENDPOINT")
        
        if api_key:
            print(f"✅ TINFOIL_API_KEY found (length: {len(api_key)})")
        else:
            print("❌ TINFOIL_API_KEY not found in environment")
        
        if endpoint:
            print(f"✅ TINFOIL_ENDPOINT found: {endpoint}")
        else:
            print("❌ TINFOIL_ENDPOINT not found in environment")
        
        # Test agent initialization
        try:
            agent = TinfoilAgent("llama3-3-70b")
            print("✅ TinfoilAgent initialized successfully")
            
            # Test simple call
            test_response = agent.getResponse("Hello, please respond with 'API working' if you can see this message.")
            if test_response:
                print(f"✅ Test API call successful: {test_response[:50]}...")
                return True
            else:
                print("❌ Test API call returned None")
                return False
                
        except Exception as e:
            print(f"❌ Error initializing TinfoilAgent: {e}")
            return False
            
    except ImportError as e:
        print(f"❌ Cannot import TinfoilAgent: {e}")
        return False

def test_fallback_behavior():
    """Test fallback behavior when TinfoilAgent is not available"""
    print("\n🔄 Testing Fallback Behavior")
    print("="*50)
    
    try:
        # Test with TinfoilAgent disabled
        judge = ClinicalAppropriatenessJudge(
            judge_model="llama3-3-70b",
            use_tinfoil=False,  # Force fallback
            rate_limit_delay=0.1
        )
        
        print("✅ Fallback judge initialized")
        
        # Test evaluation with fallback
        test_case = create_clinical_case_from_evaluation(
            case_id="fallback_test_001",
            patient_input="Patient with mild headache",
            model_triage="HOME",
            model_steps="Rest and over-the-counter pain medication",
            true_triage="HOME",
            true_steps="Home care with OTC analgesics"
        )
        
        scores = judge.evaluate_case(test_case)
        print(f"✅ Fallback evaluation completed: {scores.overall_score:.1f}/10")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing fallback: {e}")
        return False

def main():
    """Run all TinfoilAgent integration tests"""
    print("🤖 TinfoilAgent Integration Tests")
    print("="*60)
    
    tests = [
        ("Environment Setup", test_environment_setup),
        ("TinfoilAgent Integration", test_tinfoil_agent),
        ("Fallback Behavior", test_fallback_behavior)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("🎯 TINFOIL INTEGRATION TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\n📊 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! TinfoilAgent integration is working.")
        print("\n🚀 Ready to run clinical evaluation with TinfoilAgent:")
        print("   python optimized_master_runner.py")
    elif passed > 0:
        print("⚠️  Partial success - some features may work with limitations.")
        print("   The system will fall back to mock evaluation where needed.")
    else:
        print("❌ All tests failed - check your TinfoilAgent setup.")
        print("   Environment variables: TINFOIL_API_KEY, TINFOIL_ENDPOINT")
    
    print(f"\n💡 Configuration Summary:")
    print(f"   Judge Model: llama3-3-70b (TinfoilAgent)")
    print(f"   Fallback: Mock evaluation if API unavailable")
    print(f"   Integration: Uses your existing tinfoilAgent.py")

if __name__ == "__main__":
    main()