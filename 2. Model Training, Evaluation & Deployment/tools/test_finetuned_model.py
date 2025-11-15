from mlx_lm import load, generate

def test_medical_qa():
    """Test the fine-tuned model on medical questions"""
    
    # Load the base model with fine-tuned adapters
    model, tokenizer = load("./mlx_models/SmolLM2-1.7B-Instruct-MLX", adapter_path="./ALL_adapters/adapters_simple_clean_1.7B")
    
    # Test questions in the same format as training data
    test_questions = [
        "Does regular exercise reduce the risk of heart disease?",
        "Can antibiotics treat viral infections?", 
        "Does smoking increase cancer risk?",
        "Is vitamin D important for bone health?",
        "Does stress affect immune system function?"
    ]
    
    print("Testing Fine-tuned PubMedQA Model")
    print("=" * 50)
    
    for question in test_questions:
        prompt = f"""Given the following medical research context, please answer the question with 'yes' or 'no'.

Question: {question}

Answer:"""
        
        print(f"\nQuestion: {question}")
        print("Model Answer: ", end="")
        
        # Generate response
        response = generate(model, tokenizer, prompt=prompt, max_tokens=1)
        print(response.strip())

if __name__ == "__main__":
    test_medical_qa() 