#!/usr/bin/env python3
"""
Medical Chatbot with Manual Model Switching
- Switches between two top-performing LoRA adapters
- 1.7B Aggressive Split2 (best performer)
- 360M Aggressive Split1 (faster alternative)
"""

import re
from mlx_lm import load, generate
from typing import Tuple, Optional

class MedicalChatbot:
    def __init__(self):
        # self.base_model_path = "./mlx_models/SmolLM2-1.7B-Instruct-MLX"
        self.base_model_360m_path = "./mlx_models/SmolLM2-360M-Instruct-MLX"
        
        # Top performing adapters
        # self.adapter_1_7b_path = "./All_adapters/split_experiments_pubmedqa/adapters_1.7B_aggressive_split2"
        self.adapter_360m_path = "./All_adapters/split_experiments_pubmedqa/adapters_360M_aggressive_split1"
        self.adapter_360m_reasoning_path = "./All_adapters/o1_med_reasoning_adapters/adapters_360M_higher_lr_reasoning_split0"
        
        # Model configurations
        self.models = {
            # "1.7B": {
            #     "base_path": self.base_model_path,
            #     "adapter_path": self.adapter_1_7b_path,
            #     "model": None,
            #     "tokenizer": None,
            #     "loaded": False,
            #     "description": "1.7B Aggressive (Best Performance - 60.0% accuracy)"
            # },
            "360M_PubmedQA": {
                "base_path": self.base_model_360m_path,
                "adapter_path": self.adapter_360m_path,
                "model": None,
                "tokenizer": None,
                "loaded": False,
                "description": "360M Aggressive (Faster - 51.3% accuracy)"
            },
            "360M_Reasoning": {
                "base_path": self.base_model_360m_path,
                "adapter_path": self.adapter_360m_reasoning_path,
                "model": None,
                "tokenizer": None,
                "loaded": False,
                "description": "360M Reasoning (Higher LR, Split 0)"
            }
        }
        
        self.current_model = "360M_PubmedQA"  # Default to best performer
        
        print("🤖 Medical Chatbot with Top LoRA Adapters")
        print("=" * 50)
        print("Available models:")
        for key, config in self.models.items():
            print(f"  {key}: {config['description']}")
        print("=" * 50)
        print("Commands:")
        print("  'switch' - Change between models")
        print("  'status' - Show current model info")
        print("  'quit' - Exit")
        print("=" * 50)
        
        # Load default model
        self.load_model(self.current_model)
    
    def load_model(self, model_key: str) -> bool:
        """Load specified model with adapter"""
        if model_key not in self.models:
            print(f"❌ Unknown model: {model_key}")
            return False
        
        config = self.models[model_key]
        
        if config["loaded"]:
            print(f"✅ {model_key} already loaded!")
            return True
        
        print(f"🔄 Loading {model_key} with adapter...")
        print(f"   Base: {config['base_path']}")
        print(f"   Adapter: {config['adapter_path']}")
        
        try:
            model, tokenizer = load(
                config["base_path"], 
                adapter_path=config["adapter_path"]
            )
            
            config["model"] = model
            config["tokenizer"] = tokenizer
            config["loaded"] = True
            
            print(f"✅ {model_key} loaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error loading {model_key}: {e}")
            return False
    
    def switch_model(self):
        """Interactive model switching"""
        print("\n🔄 Available models:")
        for key, config in self.models.items():
            status = "✅ Loaded" if config["loaded"] else "⏳ Not loaded"
            current = "👈 CURRENT" if key == self.current_model else ""
            print(f"  {key}: {config['description']} [{status}] {current}")
        
        choice = input("\nSelect model (360M_PubmedQA/360M_Reasoning): ").strip()
        
        if choice in self.models:
            if self.load_model(choice):
                self.current_model = choice
                print(f"✅ Switched to {choice}")
            else:
                print(f"❌ Failed to switch to {choice}")
        else:
            print(f"❌ Invalid choice: {choice}")
    
    def show_status(self):
        """Show current model status"""
        config = self.models[self.current_model]
        print(f"\n📊 Current Model: {self.current_model}")
        print(f"   Description: {config['description']}")
        print(f"   Base Model: {config['base_path']}")
        print(f"   Adapter: {config['adapter_path']}")
        print(f"   Status: {'✅ Loaded' if config['loaded'] else '❌ Not loaded'}")
    
    def generate_response(self, user_input: str) -> str:
        """Generate response using current model"""
        config = self.models[self.current_model]
        
        if not config["loaded"]:
            if not self.load_model(self.current_model):
                return "❌ Model not available"
        
        model = config["model"]
        tokenizer = config["tokenizer"]
        
        # Format prompt for medical questions
        if self.current_model == "360M_Reasoning":
            prompt = user_input
            max_tokens = 2000
        elif any(word in user_input.lower() for word in ['does', 'can', 'is', 'do', 'will', 'should']):
            prompt = f"Medical question: {user_input}"
            max_tokens = 10  # Short answers for yes/no questions
        else:
            prompt = f"Medical question: {user_input}\nAnswer:"
            max_tokens = 50
        
        try:
            response = generate(
                model, tokenizer,
                prompt=prompt,
                max_tokens=max_tokens
            )
            return response.strip()
        except Exception as e:
            return f"❌ Error generating response: {e}"
    
    def chat(self):
        """Main chat loop"""
        print(f"\n👋 Hello! I'm using the {self.current_model} medical model.")
        print("Ask me medical questions or use commands!")
        
        while True:
            try:
                print(f"\n[{self.current_model}] " + "="*50)
                user_input = input("You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye', 'q']:
                    print("👋 Goodbye! Take care!")
                    break
                
                if user_input.lower() == 'switch':
                    self.switch_model()
                    continue
                
                if user_input.lower() == 'status':
                    self.show_status()
                    continue
                
                if not user_input:
                    continue
                
                print(f"\nBot ({self.current_model}): ", end="")
                response = self.generate_response(user_input)
                print(response)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye! Take care!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                continue

def main():
    """Run the medical chatbot"""
    try:
        bot = MedicalChatbot()
        bot.chat()
    except Exception as e:
        print(f"Failed to initialize chatbot: {e}")

if __name__ == "__main__":
    main() 