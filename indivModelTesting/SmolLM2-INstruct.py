from transformers import AutoModelForCausalLM, AutoTokenizer
import gc

checkpoint = ["HuggingFaceTB/SmolLM2-135M-Instruct", "HuggingFaceTB/SmolLM2-360M-Instruct"]

current_model = 0

device = "cpu" # for GPU usage or "cpu" for CP  U usage
tokenizer = AutoTokenizer.from_pretrained(checkpoint[1])
tokenizer.pad_token = tokenizer.eos_token
# for multiple GPUs install accelerate and do `model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto")`
model = AutoModelForCausalLM.from_pretrained(checkpoint[1]).to(device)

print(f"Using model: {checkpoint[current_model]}")

user_input = input("Enter your prompt: ")
messages = [
    {"role": "system", "content": "You are a doctor assitant named DR.SMOL. Your job is to pre-examine the patient's symptoms and provide a diagnosis, ultimately telling the patien if they should visit a hospital ot self treat."},
    {"role": "user", "content": user_input}
]


while user_input != "exit":
    input_text=tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer(input_text, return_tensors="pt", padding=True).to(device)
    outputs = model.generate(**inputs, temperature=0.2, top_p=0.9, do_sample=True, max_new_tokens=500)
    output_text = tokenizer.decode(outputs[0]).split("\n")[-1]
    messages.append({"role": "assistant", "content": output_text})
    print(output_text)
    user_input = input("Enter your prompt: ")
    messages.append({"role": "user", "content": user_input})

# Clean up model from memory
del model
del tokenizer
gc.collect()
print("Model unloaded from memory.")
