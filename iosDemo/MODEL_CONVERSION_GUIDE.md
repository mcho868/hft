# Converting MLX Models to iOS-Compatible Formats

## ✅ **Option 1: CoreML Conversion (Recommended)**

### Convert MLX → CoreML
```python
# Install required packages
pip install coremltools transformers torch

# Convert SmolLM to CoreML
import coremltools as ct
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load your fine-tuned model
model_path = "/Users/choemanseung/789/hft/iosDemo/models/SmolLM2-135M_4bit_highcap"
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Convert to CoreML
example_input = tokenizer("Patient symptoms:", return_tensors="pt", max_length=512)
traced_model = torch.jit.trace(model, example_input['input_ids'])

# Convert to CoreML format
coreml_model = ct.convert(
    traced_model,
    inputs=[ct.TensorType(shape=(1, 512), dtype=np.int32)],
    minimum_deployment_target=ct.target.iOS17,
    compute_precision=ct.precision.FLOAT16
)

# Save for iOS
coreml_model.save("SmolLM_Medical_Triage.mlpackage")
```

### Use in iOS Swift
```swift
import CoreML

class MLXToCoreMLGenerator {
    private var model: MLModel?
    
    init() {
        loadModel()
    }
    
    private func loadModel() {
        guard let modelURL = Bundle.main.url(forResource: "SmolLM_Medical_Triage", withExtension: "mlpackage") else {
            print("Model not found")
            return
        }
        
        do {
            model = try MLModel(contentsOf: modelURL)
            print("CoreML model loaded successfully")
        } catch {
            print("Failed to load model: \(error)")
        }
    }
    
    func generate(prompt: String) async throws -> String {
        guard let model = model else {
            throw GenerationError.modelNotLoaded
        }
        
        // Tokenize input
        let tokens = tokenize(prompt)
        
        // Create MLMultiArray input
        let input = try MLMultiArray(shape: [1, NSNumber(value: tokens.count)], dataType: .int32)
        for (i, token) in tokens.enumerated() {
            input[i] = NSNumber(value: token)
        }
        
        // Run inference
        let prediction = try await model.prediction(from: MLDictionaryFeatureProvider(dictionary: ["input_ids": input]))
        
        // Decode output
        return decodeTokens(prediction)
    }
}
```

## ✅ **Option 2: ONNX Runtime (Alternative)**

### Convert MLX → ONNX → iOS
```python
# Convert to ONNX first
import torch.onnx

# Export model to ONNX
torch.onnx.export(
    model,
    example_input['input_ids'],
    "smollm_medical.onnx",
    input_names=['input_ids'],
    output_names=['logits'],
    dynamic_axes={'input_ids': {1: 'sequence_length'}}
)
```

### Use ONNX Runtime in iOS
```swift
import ONNXRuntime

class ONNXGenerator {
    private var ortSession: ORTSession?
    
    func generate(tokens: [Int32]) throws -> [Float] {
        let inputTensor = try ORTValue(tensorData: NSMutableData(bytes: tokens, length: tokens.count * 4),
                                      elementType: .int32,
                                      shape: [1, tokens.count])
        
        let outputs = try ortSession!.run(withInputs: ["input_ids": inputTensor],
                                         outputNames: ["logits"],
                                         runOptions: nil)
        
        return try outputs["logits"]!.tensorData() as! [Float]
    }
}
```

## 🚀 **Practical Implementation Steps**

### 1. Update your current project:
```bash
cd /Users/choemanseung/789/hft/iosDemo/TriageApp
```

### 2. Convert your model:
```python
# Run conversion script
python convert_mlx_to_coreml.py
```

### 3. Replace MLXGenerator.swift:
- Remove mock responses
- Add CoreML inference
- Keep same interface

### 4. Bundle converted model:
- Add .mlpackage to Xcode project
- Update Bundle resource loading

## 📊 **Performance Comparison**

| Format | Speed | Memory | iOS Support |
|--------|-------|---------|-------------|
| CoreML | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| ONNX | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| MLX | ⭐⭐ | ⭐⭐ | ⭐ |

**Recommendation**: Use CoreML for best iOS performance and Apple ecosystem integration.