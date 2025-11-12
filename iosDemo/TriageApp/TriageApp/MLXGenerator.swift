import Foundation
import MLX
import MLXLMCommon
import MLXLLM
import MLXNN
import Tokenizers

@MainActor
final class MLXGenerator {
    private var modelContainer: ModelContainer?
    private var isInitialized = false
    private var currentModelType: ModelType = .highcapAdapter  // Default to high capacity adapter
    private var adapterURL: URL?  // Store adapter URL for use during generation

    enum ModelType: String, CaseIterable {
        case baseModel = "SmolLM2-135M-Base"
        case highcapAdapter = "SmolLM2-135M-HighCap"
        case perfsafeAdapter = "SmolLM2-135M-PerfSafe"

        var displayName: String {
            switch self {
            case .baseModel: return "Base Model (No Adapter)"
            case .highcapAdapter: return "High Capacity (MLX + Adapter)"
            case .perfsafeAdapter: return "Performance Safe (MLX + Adapter)"
            }
        }

        var expectedSpeed: String {
            switch self {
            case .baseModel: return "~50-100ms"
            case .highcapAdapter: return "~50-100ms"
            case .perfsafeAdapter: return "~50-100ms"
            }
        }

        var adapterPath: String? {
            switch self {
            case .baseModel:
                return nil  // No adapter for base model
            case .highcapAdapter:
                return "MLXModels/adapter_safe_triage_SmolLM2-135M_4bit_high_capacity_safe"
            case .perfsafeAdapter:
                return "MLXModels/adapter_safe_triage_SmolLM2-135M_4bit_performance_safe"
            }
        }

        var adapterFileName: String? {
            switch self {
            case .baseModel:
                return nil  // No adapter for base model
            case .highcapAdapter:
                return "adapters_highcap.safetensors"
            case .perfsafeAdapter:
                return "adapters_perfsafe.safetensors"
            }
        }
    }

    init(modelType: ModelType = .highcapAdapter) {
        self.currentModelType = modelType
    }

    func initialize() async throws {
        print("🚀 Initializing MLX model: \(currentModelType.displayName)")

        // Create a temporary directory with only base model files (no adapters)
        let fileManager = FileManager.default
        let tempDir = fileManager.temporaryDirectory.appendingPathComponent("SmolLM-Base-\(UUID().uuidString)")
        try fileManager.createDirectory(at: tempDir, withIntermediateDirectories: true)

        // Copy base model files - include chat_template only for base model (not adapters)
        var baseModelFiles = ["config.json", "model.safetensors", "tokenizer.json",
                             "vocab.json", "merges.txt", "special_tokens_map.json",
                             "tokenizer_config.json", "generation_config.json"]

        // Only include chat template for base model (adapters were trained without ChatML)
        if currentModelType == .baseModel {
            baseModelFiles.append("chat_template.jinja")
        }

        guard let bundlePath = Bundle.main.resourcePath else {
            throw MLXError.modelNotFound("Bundle resource path not found")
        }

        for fileName in baseModelFiles {
            let sourcePath = URL(fileURLWithPath: bundlePath).appendingPathComponent(fileName)
            let destPath = tempDir.appendingPathComponent(fileName)
            if fileManager.fileExists(atPath: sourcePath.path) {
                try? fileManager.copyItem(at: sourcePath, to: destPath)
            }
        }

        print("📁 Model directory: \(tempDir.path)")

        // Create model configuration for local bundled model
        let configuration = ModelConfiguration(
            directory: tempDir,
            defaultPrompt: "What is the capital of France?"
        )

        print("📁 Loading model with LLMModelFactory...")

        // Load the model container using LLMModelFactory
        // Use loadContainer with overrideTokenizer to handle models with LoRA keys
        do {
            self.modelContainer = try await LLMModelFactory.shared.loadContainer(
                configuration: configuration
            ) { progress in
                print("📊 Loading progress: \(progress)")
            }
            print("✅ Base model loaded successfully")
        } catch {
            print("⚠️ Standard loading failed: \(error)")
            print("🔄 Attempting to load with LoRA compatibility...")
            throw error
        }

        // Check if this model type uses an adapter
        if let adapterFileName = currentModelType.adapterFileName,
           let adapterPath = currentModelType.adapterPath,
           let adapterFile = Bundle.main.url(forResource: adapterFileName, withExtension: nil),
           let container = self.modelContainer {
            // Apply adapter ONCE during initialization
            print("📁 Adapter path: \(adapterFile.path)")

            // Load adapter configuration to get scale parameter
            struct AdapterConfig: Codable {
                struct LoRAParameters: Codable {
                    let rank: Int
                    let scale: Float
                }
                let num_layers: Int
                let lora_parameters: LoRAParameters
            }

            var adapterConfig: AdapterConfig?
            // Config filename matches adapter filename (e.g., adapters_highcap.safetensors -> adapter_config_highcap.json)
            let configName = adapterFileName.replacingOccurrences(of: "adapters_", with: "adapter_config_").replacingOccurrences(of: ".safetensors", with: "")
            if let configPath = Bundle.main.path(forResource: configName, ofType: "json"),
               let configData = try? Data(contentsOf: URL(fileURLWithPath: configPath)) {
                adapterConfig = try? JSONDecoder().decode(AdapterConfig.self, from: configData)
                if let config = adapterConfig {
                    print("📋 Loaded adapter config: rank=\(config.lora_parameters.rank), scale=\(config.lora_parameters.scale)")
                } else {
                    print("⚠️ Failed to decode adapter config from \(configName).json")
                }
            } else {
                print("⚠️ Could not find adapter config: \(configName).json")
            }

            await container.perform { context in
                // Check if model supports LoRA
                guard let loraModel = context.model as? (any LoRAModel) else {
                    print("⚠️ Model does not support LoRA")
                    return
                }

                print("🔧 Converting model layers for LoRA...")
                // Get all LoRA-capable linear layers
                let allLayers = Array(loraModel.loraLinearLayers())
                print("🔍 Total LoRA-capable layers: \(allLayers.count)")

                // Adapters were trained on last N transformer blocks (from adapter config)
                let numBlocksToConvert = adapterConfig?.num_layers ?? 16
                let rank = adapterConfig?.lora_parameters.rank ?? 8
                let scale = adapterConfig?.lora_parameters.scale ?? 20.0

                let layers = Array(allLayers.suffix(numBlocksToConvert))
                print("🔧 Converting \(layers.count) blocks with rank=\(rank), scale=\(scale)")

                // Manually convert layers with correct scale (instead of LoRATrain.convert which uses default scale=20.0)
                context.model.freeze()
                for (layer, keys) in layers {
                    var update = ModuleChildren()
                    let children = layer.children()
                    for key in keys {
                        if let item = children[key], case .value(let child) = item {
                            if let linear = child as? Linear {
                                // Create LoRALinear with correct scale from config
                                let loraLinear = LoRALinear.from(linear: linear, rank: rank, scale: scale)
                                update[key] = .value(loraLinear)
                                print("  ✓ Converted \(key) with scale=\(scale)")
                            }
                        }
                    }
                    layer.update(modules: update)
                }

                print("🔧 Loading LoRA weights from \(adapterFile.lastPathComponent)...")
                do {
                    try LoRATrain.loadLoRAWeights(model: context.model, url: adapterFile)
                    print("✅ LoRA adapter loaded successfully")
                } catch {
                    print("❌ Failed to load LoRA weights: \(error)")
                }
            }

            self.adapterURL = adapterFile
            isInitialized = true
            print("🎉 MLX Generator ready with \(currentModelType.displayName)")
        } else {
            // Base model only (no adapter)
            self.adapterURL = nil
            isInitialized = true
            print("🎉 MLX Generator ready with \(currentModelType.displayName) (Base Model - No Adapter)")
        }
    }

    func switchModel(to modelType: ModelType) async throws {
        guard modelType != currentModelType else { return }

        currentModelType = modelType
        isInitialized = false
        modelContainer = nil

        try await initialize()
    }

    func generate(prompt: String, maxTokens: Int, temperature: Float) async throws -> String {
        guard isInitialized, let modelContainer = modelContainer else {
            throw MLXError.modelNotInitialized
        }

        print("\n🔮 GENERATING RESPONSE")
        print("📝 Input prompt: \(prompt)")
        print("⚙️  Parameters: maxTokens=\(maxTokens), temp=\(temperature)")

        let startTime = Date()

        // Create generation parameters
        // Use topP=1.0 for deterministic generation when temp=0
        let generateParameters = GenerateParameters(
            maxTokens: maxTokens,
            temperature: temperature,
            topP: temperature == 0.0 ? 1.0 : 0.95
        )

        // Generate response using the container's perform method
        let result = try await modelContainer.perform { context in
            // Prepare input using processor (handles chat template if enabled)
            print("🔧 Preparing input with processor...")
            let input = try await context.processor.prepare(
                input: .init(prompt: .text(prompt))
            )
            print("🔍 Input prepared")

            // Call generate to get an asynchronous stream
            let stream = try MLXLMCommon.generate(
                input: input, parameters: generateParameters, context: context)

            // Loop over the stream to collect results
            var allTokens: [Int] = []
            var response = ""
            for await batch in stream {
                if let newPart = batch.chunk, !newPart.isEmpty {
                    response += newPart
                    // We can still collect tokens if needed for debugging
                    allTokens.append(contentsOf: context.tokenizer.encode(text: newPart))
                }
            }

            print("🔍 Generated \(allTokens.count) tokens")
            if allTokens.count > 0 {
                print("🔍 First 20 output token IDs: \(Array(allTokens.prefix(min(20, allTokens.count))))")
                print("🔍 First decoded token: \(context.tokenizer.decode(tokens: [allTokens[0]]))")
                if allTokens.count > 1 {
                    print("🔍 Second decoded token: \(context.tokenizer.decode(tokens: [allTokens[1]]))")
                }
            }
            print("🔍 Full decoded length: \(response.count) chars")
            return response
        }

        let elapsed = Date().timeIntervalSince(startTime)

        print("\n✅ GENERATION COMPLETE")
        print("⏱️  Time: \(Int(elapsed * 1000))ms")
        print("📄 Full response:\n---\n\(result)\n---\n")

        return result
    }
}

enum MLXError: LocalizedError {
    case modelNotInitialized
    case modelNotFound(String)
    case adapterNotFound(String)
    case generationFailed(String)

    var errorDescription: String? {
        switch self {
        case .modelNotInitialized:
            return "MLX model is not initialized. Call initialize() first."
        case .modelNotFound(let name):
            return "Model not found: \(name)"
        case .adapterNotFound(let name):
            return "Adapter not found: \(name)"
        case .generationFailed(let reason):
            return "Generation failed: \(reason)"
        }
    }
}
