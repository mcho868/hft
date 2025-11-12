import Foundation

struct TriageResult {
    let triage: String
    let nextSteps: String
    let reasoning: String
    let latencyMs: Int
    let peakMemMB: Int
    let rawOutput: String
}

@MainActor
final class TriageViewModel: ObservableObject {
    private let retriever = Retriever()
    private var generator = MLXGenerator()  // Using MLX with LoRA adapters!
    private let parser = TriageParser()

    @Published var selectedModelType: MLXGenerator.ModelType = .highcapAdapter
    @Published var modelSwitchInProgress = false
    @Published var isInitialized = false

    func initializeModel() async {
        guard !isInitialized else { return }

        do {
            try await generator.initialize()
            isInitialized = true
            print("✅ Model initialized successfully")
        } catch {
            print("❌ Model initialization failed: \(error)")
        }
    }

    func switchModel(to modelType: MLXGenerator.ModelType) async {
        guard modelType != selectedModelType else { return }

        modelSwitchInProgress = true
        selectedModelType = modelType

        do {
            try await generator.switchModel(to: modelType)
            modelSwitchInProgress = false
        } catch {
            print("❌ Model switch failed: \(error)")
            modelSwitchInProgress = false
        }
    }
    
    func run(query: String, rag: Bool) async throws -> TriageResult {
        let startTime = Date()
        
        var context = ""
        if rag {
            let chunks = try retriever.search(query: query, topM: 50, topK: 5)
            context = chunks.joined(separator: "\n\n")
        }
        
        let prompt = PromptBuilder.makePrompt(query: query, context: context, rag: rag)
        let text = try await generator.generate(
            prompt: prompt,
            maxTokens: 100,
            temperature: 0.0
        )
        
        let (triage, steps, reasoning) = parser.extract(from: text)

        let latencyMs = Int(Date().timeIntervalSince(startTime) * 1000)
        let currentMemMB = MemoryReporter.currentMB()

        return TriageResult(
            triage: triage,
            nextSteps: steps,
            reasoning: reasoning,
            latencyMs: latencyMs,
            peakMemMB: currentMemMB,
            rawOutput: text
        )
    }
}

enum MemoryReporter {
    static func currentMB() -> Int {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size)/4
        
        let kerr: kern_return_t = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
                task_info(mach_task_self_,
                         task_flavor_t(MACH_TASK_BASIC_INFO),
                         $0,
                         &count)
            }
        }
        
        if kerr == KERN_SUCCESS {
            return Int(info.resident_size) / (1024 * 1024)
        } else {
            return 0
        }
    }
}
