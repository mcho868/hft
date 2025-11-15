import Foundation

enum PromptBuilder {
    static func makePrompt(query: String, context: String, rag: Bool) -> String {
        // Use raw format - let processor.prepare() apply chat template
        // This matches what Mac Python does (passes raw text to mlx_lm.generate)
        if rag {
            return """
            Patient query: \(query)

            Context:
            \(context)

            Provide triage decision, next steps, and reasoning:
            """
        } else {
            return """
            Patient query: \(query)

            Provide triage decision, next steps, and reasoning:
            """
        }
    }
}