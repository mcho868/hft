import Foundation

final class TriageParser {
    func extract(from text: String) -> (String, String, String) {
        print("🔥 DEBUG: Parsing text: '\(text)'")
        
        // Clean the text first - remove special tokens
        let cleanedText = cleanModelOutput(text)
        print("🔥 DEBUG: Cleaned text: '\(cleanedText)'")
        
        let triage = extractTriageDecision(from: cleanedText)

        // Try both "Next Step:" and "Next steps:" (singular and plural)
        var nextSteps = capture("Next Step", from: cleanedText)
        if nextSteps == "UNKNOWN" {
            nextSteps = capture("Next steps", from: cleanedText)
        }

        let reasoning = capture("Reasoning", from: cleanedText)

        print("🔥 DEBUG: Extracted triage: '\(triage)'")
        print("🔥 DEBUG: Extracted next steps: '\(nextSteps)'")
        print("🔥 DEBUG: Extracted reasoning: '\(reasoning)'")
        return (triage, nextSteps, reasoning)
    }
    
    private func cleanModelOutput(_ text: String) -> String {
        var cleaned = text
        
        // Remove common special tokens from model output
        let specialTokens = [
            "<filename>", "</filename>",
            "<reponame>", "</reponame>", 
            "<file_sep>", "<gh_stars>",
            "<empty_output>", "<jupyter_start>",
            "<jupyter_text>", "<jupyter_code>",
            "<jupyter_output>", "<issue_start>",
            "<issue_comment>", "<issue_closed>",
            "<?>"
        ]
        
        for token in specialTokens {
            cleaned = cleaned.replacingOccurrences(of: token, with: "")
        }
        
        // Clean up extra whitespace and newlines
        cleaned = cleaned.replacingOccurrences(of: "\\s+", with: " ", options: .regularExpression)
        cleaned = cleaned.trimmingCharacters(in: .whitespacesAndNewlines)
        
        return cleaned
    }
    
    private func extractTriageDecision(from text: String) -> String {
        print("🔥 DEBUG: Looking for 'Triage Decision:' pattern in: '\(text)'")
        
        // Look for exact pattern: "Triage Decision: [something]" (case insensitive)
        let pattern = "Triage Decision:\\s*([A-Z]+)"
        
        do {
            let regex = try NSRegularExpression(pattern: pattern, options: .caseInsensitive)
            let range = NSRange(location: 0, length: text.utf16.count)
            
            if let match = regex.firstMatch(in: text, options: [], range: range) {
                let matchRange = match.range(at: 1)
                if let swiftRange = Range(matchRange, in: text) {
                    let decision = String(text[swiftRange]).trimmingCharacters(in: .whitespacesAndNewlines).uppercased()
                    print("🔥 DEBUG: Found 'Triage Decision:' with value: '\(decision)'")
                    
                    // Only return if it's exactly ED, GP, or HOME
                    if ["ED", "GP", "HOME"].contains(decision) {
                        return decision
                    } else {
                        print("🔥 DEBUG: Decision '\(decision)' is not valid ED/GP/HOME")
                        return "UNKNOWN"
                    }
                }
            }
        } catch {
            print("🔥 DEBUG: Regex error: \(error)")
        }
        
        print("🔥 DEBUG: No 'Triage Decision:' pattern found")
        return "UNKNOWN"
    }
    
    private func capture(_ label: String, from text: String) -> String {
        // Look for pattern like "Next Step: [content]" and stop at next section or end
        // Stops at: "Reasoning:", "Next Step:", "Triage Decision:", or end of string
        let pattern = "\(label):\\s*(.*?)(?=(?:Reasoning:|Next Step:|Triage Decision:|$))"

        do {
            let regex = try NSRegularExpression(pattern: pattern, options: [.caseInsensitive, .dotMatchesLineSeparators])
            let range = NSRange(location: 0, length: text.utf16.count)

            if let match = regex.firstMatch(in: text, options: [], range: range) {
                let matchRange = match.range(at: 1)
                if let swiftRange = Range(matchRange, in: text) {
                    let captured = String(text[swiftRange])
                    return captured.trimmingCharacters(in: .whitespacesAndNewlines)
                }
            }
        } catch {
            print("Regex error for \(label): \(error)")
        }

        return "UNKNOWN"
    }
}