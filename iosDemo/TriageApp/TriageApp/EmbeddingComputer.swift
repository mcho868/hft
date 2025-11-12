import Foundation
import NaturalLanguage

final class EmbeddingComputer {
    private let dimension = 384
    private var isInitialized = false
    private var tokenizer: NLTokenizer?
    private var vocabulary: [String: Int] = [:]
    
    init() {
        initializeEmbedding()
    }
    
    func computeEmbedding(for text: String) throws -> [Float] {
        guard isInitialized else {
            throw EmbeddingError.notInitialized
        }

        // ⚠️ MOCK EMBEDDING: Using simple bag-of-words + random noise
        // This does NOT match the sentence-transformers embeddings from Mac
        // RAG results will be poor/random until real embedding model is integrated
        let embedding = computeSimpleEmbedding(text: text)
        return normalizeVector(embedding)
    }
    
    private func initializeEmbedding() {
        tokenizer = NLTokenizer(unit: .word)
        tokenizer?.string = ""
        
        // Create a simple vocabulary based on common medical terms
        createMedicalVocabulary()
        
        isInitialized = true
        print("⚠️ WARNING: Using MOCK embeddings (bag-of-words + noise)")
        print("⚠️ RAG will not work correctly without real sentence-transformer model")
        print("Embedding computer initialized with \(vocabulary.count) vocabulary terms")
    }
    
    private func createMedicalVocabulary() {
        let medicalTerms = [
            "pain", "fever", "headache", "nausea", "vomiting", "diarrhea", "fatigue", "dizziness",
            "chest", "abdomen", "back", "neck", "throat", "cough", "breathing", "heart",
            "blood", "pressure", "diabetes", "infection", "inflammation", "swelling", "rash",
            "emergency", "urgent", "mild", "severe", "chronic", "acute", "symptoms", "diagnosis",
            "treatment", "medication", "doctor", "hospital", "clinic", "appointment", "monitor",
            "rest", "hydration", "exercise", "diet", "sleep", "stress", "anxiety", "depression"
        ]
        
        for (index, term) in medicalTerms.enumerated() {
            vocabulary[term.lowercased()] = index
        }
        
        // Add common words
        let commonWords = [
            "the", "and", "or", "but", "in", "on", "at", "to", "for", "with", "by", "from",
            "up", "down", "out", "off", "over", "under", "again", "further", "then", "once",
            "i", "me", "my", "myself", "we", "our", "ours", "you", "your", "yours", "he", "him",
            "his", "she", "her", "hers", "it", "its", "they", "them", "their", "theirs"
        ]
        
        for (index, word) in commonWords.enumerated() {
            vocabulary[word] = medicalTerms.count + index
        }
    }
    
    private func computeSimpleEmbedding(text: String) -> [Float] {
        let tokens = tokenizeText(text)
        var embedding = Array(repeating: Float(0.0), count: dimension)
        
        // Simple bag-of-words with position encoding
        for (position, token) in tokens.enumerated() {
            if let vocabIndex = vocabulary[token.lowercased()] {
                let weight = 1.0 / Float(position + 1) // Position weighting
                
                // Distribute the token's contribution across multiple dimensions
                for i in 0..<min(10, dimension) {
                    let dimIndex = (vocabIndex * 7 + i) % dimension
                    embedding[dimIndex] += weight * Float.random(in: 0.8...1.2)
                }
            }
        }
        
        // Add some noise to make embeddings more realistic
        for i in 0..<dimension {
            embedding[i] += Float.random(in: -0.1...0.1)
        }
        
        return embedding
    }
    
    private func tokenizeText(_ text: String) -> [String] {
        tokenizer?.string = text.lowercased()
        var tokens: [String] = []
        
        tokenizer?.enumerateTokens(in: text.startIndex..<text.endIndex) { range, _ in
            let token = String(text[range])
            tokens.append(token)
            return true
        }
        
        return tokens
    }
    
    private func normalizeVector(_ vector: [Float]) -> [Float] {
        let norm = sqrt(vector.map { $0 * $0 }.reduce(0, +))
        guard norm > 0 && norm.isFinite else { 
            // Return a random unit vector if norm is 0 or infinite
            var randomVector = Array(repeating: Float(0.0), count: dimension)
            for i in 0..<dimension {
                randomVector[i] = Float.random(in: -1...1)
            }
            let randomNorm = sqrt(randomVector.map { $0 * $0 }.reduce(0, +))
            return randomVector.map { $0 / randomNorm }
        }
        return vector.map { $0 / norm }
    }
}

enum EmbeddingError: Error {
    case notInitialized
    case computationFailed(String)
}