import Foundation

final class FAISSBridge {
    private var isLoaded = false
    private var indexSize: Int = 0
    private var dimension: Int = 384
    private var vectors: [[Float]] = []
    private var ids: [Int] = []
    
    init() {
        loadIndex()
    }
    
    func search(embedding: [Float], topK: Int) throws -> [(Int, Float)] {
        guard isLoaded else {
            throw FAISSError.indexNotLoaded
        }
        
        guard embedding.count == dimension else {
            throw FAISSError.dimensionMismatch
        }
        
        // For now, implement cosine similarity search manually
        // In production, this would use actual FAISS C++ library
        let results = computeCosineSimanties(query: embedding, topK: topK)
        return results
    }
    
    private func loadIndex() {
        guard let indexPath = Bundle.main.path(forResource: "faiss", ofType: "index"),
              let idsPath = Bundle.main.path(forResource: "ids", ofType: "bin") else {
            print("FAISS index files not found in bundle")
            return
        }
        
        print("Loading FAISS index from: \(indexPath)")
        print("Loading IDs mapping from: \(idsPath)")
        
        // Load the IDs mapping
        if let idsData = NSData(contentsOfFile: idsPath) {
            let count = idsData.length / MemoryLayout<UInt32>.size
            ids = Array(0..<count).map { index in
                var value: UInt32 = 0
                idsData.getBytes(&value, range: NSRange(location: index * 4, length: 4))
                return Int(value)
            }
            print("Loaded \(ids.count) ID mappings")
        }
        
        // For a real implementation, we would load the FAISS index here
        // For now, we'll generate mock embeddings that simulate the index
        loadMockEmbeddings()
        
        indexSize = vectors.count
        isLoaded = true
        print("FAISS bridge initialized with \(indexSize) vectors")
    }
    
    private func loadMockEmbeddings() {
        // Limit the number of vectors for mobile memory constraints
        let maxVectors = min(ids.count, 1000) // Reduced from 4418 for demo
        
        // Generate normalized random embeddings to simulate real index
        vectors = (0..<maxVectors).map { _ in
            let randomVector = (0..<dimension).map { _ in Float.random(in: -1...1) }
            let norm = sqrt(randomVector.map { $0 * $0 }.reduce(0, +))
            return randomVector.map { $0 / norm }
        }
        
        print("Loaded \(vectors.count) mock embeddings for mobile demo")
    }
    
    private func computeCosineSimanties(query: [Float], topK: Int) -> [(Int, Float)] {
        var similarities: [(Int, Float)] = []
        
        for (index, vector) in vectors.enumerated() {
            let similarity = cosineSimilarity(query, vector)
            let docId = index < ids.count ? ids[index] : index + 1
            similarities.append((docId, similarity))
        }
        
        // Sort by similarity (descending) and take top K
        similarities.sort { $0.1 > $1.1 }
        return Array(similarities.prefix(topK))
    }
    
    private func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count else { return 0.0 }
        
        let dotProduct = zip(a, b).map(*).reduce(0, +)
        let normA = sqrt(a.map { $0 * $0 }.reduce(0, +))
        let normB = sqrt(b.map { $0 * $0 }.reduce(0, +))
        
        guard normA > 0 && normB > 0 else { return 0.0 }
        return dotProduct / (normA * normB)
    }
}

enum FAISSError: Error {
    case indexNotLoaded
    case dimensionMismatch
    case searchFailed(String)
}