import Foundation
import SQLite3

final class Retriever {
    private var db: OpaquePointer?
    private let faissBridge = FAISSBridge()
    private let embeddingComputer = EmbeddingComputer()
    
    init() {
        openDatabase()
    }
    
    deinit {
        sqlite3_close(db)
    }
    
    func search(query: String, topM: Int, topK: Int) throws -> [String] {
        // Limit topM to reasonable size for mobile performance
        let adjustedTopM = min(topM, 20)
        let adjustedTopK = min(topK, 5)
        
        let denseResults = try performDenseSearch(query: query, topM: adjustedTopM)
        let lexicalResults = try performLexicalSearch(query: query, topM: adjustedTopM)
        
        let fusedResults = RRF.fuse(
            dense: denseResults,
            lex: lexicalResults,
            alpha: 0.7,
            k: 60
        )
        
        let topChunkIds = Array(fusedResults.prefix(adjustedTopK)).map { $0.id }
        let chunks = try fetchChunkTexts(ids: topChunkIds)
        
        // Limit context length for mobile memory constraints
        return chunks.map { chunk in
            chunk.count > 1000 ? String(chunk.prefix(1000)) + "..." : chunk
        }
    }
    
    private func openDatabase() {
        db = DatabaseHelper.shared.getDatabase()
    }
    
    private func performDenseSearch(query: String, topM: Int) throws -> [(Int, Float)] {
        let embedding = try computeEmbedding(query)
        return try searchFAISS(embedding: embedding, topM: topM)
    }
    
    private func performLexicalSearch(query: String, topM: Int) throws -> [(Int, Float)] {
        guard let db = db else {
            throw RetrievalError.databaseNotOpen
        }
        
        // Use FTS5 for lexical search
        let sql = """
        SELECT c.id, bm25(chunks_fts) as score
        FROM chunks_fts 
        JOIN chunks c ON chunks_fts.rowid = c.id
        WHERE chunks_fts MATCH ?
        ORDER BY score DESC
        LIMIT ?
        """
        
        var statement: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &statement, nil) == SQLITE_OK else {
            print("SQL prepare error: \(String(cString: sqlite3_errmsg(db)))")
            // Fallback to simpler search if FTS5 with BM25 not available
            return try performSimpleLexicalSearch(query: query, topM: topM)
        }
        
        defer { sqlite3_finalize(statement) }
        
        sqlite3_bind_text(statement, 1, query, -1, nil)
        sqlite3_bind_int(statement, 2, Int32(topM))
        
        var results: [(Int, Float)] = []
        while sqlite3_step(statement) == SQLITE_ROW {
            let id = Int(sqlite3_column_int(statement, 0))
            let score = Float(sqlite3_column_double(statement, 1))
            results.append((id, abs(score))) // BM25 returns negative scores, take absolute
        }
        
        return results
    }
    
    private func performSimpleLexicalSearch(query: String, topM: Int) throws -> [(Int, Float)] {
        guard let db = db else {
            throw RetrievalError.databaseNotOpen
        }
        
        let sql = """
        SELECT id, 1.0 as score
        FROM chunks
        WHERE text LIKE ?
        LIMIT ?
        """
        
        var statement: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &statement, nil) == SQLITE_OK else {
            throw RetrievalError.sqlError("Failed to prepare simple lexical search")
        }
        
        defer { sqlite3_finalize(statement) }
        
        let searchTerm = "%\(query)%"
        sqlite3_bind_text(statement, 1, searchTerm, -1, nil)
        sqlite3_bind_int(statement, 2, Int32(topM))
        
        var results: [(Int, Float)] = []
        while sqlite3_step(statement) == SQLITE_ROW {
            let id = Int(sqlite3_column_int(statement, 0))
            let score = Float(sqlite3_column_double(statement, 1))
            results.append((id, score))
        }
        
        return results
    }
    
    private func fetchChunkTexts(ids: [Int]) throws -> [String] {
        guard let db = db else {
            throw RetrievalError.databaseNotOpen
        }
        
        let placeholders = ids.map { _ in "?" }.joined(separator: ",")
        let sql = "SELECT text FROM chunks WHERE id IN (\(placeholders))"
        
        var statement: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &statement, nil) == SQLITE_OK else {
            throw RetrievalError.sqlError("Failed to prepare fetch statement")
        }
        
        defer { sqlite3_finalize(statement) }
        
        for (index, id) in ids.enumerated() {
            sqlite3_bind_int(statement, Int32(index + 1), Int32(id))
        }
        
        var texts: [String] = []
        while sqlite3_step(statement) == SQLITE_ROW {
            if let textPtr = sqlite3_column_text(statement, 0) {
                let text = String(cString: textPtr)
                texts.append(text)
            }
        }
        
        return texts
    }
    
    private func computeEmbedding(_ text: String) throws -> [Float] {
        return try embeddingComputer.computeEmbedding(for: text)
    }
    
    private func searchFAISS(embedding: [Float], topM: Int) throws -> [(Int, Float)] {
        return try faissBridge.search(embedding: embedding, topK: topM)
    }
}

enum RRF {
    static func fuse(
        dense: [(Int, Float)],
        lex: [(Int, Float)],
        alpha: Double,
        k: Int
    ) -> [(id: Int, score: Double)] {
        
        func createRankMap(_ results: [(Int, Float)]) -> [Int: Int] {
            return Dictionary(uniqueKeysWithValues: results.enumerated().map { (index, item) in
                (item.0, index + 1)
            })
        }
        
        let denseRanks = createRankMap(dense)
        let lexRanks = createRankMap(lex)
        
        let allIds = Set(dense.map { $0.0 }).union(Set(lex.map { $0.0 }))
        
        let fusedResults = allIds.map { id in
            let denseScore = alpha * (1.0 / Double(k + (denseRanks[id] ?? 1_000_000)))
            let lexScore = (1.0 - alpha) * (1.0 / Double(k + (lexRanks[id] ?? 1_000_000)))
            return (id: id, score: denseScore + lexScore)
        }
        
        return fusedResults.sorted { $0.score > $1.score }
    }
}

enum RetrievalError: Error {
    case databaseNotOpen
    case sqlError(String)
    case embeddingError(String)
    case faissError(String)
}