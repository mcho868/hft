import Foundation
import SQLite3

class DatabaseHelper {
    static let shared = DatabaseHelper()
    private var db: OpaquePointer?
    
    private init() {}
    
    func getDatabase() -> OpaquePointer? {
        if db != nil {
            return db
        }
        
        let dbPath = getDatabasePath()
        
        // Copy database from bundle to documents directory if it doesn't exist
        if !FileManager.default.fileExists(atPath: dbPath) {
            copyDatabaseFromBundle(to: dbPath)
        }
        
        // Open database
        if sqlite3_open(dbPath, &db) == SQLITE_OK {
            print("✅ Database opened successfully at: \(dbPath)")
            return db
        } else {
            print("❌ Unable to open database: \(String(cString: sqlite3_errmsg(db)))")
            return nil
        }
    }
    
    private func getDatabasePath() -> String {
        let documentsPath = NSSearchPathForDirectoriesInDomains(.documentDirectory, .userDomainMask, true)[0]
        return "\(documentsPath)/chunks.sqlite"
    }
    
    private func copyDatabaseFromBundle(to destinationPath: String) {
        guard let bundlePath = Bundle.main.path(forResource: "chunks", ofType: "sqlite") else {
            print("❌ Database file not found in bundle")
            return
        }
        
        do {
            try FileManager.default.copyItem(atPath: bundlePath, toPath: destinationPath)
            print("✅ Database copied to documents directory")
        } catch {
            print("❌ Failed to copy database: \(error)")
        }
    }
    
    deinit {
        if db != nil {
            sqlite3_close(db)
        }
    }
}