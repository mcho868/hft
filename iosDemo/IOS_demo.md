	•	Embeddings: all-MiniLM-L6-v2 (384-d)
	•	Generator: /Users/choemanseung/789/hft/iosDemo/models/SmolLM2-135M_4bit_highcap, /Users/choemanseung/789/hft/iosDemo/models/SmolLM2-135M_4bit_perfsafe, /Users/choemanseung/789/hft/iosDemo/models/SmolLM2-135M_4bit_highcap + contextual RAG
	•	Retrieval: FAISS (IndexFlatIP, cosine) + BM25 with RRF fusion
	•	UI: SwiftUI, single screen, toggle for RAG on/off, inputs for Name, Age, Gender, Symptoms, output shows Triage and Next steps

Architecture at a glance

[SwiftUI View]
   │
   └──> ViewModel (Pipeline)
         1) Collect inputs (name/age/gender/symptoms, RAG toggle)
         2) if RAG:
              a. Dense retrieval (FAISS; 384-d, normalized)
              b. Lexical retrieval (BM25/FTS5)
              c. RRF fusion + dedup → top-k chunks
              d. Context packing (length-limited)
            else:
              context=""
         3) PromptBuilder → prompt text (per your format)
         4) MLXGenerator(SmolLM-135M int4) → streamed output
         5) Parser → {triage, next_steps, reasoning}
         6) Show result + citations + metrics (latency, memory)

Offline prep (once, on your Mac)
	1.	Freeze corpus & chunking
Use your structured medical chunking (the winning method). Store:
	•	chunks.sqlite with table chunks(id, source, title, text, url, headings_path)
	•	Optional chunks_fts (FTS5) for lexical search
	2.	Embeddings & FAISS
Compute L2-normalized 384-d embeddings (all-MiniLM-L6-v2).
Build IndexFlatIP and save to faiss.index (CPU is fine). Keep a parallel ids.bin (uint32 ids aligned with FAISS rows).
	3.	BM25 index
Either:
	•	Use SQLite FTS5 and its rank (simple and tiny on iOS), or
	•	Precompute BM25 scores offline per query type (if you prefer a fixed demo)
	4.	Model asset
Export SmolLM-135M (int4) for MLX. If you used LoRA, merge adapters offline so you ship a single compact model. Package tokenizer files.
	5.	Bundle your assets
	•	faiss.index (and ids.bin)
	•	chunks.sqlite (and chunks_fts)
	•	mlx_model/ (weights + tokenizer)
	•	meta.json (RRF weights, top-k, max context chars, etc.)

On-device pipeline (Swift/SwiftUI)

Prompt templates (exactly as you asked)

With RAG:
Patient query: {query}

Context:
{context}

Provide triage decision, next steps, and reasoning:

Without RAG:

Patient query: {query}

Provide triage decision, next steps, and reasoning:

To include demographics cleanly, fold them into {query}:

{Age}-year-old {Gender}. Symptoms: {free-text symptom description}.

That keeps your two-branch format intact while using the Name/Age/Gender inputs you collect.

RRF fusion (semantic + lexical)
	•	Dense: cosine scores via FAISS (IndexFlatIP with normalized vectors).
	•	Lexical: BM25/FTS5 score.
	•	Combine with Reciprocal Rank Fusion (robust, simple):

\text{RRF}(d) = \alpha \cdot \frac{1}{k + \text{rank}\text{dense}(d)} +
(1-\alpha) \cdot \frac{1}{k + \text{rank}\text{lex}(d)}
Use your established weights (e.g., \alpha = 0.7, k=60).
Pick top-k=5 final chunks to fit the SLM’s context limit; strip duplication and trim long chunks to a max context length (e.g., ~6–7k chars worked in your evals).

SwiftUI skeleton (minimal but complete)

import SwiftUI

struct ContentView: View {
    @State private var name = ""
    @State private var age = ""
    @State private var gender = "Female" // or picker
    @State private var symptoms = ""
    @State private var ragEnabled = true
    @State private var isRunning = false
    @State private var triage = ""
    @State private var nextSteps = ""
    @State private var reasoning = ""
    @State private var latencyMs = 0
    @State private var memMB = 0

    let vm = TriageViewModel()

    var body: some View {
        VStack(spacing: 12) {
            Toggle("Use Retrieval (RAG)", isOn: $ragEnabled)
            TextField("Name", text: $name)
            TextField("Age (years)", text: $age).keyboardType(.numberPad)
            TextField("Gender", text: $gender)
            TextField("Describe your symptoms…", text: $symptoms, axis: .vertical)
                .lineLimit(4...8)

            Button(isRunning ? "Running…" : "Run Triage") {
                Task {
                    isRunning = true
                    defer { isRunning = false }
                    let q = "\(age)-year-old \(gender). Symptoms: \(symptoms)"
                    let result = try await vm.run(query: q, rag: ragEnabled)
                    triage = result.triage
                    nextSteps = result.nextSteps
                    reasoning = result.reasoning
                    latencyMs = result.latencyMs
                    memMB = result.peakMemMB
                }
            }.disabled(isRunning || symptoms.isEmpty)

            Divider().padding(.vertical, 4)

            if !triage.isEmpty {
                Text("Triage: \(triage)").font(.headline)
                Text("Next steps: \(nextSteps)")
                Text("Reasoning: \(reasoning)").foregroundStyle(.secondary)
            }

            Text("Latency: \(latencyMs) ms • Peak memory: \(memMB) MB")
                .font(.caption).foregroundStyle(.secondary)
        }
        .padding()
    }
}

ViewModel (pipeline orchestration)

struct TriageResult {
    let triage: String
    let nextSteps: String
    let reasoning: String
    let latencyMs: Int
    let peakMemMB: Int
}

@MainActor
final class TriageViewModel {
    private let retriever = Retriever()        // FAISS + BM25 + RRF
    private let generator = MLXGenerator()     // SmolLM-135M int4
    private let parser = TriageParser()

    func run(query: String, rag: Bool) async throws -> TriageResult {
        let t0 = Date()
        var context = ""
        if rag {
            let fusion = try retriever.search(query: query, topM: 50, topK: 5)
            context = fusion.joined(separator: "\n\n")
        }
        let prompt = PromptBuilder.makePrompt(query: query, context: context, rag: rag)
        let text = try await generator.generate(prompt: prompt,
                                                maxTokens: 256,
                                                temperature: 0.1)
        let (triage, steps, reasoning) = parser.extract(from: text)
        let ms = Int(Date().timeIntervalSince(t0) * 1000)
        let mem = MemoryReporter.peakMB()
        return .init(triage: triage, nextSteps: steps, reasoning: reasoning,
                     latencyMs: ms, peakMemMB: mem)
    }
}

Prompt builder

enum PromptBuilder {
    static func makePrompt(query: String, context: String, rag: Bool) -> String {
        if rag {
            return """
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

FAISS + BM25 (retriever stub)
	•	FAISS: compile as a static lib for iOS; use IndexFlatIP.
	•	BM25: easiest on iOS is SQLite FTS5; query with MATCH and use its rank (or your own BM25 scorer in Swift over stored term stats).

final class Retriever {
    func search(query: String, topM: Int, topK: Int) throws -> [String] {
        let dense = try FaissBridge.topM(query: query, m: topM)        // [(id, score)]
        let lex   = try LexicalIndex.topM(query: query, m: topM)       // [(id, score)]
        let fused = RRF.fuse(dense: dense, lex: lex, alpha: 0.7, k: 60)
        let chunkIds = fused.prefix(topK).map { $0.id }
        return try ChunkStore.fetchTexts(ids: chunkIds)                // from chunks.sqlite
    }
}

enum RRF {
    static func fuse(dense: [(Int, Float)], lex: [(Int, Float)], alpha: Double, k: Int) -> [(id: Int, s: Double)] {
        func ranks(_ a: [(Int, Float)]) -> [Int: Int] {
            Dictionary(uniqueKeysWithValues: a.enumerated().map { ($1.0, $0 + 1) })
        }
        let rd = ranks(dense), rl = ranks(lex)
        let ids = Set(dense.map{$0.0}).union(lex.map{$0.0})
        return ids.map { id in
            let sd = alpha * 1.0 / Double(k + (rd[id] ?? 1_000_000))
            let sl = (1 - alpha) * 1.0 / Double(k + (rl[id] ?? 1_000_000))
            return (id, sd + sl)
        }.sorted { $0.s > $1.s }
    }
}

Parsing triage safely

Your evals showed “UNKNOWN” comes from formatting failures. Nudge the model by appending a tight schema to the prompt (no need to change your two branches):

Output exactly:
Triage decision: {ED|GP|HOME}
Next steps: <2–4 short sentences>
Reasoning: <1–3 short sentences>

Then parse:

final class TriageParser {
    func extract(from text: String) -> (String, String, String) {
        func capture(_ label: String) -> String {
            let pattern = "\(label):\\s*(.*)"
            return RegexUtil.firstLine(matching: pattern, in: text) ?? "UNKNOWN"
        }
        return (capture("Triage decision"),
                capture("Next steps"),
                capture("Reasoning"))
    }
}

Performance & memory
	•	Model: SmolLM-135M int4 typically fits well within your ~400 MB envelope (weights + tokenizer + runtime buffers).
	•	Indexes: keep FAISS + SQLite under ~100–150 MB for the demo by shipping a subset of the corpus (e.g., the most frequent triage topics).
	•	Use memory-mapped files for FAISS and SQLite to avoid large heap allocations.
	•	Stream tokens and reuse preallocated buffers in MLX; keep temperature low (0–0.2) for deterministic output.

Demo flow (what the examiner will see)

The user selects RAG on/off, enters Name, Age, Gender, Symptoms, taps Run.
If RAG on: the app fetches ~5 fused chunks, builds the Context: prompt and generates.
If RAG off: it builds the Patient query: prompt and generates.
The UI shows Triage decision and Next steps (and optional Reasoning), along with latency and peak memory in a small status line.
Hard-code a conspicuous “research demo – not for clinical use” label in the footer.

Build tips that save time
	•	If compiling FAISS for iOS is a hassle, ship only pure semantic retrieval (your results showed nearly equal Pass@5 with ~19× speedup), or use SQLite FTS5 only; both still demonstrate the point.
	•	Merge LoRA offline so you only load a single MLX model at runtime.
	•	Keep your context length fixed (e.g., ~7k chars) to ensure reproducible latency.

