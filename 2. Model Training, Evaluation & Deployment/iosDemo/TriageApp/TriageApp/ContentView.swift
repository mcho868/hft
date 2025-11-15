import SwiftUI

struct ContentView: View {
    @State private var age = ""
    @State private var gender = "Female"
    @State private var symptoms = ""
    @State private var ragEnabled = false  // RAG off by default
    @State private var isRunning = false
    @State private var triage = ""
    @State private var nextSteps = ""
    @State private var reasoning = ""
    @State private var latencyMs = 0
    @State private var memMB = 0
    @State private var rawOutput = ""
    @State private var showRawOutput = false
    @FocusState private var isInputFocused: Bool

    @StateObject private var vm = TriageViewModel()

    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 16) {
                Text("Medical Triage Demo")
                    .font(.title)
                    .fontWeight(.bold)
                    .onAppear {
                        Task {
                            await vm.initializeModel()
                        }
                    }
                
                VStack(spacing: 12) {
                    // Model Selector Dropdown
                    HStack {
                        Text("AI Model:")
                            .font(.headline)

                        Spacer()

                        Menu {
                            ForEach(MLXGenerator.ModelType.allCases, id: \.self) { modelType in
                                Button(action: {
                                    Task {
                                        await vm.switchModel(to: modelType)
                                    }
                                }) {
                                    HStack {
                                        Text(modelType.displayName)
                                        if vm.selectedModelType == modelType {
                                            Spacer()
                                            Image(systemName: "checkmark")
                                        }
                                    }
                                }
                            }
                        } label: {
                            HStack {
                                Text(vm.selectedModelType.displayName)
                                    .foregroundColor(.primary)
                                Image(systemName: "chevron.down")
                                    .foregroundColor(.secondary)
                                    .font(.caption)
                            }
                            .padding(.horizontal, 12)
                            .padding(.vertical, 8)
                            .background(Color.gray.opacity(0.1))
                            .cornerRadius(8)
                        }
                    }
                    
                    if vm.modelSwitchInProgress {
                        HStack {
                            ProgressView()
                                .scaleEffect(0.8)
                            Text("Switching model...")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    }
                    
                    Toggle("Use Retrieval (RAG)", isOn: $ragEnabled)
                        .toggleStyle(SwitchToggleStyle())

                    TextField("Age (years)", text: $age)
                        .keyboardType(.numberPad)
                        .textFieldStyle(RoundedBorderTextFieldStyle())
                        .focused($isInputFocused)

                    TextField("Gender", text: $gender)
                        .textFieldStyle(RoundedBorderTextFieldStyle())
                        .focused($isInputFocused)

                    TextField("Describe your symptoms…", text: $symptoms, axis: .vertical)
                        .lineLimit(4...8)
                        .textFieldStyle(RoundedBorderTextFieldStyle())
                        .focused($isInputFocused)
                }
                .padding(.horizontal)

                Button(action: {
                    // Dismiss keyboard
                    isInputFocused = false

                    Task {
                        await runTriage()
                    }
                }) {
                    Text(isRunning ? "Running…" : "Run Triage")
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(isRunning || symptoms.isEmpty ? Color.gray : Color.blue)
                        .foregroundColor(.white)
                        .cornerRadius(8)
                }
                .disabled(isRunning || symptoms.isEmpty)
                .padding(.horizontal)

                if !triage.isEmpty {
                    Divider()
                        .padding(.vertical, 8)
                    
                    VStack(spacing: 16) {
                        // Large triage decision display
                        VStack(spacing: 8) {
                            Text("Triage Decision")
                                .font(.headline)
                                .foregroundStyle(.secondary)
                            
                            Text(triage)
                                .font(.system(size: 48, weight: .bold, design: .rounded))
                                .foregroundColor(triageColor(triage))
                                .padding(.horizontal, 24)
                                .padding(.vertical, 16)
                                .background(triageColor(triage).opacity(0.1))
                                .overlay(
                                    RoundedRectangle(cornerRadius: 12)
                                        .stroke(triageColor(triage), lineWidth: 2)
                                )
                                .cornerRadius(12)
                        }
                        
                        // Show interpretation
                        Text(triageInterpretation(triage))
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                            .multilineTextAlignment(.center)
                            .padding(.horizontal)

                        // Show next steps first, then reasoning
                        if !nextSteps.isEmpty && !nextSteps.contains("<") && nextSteps != "UNKNOWN" {
                            VStack(alignment: .leading, spacing: 6) {
                                Text("Next Steps:")
                                    .fontWeight(.semibold)
                                Text(nextSteps)
                                    .foregroundStyle(.primary)
                            }
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding(.horizontal)
                        }

                        if !reasoning.isEmpty && !reasoning.contains("<") && reasoning != "UNKNOWN" {
                            VStack(alignment: .leading, spacing: 6) {
                                Text("Reasoning:")
                                    .fontWeight(.semibold)
                                Text(reasoning)
                                    .foregroundStyle(.secondary)
                            }
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding(.horizontal)
                        }

                        // Raw output toggle button
                        Button(action: {
                            showRawOutput.toggle()
                        }) {
                            HStack {
                                Image(systemName: showRawOutput ? "chevron.down" : "chevron.right")
                                    .font(.caption)
                                Text("Raw Model Output")
                                    .font(.caption)
                                    .fontWeight(.medium)
                            }
                            .foregroundColor(.blue)
                            .padding(.vertical, 4)
                        }
                        .padding(.horizontal)

                        // Raw output display
                        if showRawOutput && !rawOutput.isEmpty {
                            VStack(alignment: .leading, spacing: 6) {
                                Text(rawOutput)
                                    .font(.system(.caption, design: .monospaced))
                                    .foregroundStyle(.secondary)
                                    .padding(12)
                                    .background(Color.gray.opacity(0.1))
                                    .cornerRadius(8)
                            }
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding(.horizontal)
                        }
                    }
                }

                Text("Latency: \(latencyMs) ms • Memory: \(memMB) MB")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .padding(.top, 8)
                
                Text("⚠️ Research demo – not for clinical use")
                    .font(.caption)
                    .foregroundStyle(.red)
                    .fontWeight(.medium)
                    .padding(.bottom)
            }
            .padding()
        }
        .navigationBarHidden(true)
        }
        .navigationViewStyle(StackNavigationViewStyle())
    }
    
    private func runTriage() async {
        isRunning = true
        defer { isRunning = false }

        do {
            let query = "Kia ora, I'm a \(age)-year-old \(gender). \(symptoms)"
            print("🔥 DEBUG: About to run triage with query: \(query)")
            let result = try await vm.run(query: query, rag: ragEnabled)

            triage = result.triage
            nextSteps = result.nextSteps
            reasoning = result.reasoning
            latencyMs = result.latencyMs
            memMB = result.peakMemMB
            rawOutput = result.rawOutput
            print("🔥 DEBUG: Triage completed successfully")
        } catch {
            print("🔥 DEBUG: Triage failed with error: \(error)")
            triage = "ERROR"
            nextSteps = "CoreML Error: \(error.localizedDescription)"
            reasoning = "Error Type: \(type(of: error)) - \(error)"
            latencyMs = 0
            memMB = 0
        }
    }
    
    private func triageColor(_ triage: String) -> Color {
        switch triage.uppercased() {
        case "ED":
            return .red
        case "GP":
            return .orange
        case "HOME":
            return .green
        default:
            return .gray
        }
    }
    
    private func triageInterpretation(_ triage: String) -> String {
        switch triage.uppercased() {
        case "ED":
            return "Emergency Department - Seek immediate medical attention"
        case "GP":
            return "General Practitioner - Schedule an appointment with your doctor"
        case "HOME":
            return "Home Care - Rest and monitor symptoms"
        case "ERROR":
            return "Unable to process - Please try again"
        default:
            return "Processing result..."
        }
    }
}

#Preview {
    ContentView()
}