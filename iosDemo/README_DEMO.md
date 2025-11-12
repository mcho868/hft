# iOS Medical Triage Demo - Setup & Run Instructions

## Prerequisites
- Xcode 15.0 or later
- iOS 17.0+ device or simulator
- macOS with Apple Silicon (recommended for best performance)

## Running the App

### Option 1: Xcode (Recommended)
1. **Open the project:**
   ```bash
   cd /Users/choemanseung/789/hft/iosDemo/TriageApp
   open TriageApp.xcodeproj
   ```

2. **Build and run:**
   - Select your target device/simulator in Xcode
   - Press Cmd+R or click the "Play" button
   - The app should build and launch automatically

### Option 2: Command Line
```bash
cd /Users/choemanseung/789/hft/iosDemo/TriageApp
xcodebuild -project TriageApp.xcodeproj -scheme TriageApp -destination 'platform=iOS Simulator,name=iPhone 15' build
```

## Testing the App

1. **Database Test**: Tap "Test Database" to verify SQLite access
2. **Pipeline Test**: Tap "Test Pipeline" to run end-to-end scenarios
3. **Component Test**: Tap "Test Components" to validate individual parts
4. **Manual Test**: Enter patient data and tap "Run Triage"

## Expected Behavior

- **With RAG ON**: Uses vector search + lexical search + RRF fusion
- **With RAG OFF**: Uses only the contextual generation model
- **Performance**: ~500-2000ms latency, ~50-150MB memory usage
- **Results**: Color-coded triage decisions (RED=ED, ORANGE=GP, GREEN=HOME)

## Troubleshooting

- **Database not found**: Check that chunks.sqlite is in app bundle
- **Build errors**: Ensure iOS 17.0+ deployment target
- **Slow performance**: Use device instead of simulator for better performance