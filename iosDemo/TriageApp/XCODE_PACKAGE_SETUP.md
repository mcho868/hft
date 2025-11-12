# Xcode Package Dependencies Setup Guide

## The Problem
You're getting "No such module 'MLX'" because the MLX Swift packages haven't been added to your Xcode project yet.

## Solution: Add Packages in Xcode (NOT Package.swift)

Your project uses `.xcodeproj` format, which requires adding dependencies through Xcode's UI, not a `Package.swift` file.

---

## Step-by-Step Instructions

### 1. Open Your Project
Open `TriageApp.xcodeproj` in Xcode

### 2. Navigate to Package Dependencies
1. Click on **TriageApp** (blue project icon) in the Project Navigator
2. Make sure **TriageApp** target is selected
3. Click the **"Package Dependencies"** tab at the top

### 3. Add MLX-Swift Package

Click the **"+"** button at the bottom of the package list:

**Package URL:** `https://github.com/ml-explore/mlx-swift`

**Settings:**
- Dependency Rule: **Up to Next Major Version**
- Version: **0.20.0**

**Libraries to Add (check these):**
- ✅ MLX
- ✅ MLXNN
- ✅ MLXOptimizers
- ✅ MLXRandom

Click **"Add Package"**

---

### 4. Add MLX-Swift-Examples Package

Click **"+"** again:

**Package URL:** `https://github.com/ml-explore/mlx-swift-examples`

**Settings:**
- Dependency Rule: **Up to Next Major Version**
- Version: **0.20.0**

**Libraries to Add (check these):**
- ✅ MLXLLM
- ✅ MLXLMCommon

Click **"Add Package"**

---

### 5. Add Swift Transformers Package

Click **"+"** again:

**Package URL:** `https://github.com/huggingface/swift-transformers`

**Settings:**
- Dependency Rule: **Up to Next Major Version**
- Version: **0.1.0**

**Libraries to Add (check these):**
- ✅ Transformers

Click **"Add Package"**

---

## What Xcode Will Do

After adding each package, Xcode will:
1. Download the package source code
2. Resolve dependencies
3. Build the packages
4. Make them available for import

**This may take 5-10 minutes** depending on your internet connection.

---

## Verify Installation

Once all packages are added, you should see them listed in the "Package Dependencies" tab:

```
📦 mlx-swift (0.20.0)
   └─ MLX, MLXNN, MLXOptimizers, MLXRandom

📦 mlx-swift-examples (0.20.0)
   └─ MLXLLM, MLXLMCommon

📦 swift-transformers (0.1.0)
   └─ Transformers
```

---

## After Packages Are Added

1. **Clean Build Folder**
   - Product → Clean Build Folder (⌘⇧K)

2. **Build Project**
   - Product → Build (⌘B)

3. The "No such module 'MLX'" error should be resolved ✅

---

## Troubleshooting

### If packages fail to download:

1. **Reset Package Caches:**
   - File → Packages → Reset Package Caches

2. **Update to Latest Versions:**
   - File → Packages → Update to Latest Package Versions

3. **Check Internet Connection:**
   - Make sure you can access github.com

### If build still fails:

1. **Check minimum deployment target:**
   - Go to project settings → General
   - Make sure "iOS Deployment Target" is **iOS 16.0 or higher**

2. **Check Swift version:**
   - Build Settings → Swift Language Version
   - Should be **Swift 5.9 or higher**

---

## File Structure After Setup

Your project structure should look like:

```
TriageApp/
├── TriageApp.xcodeproj/
├── TriageApp/
│   ├── MLXGenerator.swift       ✅ (imports MLX modules)
│   ├── MLXModels/                ✅ (model files)
│   ├── ContentView.swift
│   ├── TriageViewModel.swift
│   └── ... (other files)
└── (no Package.swift needed)
```

---

## Alternative: If You Want to Use Swift Package Manager

If you prefer command-line builds with `swift build`, you would need to:
1. Convert your Xcode project to a pure Swift Package
2. Use the `Package.swift` file
3. Build with `swift build` instead of Xcode

**But for iOS apps, using Xcode with .xcodeproj is recommended.**

---

## Next Steps After Packages Are Added

Once the build succeeds:
1. Follow the rest of the migration guide in `MLX_MIGRATION_GUIDE.md`
2. Add model files to the Xcode project
3. Test on a physical iOS device

---

**Current Status:**
- ✅ Removed duplicate "MLXGenerator 2.swift"
- ✅ Removed unused "Package.swift"
- ⏳ **NEXT:** Add packages in Xcode using the steps above
