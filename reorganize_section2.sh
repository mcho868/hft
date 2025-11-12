#!/bin/bash
#
# Section 2 Reorganization Script
# Organizes Model Training, Evaluation & Deployment (Stages 5-8)
#

set -e

PROJECT_ROOT="/Users/choemanseung/789/hft"
SECTION2_DIR="$PROJECT_ROOT/2. Model Training, Evaluation & Deployment"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

# Parse command line arguments
DRY_RUN=false
if [ "$1" == "--dry-run" ]; then
    DRY_RUN=true
fi

echo "═══════════════════════════════════════════════════════════"
if [ "$DRY_RUN" = true ]; then
    echo "  SECTION 2 REORGANIZATION - DRY RUN"
else
    echo "  SECTION 2 REORGANIZATION"
fi
echo "  Model Training, Evaluation & Deployment (Stages 5-8)"
echo "═══════════════════════════════════════════════════════════"
echo ""

# Function for safe directory moving
safe_move_dir() {
    local source=$1
    local dest_dir=$2
    local name=$(basename "$source")

    if [ ! -d "$source" ] && [ ! -f "$source" ]; then
        echo -e "${YELLOW}⚠️  Not found: $name${NC}"
        return
    fi

    if [ "$DRY_RUN" = true ]; then
        echo -e "${BLUE}[DRY RUN]${NC} Would move: $name → $dest_dir/"
    else
        mkdir -p "$dest_dir"
        mv "$source" "$dest_dir/"
        echo -e "${GREEN}✓${NC} Moved: $name → $dest_dir/"
    fi
}

# Create Section 2 directory structure
if [ "$DRY_RUN" = true ]; then
    echo -e "${BLUE}[DRY RUN]${NC} Would create: $SECTION2_DIR"
else
    echo "📁 Creating Section 2 directory structure..."
    mkdir -p "$SECTION2_DIR"
    echo -e "${GREEN}✓${NC} Created: 2. Model Training, Evaluation & Deployment"
fi

echo ""
echo "───────────────────────────────────────────────────────────"
echo "Stage 5: Model Fine-Tuning"
echo "───────────────────────────────────────────────────────────"

# Move finetune directory
safe_move_dir "$PROJECT_ROOT/finetune" "$SECTION2_DIR"

# Move safety_triage_adapters
safe_move_dir "$PROJECT_ROOT/safety_triage_adapters" "$SECTION2_DIR"

# Move triage_adapters if exists
if [ -d "$PROJECT_ROOT/triage_adapters" ]; then
    safe_move_dir "$PROJECT_ROOT/triage_adapters" "$SECTION2_DIR"
fi

echo ""
echo "───────────────────────────────────────────────────────────"
echo "Stage 6: Retrieval Testing"
echo "───────────────────────────────────────────────────────────"

# Move final_retrieval_testing
safe_move_dir "$PROJECT_ROOT/final_retrieval_testing" "$SECTION2_DIR"

# Move README_retrieval_testing.md if exists
if [ -f "$PROJECT_ROOT/README_retrieval_testing.md" ]; then
    safe_move_dir "$PROJECT_ROOT/README_retrieval_testing.md" "$SECTION2_DIR"
fi

echo ""
echo "───────────────────────────────────────────────────────────"
echo "Stage 7: Generation Validation & Testing"
echo "───────────────────────────────────────────────────────────"

# Move evaluation_framework_final
safe_move_dir "$PROJECT_ROOT/evaluation_framework_final" "$SECTION2_DIR"

# Move evaluation-related files in root
echo ""
echo "📄 Moving evaluation result files..."
for file in "$PROJECT_ROOT"/comprehensive_evaluation_results_*.json \
            "$PROJECT_ROOT"/evaluation_progress_*.json \
            "$PROJECT_ROOT"/evaluation_results_*.json; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        if [ "$DRY_RUN" = true ]; then
            echo -e "${BLUE}[DRY RUN]${NC} Would move: $filename"
        else
            mv "$file" "$SECTION2_DIR/"
            echo -e "${GREEN}✓${NC} Moved: $filename"
        fi
    fi
done

# Move evaluation planning docs
for file in "$PROJECT_ROOT"/evaluation_framework_implementation_plan.md \
            "$PROJECT_ROOT"/evaluation_framework_implemetation_plan.md; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        safe_move_dir "$file" "$SECTION2_DIR"
    fi
done

echo ""
echo "───────────────────────────────────────────────────────────"
echo "Stage 8: iOS Deployment"
echo "───────────────────────────────────────────────────────────"

# Move iosDemo directory
safe_move_dir "$PROJECT_ROOT/iosDemo" "$SECTION2_DIR"

# Move iOS-related documentation
if [ -f "$PROJECT_ROOT/TINFOIL_REST_SETUP.md" ]; then
    safe_move_dir "$PROJECT_ROOT/TINFOIL_REST_SETUP.md" "$SECTION2_DIR"
fi

echo ""
echo "═══════════════════════════════════════════════════════════"
if [ "$DRY_RUN" = true ]; then
    echo "  DRY RUN COMPLETE - No changes made"
else
    echo "  SECTION 2 REORGANIZATION COMPLETE"
fi
echo "═══════════════════════════════════════════════════════════"
echo ""

if [ "$DRY_RUN" = false ]; then
    echo -e "${GREEN}✓ Section 2 reorganization completed successfully!${NC}"
    echo ""
    echo "📁 Organized structure:"
    echo "  2. Model Training, Evaluation & Deployment/"
    echo "  ├── finetune/                        (Stage 5: Fine-tuning scripts)"
    echo "  ├── safety_triage_adapters/          (LoRA adapters)"
    echo "  ├── final_retrieval_testing/         (Stage 6: Retrieval testing)"
    echo "  ├── evaluation_framework_final/      (Stage 7: Generation testing)"
    echo "  ├── iosDemo/                         (Stage 8: iOS app)"
    echo "  ├── comprehensive_evaluation_results_*.json"
    echo "  ├── evaluation_framework_*_plan.md"
    echo "  └── README_retrieval_testing.md"
    echo ""
    echo -e "${BLUE}Next steps:${NC}"
    echo "  1. Verify scripts work from new locations"
    echo "  2. Update any hardcoded paths if needed"
    echo "  3. Update documentation to reflect new structure"
else
    echo -e "${BLUE}To execute reorganization, run:${NC}"
    echo "  ./reorganize_section2.sh"
    echo ""
    echo -e "${YELLOW}This will move:${NC}"
    echo "  - finetune/ → 2. Model Training, Evaluation & Deployment/"
    echo "  - safety_triage_adapters/ → 2. Model Training, Evaluation & Deployment/"
    echo "  - final_retrieval_testing/ → 2. Model Training, Evaluation & Deployment/"
    echo "  - evaluation_framework_final/ → 2. Model Training, Evaluation & Deployment/"
    echo "  - iosDemo/ → 2. Model Training, Evaluation & Deployment/"
    echo "  - evaluation result files → 2. Model Training, Evaluation & Deployment/"
fi

echo ""
