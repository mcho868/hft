#!/bin/bash
#
# Move Section 1 scripts into organized folder structure
# Section 1: Knowledge Base and Dataset Construction (Stages 1-4)
#

set -e

PROJECT_ROOT="/Users/choemanseung/789/hft"
SECTION1_DIR="$PROJECT_ROOT/1. KB and DS construction"

GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "═══════════════════════════════════════════════════════════"
echo "  ORGANIZING SECTION 1 SCRIPTS"
echo "  Knowledge Base & Dataset Construction (Stages 1-4)"
echo "═══════════════════════════════════════════════════════════"
echo ""

# Create Section 1 directory structure
echo "📁 Creating Section 1 directory structure..."
mkdir -p "$SECTION1_DIR"

# Move Section 1 scripts
echo ""
echo "📦 Moving Section 1 scripts..."
echo ""

# Stage 1: Web Scraping
echo "Stage 1: Web Scraping"
if [ -d "$PROJECT_ROOT/webscrapers" ]; then
    mv "$PROJECT_ROOT/webscrapers" "$SECTION1_DIR/"
    echo -e "${GREEN}✓${NC} Moved: webscrapers/"
fi

# Stage 2: Dataset Preparation
echo ""
echo "Stage 2: Dataset Preparation"
for file in "generate_triage_dialogues.py" "extract_medical_conditions.py" \
            "filter_medical_conditions.py" "ai_filter_medical_conditions.py" \
            "convert_to_mlx_format.py"; do
    if [ -f "$PROJECT_ROOT/$file" ]; then
        mv "$PROJECT_ROOT/$file" "$SECTION1_DIR/"
        echo -e "${GREEN}✓${NC} Moved: $file"
    fi
done

if [ -d "$PROJECT_ROOT/preparing dataset" ]; then
    mv "$PROJECT_ROOT/preparing dataset" "$SECTION1_DIR/"
    echo -e "${GREEN}✓${NC} Moved: preparing dataset/"
fi

# Stage 3: Document Chunking
echo ""
echo "Stage 3: Document Chunking"
if [ -f "$PROJECT_ROOT/main_chunking_script_v4.py" ]; then
    mv "$PROJECT_ROOT/main_chunking_script_v4.py" "$SECTION1_DIR/"
    echo -e "${GREEN}✓${NC} Moved: main_chunking_script_v4.py"
fi

# Stage 4: Vector Index Building
echo ""
echo "Stage 4: Vector Index Building"
if [ -f "$PROJECT_ROOT/main_build_script_index_only.py" ]; then
    mv "$PROJECT_ROOT/main_build_script_index_only.py" "$SECTION1_DIR/"
    echo -e "${GREEN}✓${NC} Moved: main_build_script_index_only.py"
fi

if [ -f "$PROJECT_ROOT/contextual_retrieval_config.json" ]; then
    mv "$PROJECT_ROOT/contextual_retrieval_config.json" "$SECTION1_DIR/"
    echo -e "${GREEN}✓${NC} Moved: contextual_retrieval_config.json"
fi

# Move reference files
echo ""
echo "📄 Moving reference files..."
for file in "unique_medical_conditions.txt" "ai_filtered_medical_conditions.txt" \
            "filtered_medical_conditions.txt"; do
    if [ -f "$PROJECT_ROOT/$file" ]; then
        mv "$PROJECT_ROOT/$file" "$SECTION1_DIR/"
        echo -e "${GREEN}✓${NC} Moved: $file"
    fi
done

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  SECTION 1 ORGANIZATION COMPLETE"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo -e "${GREEN}✓ All Section 1 scripts moved to:${NC}"
echo "  $SECTION1_DIR"
echo ""
echo -e "${BLUE}Structure:${NC}"
echo "  1. KB and DS construction/"
echo "  ├── webscrapers/                     (5 scrapers)"
echo "  ├── preparing dataset/               (3 tools)"
echo "  ├── generate_triage_dialogues.py"
echo "  ├── extract_medical_conditions.py"
echo "  ├── filter_medical_conditions.py"
echo "  ├── ai_filter_medical_conditions.py"
echo "  ├── convert_to_mlx_format.py"
echo "  ├── main_chunking_script_v4.py"
echo "  ├── main_build_script_index_only.py"
echo "  ├── contextual_retrieval_config.json"
echo "  └── *_medical_conditions.txt         (3 files)"
echo ""
echo -e "${BLUE}Next: Update path dependencies in moved scripts${NC}"
echo ""
