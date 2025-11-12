#!/bin/bash
#
# Fix Dependencies Script - Update file paths before reorganization
# Run this BEFORE reorganize_section1.sh
#

set -e

PROJECT_ROOT="/Users/choemanseung/789/hft"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "═══════════════════════════════════════════════════════════"
echo "  FIXING DEPENDENCIES - Updating File Paths"
echo "═══════════════════════════════════════════════════════════"
echo ""

# Fix 1: Update healthify_scraper.py to use RAGdatav3/healthify instead of RAGdata
echo "📝 Fixing healthify_scraper.py..."
if [ -f "$PROJECT_ROOT/webscrapers/healthify_scraper.py" ]; then
    # Backup original
    cp "$PROJECT_ROOT/webscrapers/healthify_scraper.py" "$PROJECT_ROOT/webscrapers/healthify_scraper.py.bak"

    # Update paths
    sed -i '' "s|'/Users/choemanseung/789/hft/RAGdata/healthify_data'|'/Users/choemanseung/789/hft/RAGdatav3/healthify'|g" "$PROJECT_ROOT/webscrapers/healthify_scraper.py"
    sed -i '' "s|'/Users/choemanseung/789/hft/RAGdata/healthify_urls.json'|'/Users/choemanseung/789/hft/RAGdatav3/healthify_urls.json'|g" "$PROJECT_ROOT/webscrapers/healthify_scraper.py"
    sed -i '' "s|'/Users/choemanseung/789/hft/RAGdata/healthify_existing_urls.json'|'/Users/choemanseung/789/hft/RAGdatav3/healthify_existing_urls.json'|g" "$PROJECT_ROOT/webscrapers/healthify_scraper.py"

    echo -e "${GREEN}✓${NC} Updated healthify_scraper.py"
    echo "   RAGdata/healthify_data → RAGdatav3/healthify"
else
    echo -e "${YELLOW}⚠️  healthify_scraper.py not found${NC}"
fi
echo ""

# Fix 2: Update convert_to_mlx_format.py to output to Final_dataset
echo "📝 Fixing convert_to_mlx_format.py..."
if [ -f "$PROJECT_ROOT/convert_to_mlx_format.py" ]; then
    # Backup original
    cp "$PROJECT_ROOT/convert_to_mlx_format.py" "$PROJECT_ROOT/convert_to_mlx_format.py.bak"

    # Update output path
    sed -i '' 's|base_output_dir = "/Users/choemanseung/789/hft/mlx_dataset/triage_dialogues_mlx"|base_output_dir = "/Users/choemanseung/789/hft/Final_dataset/triage_dialogues_mlx"|g' "$PROJECT_ROOT/convert_to_mlx_format.py"

    echo -e "${GREEN}✓${NC} Updated convert_to_mlx_format.py"
    echo "   mlx_dataset/triage_dialogues_mlx → Final_dataset/triage_dialogues_mlx"
else
    echo -e "${YELLOW}⚠️  convert_to_mlx_format.py not found${NC}"
fi
echo ""

# Fix 3: Update .gitignore to exclude new paths (if needed)
echo "📝 Checking .gitignore..."
if [ -f "$PROJECT_ROOT/.gitignore" ]; then
    # Add patterns if not already present
    if ! grep -q "RAGdatav4/indiv_embeddings" "$PROJECT_ROOT/.gitignore"; then
        echo "" >> "$PROJECT_ROOT/.gitignore"
        echo "# Vector indices (large files)" >> "$PROJECT_ROOT/.gitignore"
        echo "RAGdatav4/indiv_embeddings/*.index" >> "$PROJECT_ROOT/.gitignore"
        echo -e "${GREEN}✓${NC} Updated .gitignore with index file patterns"
    else
        echo -e "${BLUE}ℹ${NC}  .gitignore already has index patterns"
    fi

    if ! grep -q "backup_bin/RAGdata_v1" "$PROJECT_ROOT/.gitignore"; then
        echo "" >> "$PROJECT_ROOT/.gitignore"
        echo "# Backup directories" >> "$PROJECT_ROOT/.gitignore"
        echo "backup_bin/RAGdata_v1/" >> "$PROJECT_ROOT/.gitignore"
        echo "backup_bin/old_datasets/" >> "$PROJECT_ROOT/.gitignore"
        echo -e "${GREEN}✓${NC} Updated .gitignore with backup patterns"
    else
        echo -e "${BLUE}ℹ${NC}  .gitignore already has backup patterns"
    fi
else
    echo -e "${YELLOW}⚠️  .gitignore not found${NC}"
fi
echo ""

# Summary
echo "═══════════════════════════════════════════════════════════"
echo "  DEPENDENCY FIXES COMPLETE"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo -e "${GREEN}✓ Fixed file paths in:${NC}"
echo "  - webscrapers/healthify_scraper.py"
echo "  - convert_to_mlx_format.py"
echo "  - .gitignore"
echo ""
echo "📦 Backup files created:"
echo "  - webscrapers/healthify_scraper.py.bak"
echo "  - convert_to_mlx_format.py.bak"
echo ""
echo -e "${BLUE}Next: Run './reorganize_section1.sh --dry-run' to preview reorganization${NC}"
echo ""
