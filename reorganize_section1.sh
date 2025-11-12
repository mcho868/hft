#!/bin/bash
#
# Project Reorganization Script - Section 1: Knowledge Base & Dataset Construction
# Stages 1-4: Web Scraping, Dataset Prep, Chunking, Index Building
#
# Usage: bash reorganize_section1.sh [--dry-run]
#

set -e  # Exit on error

PROJECT_ROOT="/Users/choemanseung/789/hft"
BACKUP_ROOT="$PROJECT_ROOT/backup_bin"

# Parse arguments
DRY_RUN=false
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "🔍 DRY RUN MODE - No files will be moved"
    echo ""
fi

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to safely move files
safe_move() {
    local source="$1"
    local dest_dir="$2"
    local file_name=$(basename "$source")

    if [[ ! -e "$source" ]]; then
        echo -e "${YELLOW}⚠️  SKIP: $source (does not exist)${NC}"
        return 0
    fi

    if [[ $DRY_RUN == true ]]; then
        echo -e "${BLUE}[DRY RUN]${NC} Would move: $source → $dest_dir/$file_name"
        return 0
    fi

    # Create destination directory
    mkdir -p "$dest_dir"

    # Move file
    if mv "$source" "$dest_dir/"; then
        echo -e "${GREEN}✓${NC} Moved: $file_name → ${dest_dir#$PROJECT_ROOT/}"
    else
        echo -e "${RED}✗${NC} Failed to move: $source"
        return 1
    fi
}

# Function to delete files
safe_delete() {
    local target="$1"

    if [[ ! -e "$target" ]]; then
        echo -e "${YELLOW}⚠️  SKIP: $target (does not exist)${NC}"
        return 0
    fi

    if [[ $DRY_RUN == true ]]; then
        echo -e "${BLUE}[DRY RUN]${NC} Would delete: $target"
        return 0
    fi

    if rm -rf "$target"; then
        echo -e "${GREEN}✓${NC} Deleted: ${target#$PROJECT_ROOT/}"
    else
        echo -e "${RED}✗${NC} Failed to delete: $target"
        return 1
    fi
}

echo "═══════════════════════════════════════════════════════════"
echo "  SECTION 1 REORGANIZATION: Knowledge Base & Dataset       "
echo "═══════════════════════════════════════════════════════════"
echo ""

# Create backup directory structure
echo "📁 Creating backup directory structure..."
BACKUP_DIRS=(
    "$BACKUP_ROOT/data_quality_checks"
    "$BACKUP_ROOT/dataset_analysis"
    "$BACKUP_ROOT/dataset_tools_old"
    "$BACKUP_ROOT/chunking_tools"
    "$BACKUP_ROOT/chunking_old"
    "$BACKUP_ROOT/retrieval_old"
    "$BACKUP_ROOT/demos"
    "$BACKUP_ROOT/samples"
    "$BACKUP_ROOT/old_datasets"
    "$BACKUP_ROOT/RAGdata_v1"
    "$BACKUP_ROOT/logs"
)

for dir in "${BACKUP_DIRS[@]}"; do
    if [[ $DRY_RUN == false ]]; then
        mkdir -p "$dir"
        echo -e "${GREEN}✓${NC} Created: ${dir#$PROJECT_ROOT/}"
    else
        echo -e "${BLUE}[DRY RUN]${NC} Would create: ${dir#$PROJECT_ROOT/}"
    fi
done
echo ""

# Stage 1: Web Scraping - Backup data quality check scripts
echo "───────────────────────────────────────────────────────────"
echo "Stage 1: Web Scraping - Cleaning up debug scripts"
echo "───────────────────────────────────────────────────────────"

safe_move "$PROJECT_ROOT/check_healthify_consistency.py" "$BACKUP_ROOT/data_quality_checks"
safe_move "$PROJECT_ROOT/check_healthify_files.py" "$BACKUP_ROOT/data_quality_checks"
safe_move "$PROJECT_ROOT/analyze_healthify_duplicates.py" "$BACKUP_ROOT/data_quality_checks"
safe_move "$PROJECT_ROOT/check_source_document.py" "$BACKUP_ROOT/data_quality_checks"
safe_move "$PROJECT_ROOT/compare_mayo_ids.py" "$BACKUP_ROOT/data_quality_checks"
safe_move "$PROJECT_ROOT/missing_sources.py" "$BACKUP_ROOT/data_quality_checks"
echo ""

# Backup old RAGdata directory
echo "📦 Backing up old RAGdata (v1)..."
if [[ -d "$PROJECT_ROOT/RAGdata" ]]; then
    if [[ $DRY_RUN == false ]]; then
        if cp -r "$PROJECT_ROOT/RAGdata" "$BACKUP_ROOT/RAGdata_v1/"; then
            rm -rf "$PROJECT_ROOT/RAGdata"
            echo -e "${GREEN}✓${NC} Backed up and removed: RAGdata/"
        else
            echo -e "${RED}✗${NC} Failed to backup RAGdata/"
        fi
    else
        echo -e "${BLUE}[DRY RUN]${NC} Would backup: RAGdata/ → backup_bin/RAGdata_v1/"
    fi
else
    echo -e "${YELLOW}⚠️  SKIP: RAGdata/ (does not exist)${NC}"
fi
echo ""

# Stage 2: Dataset Preparation - Backup analysis scripts
echo "───────────────────────────────────────────────────────────"
echo "Stage 2: Dataset Preparation - Cleaning up utilities"
echo "───────────────────────────────────────────────────────────"

# Root directory scripts
safe_move "$PROJECT_ROOT/analyze_discrepancy.py" "$BACKUP_ROOT/dataset_analysis"
safe_move "$PROJECT_ROOT/analyze_symptoms.py" "$BACKUP_ROOT/dataset_analysis"
safe_move "$PROJECT_ROOT/debug_reasoning_issue.py" "$BACKUP_ROOT/dataset_analysis"
safe_move "$PROJECT_ROOT/dataset_transformer.py" "$BACKUP_ROOT/dataset_tools_old"
safe_move "$PROJECT_ROOT/redistribute_data.py" "$BACKUP_ROOT/dataset_tools_old"
safe_move "$PROJECT_ROOT/clean_progress_file.py" "$BACKUP_ROOT/dataset_tools_old"
echo ""

# Preparing dataset directory scripts
echo "📁 Cleaning 'preparing dataset/' directory..."
safe_move "$PROJECT_ROOT/preparing dataset/compare_datasets.py" "$BACKUP_ROOT/dataset_analysis"
safe_move "$PROJECT_ROOT/preparing dataset/create_reasoning_dataset.py" "$BACKUP_ROOT/dataset_tools_old"
safe_move "$PROJECT_ROOT/preparing dataset/dataset_load.py" "$BACKUP_ROOT/dataset_analysis"
safe_move "$PROJECT_ROOT/preparing dataset/prepare_simple_dataset.py" "$BACKUP_ROOT/dataset_tools_old"
echo ""

# Backup old triage dialogues MLX format
echo "📦 Backing up old triage dialogues (dirty prompts)..."
if [[ -d "$PROJECT_ROOT/Final_dataset/triage_dialogues_mlx" ]]; then
    if [[ $DRY_RUN == false ]]; then
        if cp -r "$PROJECT_ROOT/Final_dataset/triage_dialogues_mlx" "$BACKUP_ROOT/old_datasets/"; then
            rm -rf "$PROJECT_ROOT/Final_dataset/triage_dialogues_mlx"
            echo -e "${GREEN}✓${NC} Backed up and removed: Final_dataset/triage_dialogues_mlx/"
        else
            echo -e "${RED}✗${NC} Failed to backup triage_dialogues_mlx/"
        fi
    else
        echo -e "${BLUE}[DRY RUN]${NC} Would backup: Final_dataset/triage_dialogues_mlx/"
    fi
else
    echo -e "${YELLOW}⚠️  SKIP: triage_dialogues_mlx/ (does not exist)${NC}"
fi
echo ""

# Backup log file
safe_move "$PROJECT_ROOT/ai_filtering_detailed_log.txt" "$BACKUP_ROOT/logs"
echo ""

# Stage 3: Document Chunking - Backup old chunkers
echo "───────────────────────────────────────────────────────────"
echo "Stage 3: Document Chunking - Cleaning up old scripts"
echo "───────────────────────────────────────────────────────────"

safe_move "$PROJECT_ROOT/agent_chunk_deduplicator.py" "$BACKUP_ROOT/chunking_tools"
safe_move "$PROJECT_ROOT/simple_paragraphchunk_maker_from_deduplicated.py" "$BACKUP_ROOT/chunking_old"
safe_move "$PROJECT_ROOT/medical_file_filter.py" "$BACKUP_ROOT/chunking_tools"
echo ""

# Delete temporary progress files from root
echo "🗑️  Removing temporary progress files from root..."
safe_delete "$PROJECT_ROOT/agent_chunking_progress_nhs_agent_mlx_gemma3_4b_1_you_are_an_expert_me.json"
safe_delete "$PROJECT_ROOT/agent_chunking_progress_nhs_agent_mlx_gemma3_4b_2_analyze_the_followin.json"
safe_delete "$PROJECT_ROOT/agent_chunking_progress_nhs_agent_mlx_gemma3_4b_3_deconstruct_the_foll.json"
echo ""

# Delete large contextual progress files from backup_bin (500MB, regenerable)
echo "🗑️  Removing large temporary contextual progress files..."
if [[ -d "$BACKUP_ROOT" ]]; then
    find "$BACKUP_ROOT" -name "contextual_chunking_progress_*.json" -type f | while read file; do
        safe_delete "$file"
    done
fi
echo ""

# Stage 4: Vector Index Building - Backup old retrieval scripts
echo "───────────────────────────────────────────────────────────"
echo "Stage 4: Vector Index Building - Cleaning up old retrieval"
echo "───────────────────────────────────────────────────────────"

safe_move "$PROJECT_ROOT/basic_retrieval_demo.py" "$BACKUP_ROOT/demos"
safe_move "$PROJECT_ROOT/contextual_retrieval_demo.py" "$BACKUP_ROOT/demos"
safe_move "$PROJECT_ROOT/retrieval_function.py" "$BACKUP_ROOT/retrieval_old"
safe_move "$PROJECT_ROOT/retrieval_function_v2.py" "$BACKUP_ROOT/retrieval_old"
safe_move "$PROJECT_ROOT/retrieval_testv2.py" "$BACKUP_ROOT/retrieval_old"
safe_move "$PROJECT_ROOT/rag_chat.py" "$BACKUP_ROOT/demos"
echo ""

# Backup sample directory
if [[ -d "$PROJECT_ROOT/contextual_retrieval _sample" ]]; then
    if [[ $DRY_RUN == false ]]; then
        if cp -r "$PROJECT_ROOT/contextual_retrieval _sample" "$BACKUP_ROOT/samples/"; then
            rm -rf "$PROJECT_ROOT/contextual_retrieval _sample"
            echo -e "${GREEN}✓${NC} Backed up and removed: contextual_retrieval _sample/"
        else
            echo -e "${RED}✗${NC} Failed to backup contextual_retrieval _sample/"
        fi
    else
        echo -e "${BLUE}[DRY RUN]${NC} Would backup: contextual_retrieval _sample/"
    fi
else
    echo -e "${YELLOW}⚠️  SKIP: contextual_retrieval _sample/ (does not exist)${NC}"
fi
echo ""

# Summary
echo "═══════════════════════════════════════════════════════════"
echo "  REORGANIZATION COMPLETE                                  "
echo "═══════════════════════════════════════════════════════════"
echo ""

if [[ $DRY_RUN == true ]]; then
    echo -e "${BLUE}This was a DRY RUN. No files were actually moved.${NC}"
    echo "Run without --dry-run to execute the reorganization."
else
    echo -e "${GREEN}✓ Section 1 reorganization completed successfully!${NC}"
    echo ""
    echo "📁 Kept directories:"
    echo "   - webscrapers/              (all scraper code)"
    echo "   - RAGdatav3/               (source data + chunking scripts)"
    echo "   - RAGdatav4/               (chunks + indices)"
    echo "   - Final_dataset/           (production training data)"
    echo "   - preparing dataset/       (core dataset scripts only)"
    echo ""
    echo "📦 Backed up to:"
    echo "   - backup_bin/data_quality_checks/    (6 files)"
    echo "   - backup_bin/dataset_analysis/       (4 files)"
    echo "   - backup_bin/dataset_tools_old/      (5 files)"
    echo "   - backup_bin/chunking_tools/         (2 files)"
    echo "   - backup_bin/chunking_old/           (1 file)"
    echo "   - backup_bin/retrieval_old/          (3 files)"
    echo "   - backup_bin/demos/                  (3 files)"
    echo "   - backup_bin/samples/                (1 dir)"
    echo "   - backup_bin/old_datasets/           (1 dir)"
    echo "   - backup_bin/RAGdata_v1/             (1 dir)"
    echo "   - backup_bin/logs/                   (1 file)"
    echo ""
    echo "🗑️  Deleted:"
    echo "   - Temporary progress files (~500MB)"
    echo ""
    echo "Next: Run 'reorganize_section2.sh' for model training cleanup"
fi
echo ""
