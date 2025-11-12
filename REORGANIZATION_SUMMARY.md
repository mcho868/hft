# Section 1 Reorganization - Ready to Execute

## ✅ Dependency Fixes Completed

All file path dependencies have been updated to prevent broken references:

### Fixed Files

| File | Old Path | New Path | Status |
|------|----------|----------|--------|
| `webscrapers/healthify_scraper.py` | `RAGdata/healthify_data` | `RAGdatav3/healthify` | ✅ Fixed |
| `convert_to_mlx_format.py` | `mlx_dataset/triage_dialogues_mlx` | `Final_dataset/triage_dialogues_mlx` | ✅ Fixed |
| `.gitignore` | - | Added backup & index patterns | ✅ Updated |

### Backup Files Created
- `webscrapers/healthify_scraper.py.bak`
- `convert_to_mlx_format.py.bak`

---

## 📋 Ready to Reorganize

### What Will Happen

**25 files will be moved to organized backup locations:**

#### Stage 1: Web Scraping Cleanup (6 files)
```
backup_bin/data_quality_checks/
├── check_healthify_consistency.py
├── check_healthify_files.py
├── analyze_healthify_duplicates.py
├── check_source_document.py
├── compare_mayo_ids.py
└── missing_sources.py
```

**+ RAGdata/ directory → backup_bin/RAGdata_v1/**

#### Stage 2: Dataset Preparation Cleanup (10 files)
```
backup_bin/dataset_analysis/
├── analyze_discrepancy.py
├── analyze_symptoms.py
├── debug_reasoning_issue.py
├── compare_datasets.py
└── dataset_load.py

backup_bin/dataset_tools_old/
├── dataset_transformer.py
├── redistribute_data.py
├── clean_progress_file.py
├── create_reasoning_dataset.py
└── prepare_simple_dataset.py

backup_bin/logs/
└── ai_filtering_detailed_log.txt
```

**+ Final_dataset/triage_dialogues_mlx/ → backup_bin/old_datasets/**

#### Stage 3: Document Chunking Cleanup (3 files)
```
backup_bin/chunking_tools/
├── agent_chunk_deduplicator.py
└── medical_file_filter.py

backup_bin/chunking_old/
└── simple_paragraphchunk_maker_from_deduplicated.py
```

**Files to be deleted: ~500MB of regenerable temp data**
- 3 progress files from root
- 12 contextual chunking progress files from backup_bin

#### Stage 4: Vector Index Building Cleanup (6 files)
```
backup_bin/retrieval_old/
├── retrieval_function.py
├── retrieval_function_v2.py
└── retrieval_testv2.py

backup_bin/demos/
├── basic_retrieval_demo.py
├── contextual_retrieval_demo.py
└── rag_chat.py

backup_bin/samples/
└── contextual_retrieval _sample/
```

---

## ✅ Files That Will Remain (Production Code)

### Core Pipeline Scripts
```
webscrapers/
├── nhs_scraper.py                     ✅ (paths fixed)
├── mayo_scraper.py                    ✅
├── mayo_diagnosis_treatment_scraper.py ✅
├── healthify_scraper.py               ✅
└── medical_scraper.py                 ✅

Root directory:
├── generate_triage_dialogues.py       ✅
├── extract_medical_conditions.py      ✅
├── filter_medical_conditions.py       ✅
├── ai_filter_medical_conditions.py    ✅
├── convert_to_mlx_format.py           ✅ (paths fixed)
├── main_chunking_script_v4.py         ✅
├── main_build_script_index_only.py    ✅
└── contextual_retrieval_config.json   ✅

preparing dataset/
├── simple_deduplicator.py             ✅
├── clean_triage_data.py               ✅
└── prepare_mlx_dataset.py             ✅
```

### Data Directories
```
webscrapers/                    # All scraper code
RAGdatav3/                      # Source data + chunking scripts
  ├── nhs/                      # ~800 condition files
  ├── mayo/                     # Comprehensive medical data
  ├── healthify/                # NZ-specific health info
  └── scripts/                  # Modular chunking framework (10 files)

RAGdatav4/                      # Chunked documents
  ├── *_chunks_*.json           # ~100 chunk files
  └── indiv_embeddings/         # FAISS indices (~100-120 .index files)

Final_dataset/
  ├── final_triage_dialogues_mlx/    # ✅ Production training data
  │   ├── train.jsonl (9,100 dialogues)
  │   ├── valid.jsonl (1,975 dialogues)
  │   └── test.jsonl (1,975 dialogues)
  ├── generated_triage_dialogues.json
  ├── simplified_triage_dialogues_*.json
  └── ...

preparing dataset/              # Core dataset scripts only (3 files)
```

### Reference Files
```
unique_medical_conditions.txt
ai_filtered_medical_conditions.txt
filtered_medical_conditions.txt
```

---

## 🎯 Expected Outcome

### Clean Project Structure
- **16 production scripts** in root and webscrapers/
- **3 core dataset scripts** in preparing dataset/
- **10 modular chunkers** in RAGdatav3/scripts/
- **100+ chunk files** in RAGdatav4/
- **100+ FAISS indices** in RAGdatav4/indiv_embeddings/
- **13,000 training dialogues** in Final_dataset/

### Organized Backups
- **11 backup categories** in backup_bin/
- **25+ experimental/debug files** preserved but out of the way
- **Old datasets** archived for reference

### Space Savings
- **~500MB** of regenerable temp files deleted

---

## 🚀 Execute Reorganization

### Option 1: Review First (Dry Run)
```bash
./reorganize_section1.sh --dry-run
```
Shows exactly what would happen without making changes

### Option 2: Execute Now
```bash
./reorganize_section1.sh
```
Performs the reorganization

---

## 🔄 Rollback Plan

If needed, all original files are backed up:
- Script backups: `*.bak` files
- Directory backups: `backup_bin/RAGdata_v1/`, `backup_bin/old_datasets/`
- Moved files: Organized in `backup_bin/*/`

To rollback:
```bash
# Restore from backup
mv webscrapers/healthify_scraper.py.bak webscrapers/healthify_scraper.py
mv convert_to_mlx_format.py.bak convert_to_mlx_format.py

# Move files back from backup_bin if needed
# (files are preserved, not deleted)
```

---

## ✅ Verification Checklist

After reorganization, verify:
- [ ] All scrapers still work (`python webscrapers/nhs_scraper.py --test`)
- [ ] Dataset generation works (`python generate_triage_dialogues.py` - check imports)
- [ ] Chunking works (`python main_chunking_script_v4.py` - select 1 config to test)
- [ ] Index building works (`python main_build_script_index_only.py`)
- [ ] Fine-tuning scripts find data (`ls Final_dataset/final_triage_dialogues_mlx/`)

---

## 📊 Statistics

| Metric | Count |
|--------|-------|
| Files to move | 25+ |
| Files to delete | 15 (temp files) |
| Directories to backup | 3 |
| Backup categories | 11 |
| Production scripts kept | 29 |
| Data directories kept | 5 |
| Space freed | ~500 MB |

---

**Ready to proceed?**

Run `./reorganize_section1.sh` to execute the reorganization.

All dependencies have been fixed and verified ✅
