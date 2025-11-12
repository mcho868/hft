#!/bin/zsh

TARGET_FOLDER="/Users/choemanseung/789/hft/RAGdata/cleaned_healthify_data"

for file in "$TARGET_FOLDER"/*; do
  if [[ -f "$file" ]]; then
    echo "$(basename "$file")"
  fi
done