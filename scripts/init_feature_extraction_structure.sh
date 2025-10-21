#!/bin/bash

# Define root
ROOT="extract_features"
SUBDIRS=("utils" "features")
FILES=(
    "README.md"
    "utils/__init__.py"
    "features/__init__.py"
    "extract_features.py"
    "check_env.py"
)

echo "📁 Initializing $ROOT structure..."

# Create root folder
mkdir -p "$ROOT"

# Create subfolders
for dir in "${SUBDIRS[@]}"; do
    mkdir -p "$ROOT/$dir"
done

# Create base files (only if not exists)
for file in "${FILES[@]}"; do
    file_path="$ROOT/$file"
    if [ ! -f "$file_path" ]; then
        touch "$file_path"
        echo "🆕 Created: $file_path"
    fi
done

# Apply safe default permissions (rwx for user, rx for group/others)
chmod -R 755 "$ROOT"

echo "✅ Folder structure initialized and permissions set for: $ROOT"