# ------------------------------------------------------------
# 🎼 MyMusic Git-Style Project Sync Script (from inside ~/MyMusic)
# ------------------------------------------------------------
# 🧠 Your pyenv environment: harmonix (already set via .python-version)
# 📁 Assumes you're already in: ~/MyMusic
# 📁 Folder structure already exists
# 📦 This only initializes Git and syncs to GitHub
# ------------------------------------------------------------

# Step 1: Initialize Git (if not already)
git init

# Step 2: Ensure .python-version points to harmonix
echo "harmonix" > .python-version

# Step 3: Create .gitignore if missing
cat <<EOF > .gitignore
__pycache__/
*.pyc
*.zip
*.log
*.mid
*.wav
dataset/
dataset_features/
.env
.idea/
.vscode/
EOF

# Step 4: Add README if missing
cat <<EOF > README.md
# 🎼 MyMusic: Genre Transformation & Feature Extraction

Genre pipeline: Classical → Jazz → Hijaz → Salsa → Trance using AI + healing frequencies.

## 📁 Structure
- `scripts/`: Core logic and runners
- `features/`: Stem/MIDI/rhythm modules
- `utils/`: File tools
- `models/`: (Future) genre transformation models
- `dataset/`: Input WAVs
- `dataset_features/`: Processed stems/features per track

## ⚙️ Quickstart
```bash
cd scripts
make install
make run
```
EOF

# Step 5: Git staging + first commit
git add .
git commit -m "🎵 Initial commit: sync local MyMusic structure to Git"

# Step 6: Add GitHub remote (update username + repo)
GITHUB_USER="your-username"
GITHUB_REPO="MyMusic"
git remote add origin git@github.com:$GITHUB_USER/$GITHUB_REPO.git

git branch -M main
git push -u origin main

# ✅ Project synced to GitHub — clean, versioned, and pyenv-aware