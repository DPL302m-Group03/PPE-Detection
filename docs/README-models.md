# Model weights management

This repository contains model weight files under `PPE-dashboard/models/` which are large binary files (.pt).

## Recommended approach

### 1) Use Git LFS for model files

- Install git-lfs: https://git-lfs.github.com/
- Add tracking (already added in `.gitattributes`):

```bash
git lfs install
git lfs track "PPE-dashboard/models/**/*.pt"
```

- If you want to move existing .pt files into LFS (rewrites history):

```bash
git lfs migrate import --include="PPE-dashboard/models/**/*.pt"
git push --force origin main
```

- If you do NOT want to rewrite history, still commit the `.gitattributes` and new files added later will be tracked by LFS.

### 2) Alternative: Host weights externally

- Publish large weights as GitHub Releases, S3, or Google Drive and provide a small download script `scripts/download_models.sh`.

### 3) Notes

- `.gitattributes` is present at repository root to track `PPE-dashboard/models/**/*.pt` with Git LFS.
- If you need help migrating the history, ask and I can provide the exact sequence and help coordinate.
