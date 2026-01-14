# Homebrew Tap for MLX Manager

This directory contains the Homebrew formula for MLX Manager.

## Setup Instructions

To make MLX Manager installable via `brew install tumma72/tap/mlx-manager`:

### 1. Create the tap repository

Create a new **public** GitHub repository named `homebrew-tap`:
- Go to https://github.com/new
- Name: `homebrew-tap`
- Make it **Public**
- Create repository

### 2. Add the formula

```bash
# Clone your new tap repository
git clone https://github.com/tumma72/homebrew-tap.git
cd homebrew-tap

# Create the Formula directory and copy the formula
mkdir -p Formula
cp /path/to/mlx-manager/homebrew/Formula/mlx-manager.rb Formula/

# Commit and push
git add Formula/mlx-manager.rb
git commit -m "Add mlx-manager formula"
git push
```

### 3. Test the installation

```bash
# Tap your repository
brew tap tumma72/tap

# Install mlx-manager
brew install mlx-manager
```

## Updating the Formula

When releasing a new version:

1. Get the new SHA256:
   ```bash
   curl -sL "https://pypi.org/pypi/mlx-manager/json" | python3 -c "
   import sys, json
   data = json.load(sys.stdin)
   sdist = next(u for u in data['urls'] if u['packagetype'] == 'sdist')
   print(f\"Version: {data['info']['version']}\")
   print(f\"SHA256: {sdist['digests']['sha256']}\")
   print(f\"URL: {sdist['url']}\")
   "
   ```

2. Update `Formula/mlx-manager.rb` with the new URL and SHA256

3. Commit and push to the tap repository

## Users

Once set up, users can install with:

```bash
brew install tumma72/tap/mlx-manager
```

Or explicitly tap first:

```bash
brew tap tumma72/tap
brew install mlx-manager
```
