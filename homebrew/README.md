# Homebrew Tap for MLX Manager

This directory contains the Homebrew formula for MLX Manager.

## Setup Instructions

To make MLX Manager installable via Homebrew, you need to create a tap repository:

### 1. Create the tap repository

Create a new GitHub repository named `homebrew-tap` under your account:

```bash
# Create the repository on GitHub, then clone it
git clone https://github.com/tumma72/homebrew-tap.git
cd homebrew-tap
mkdir Formula
```

### 2. Copy the formula

Copy `Formula/mlx-manager.rb` from this directory to the tap repository:

```bash
cp /path/to/mlx-manager/homebrew/Formula/mlx-manager.rb Formula/
```

### 3. Update the SHA256

Get the SHA256 of the PyPI package and update the formula:

```bash
# Download and calculate SHA256
curl -sL https://files.pythonhosted.org/packages/source/m/mlx-manager/mlx_manager-1.0.0.tar.gz | shasum -a 256
```

Replace `PLACEHOLDER_SHA256` in the formula with the actual hash.

### 4. Commit and push

```bash
git add Formula/mlx-manager.rb
git commit -m "Add mlx-manager formula"
git push
```

### 5. Test the installation

```bash
brew tap tumma72/tap
brew install mlx-manager
```

## Updating the Formula

When releasing a new version:

1. Update the `url` with the new version
2. Update the `sha256` hash
3. Commit and push to the tap repository

```bash
# Get new SHA256
VERSION=1.0.1
curl -sL "https://files.pythonhosted.org/packages/source/m/mlx-manager/mlx_manager-${VERSION}.tar.gz" | shasum -a 256
```

## Alternative: GitHub Releases

For larger distributions, you can also use GitHub releases:

```ruby
url "https://github.com/tumma72/mlx-manager/releases/download/v#{version}/mlx_manager-#{version}.tar.gz"
```
