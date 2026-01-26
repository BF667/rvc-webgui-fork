#!/bin/bash
set -e

# Cache directories (can be mounted as volumes)
CACHE_DIR="${CACHE_DIR:-/cache}"
UV_CACHE_DIR="${UV_CACHE_DIR:-$CACHE_DIR/uv}"
MODELS_CACHE_DIR="${MODELS_CACHE_DIR:-$CACHE_DIR/models}"
VENV_CACHE_DIR="${VENV_CACHE_DIR:-$CACHE_DIR/venv}"
PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$CACHE_DIR/python}"

# Create cache directories
mkdir -p "$UV_CACHE_DIR" "$MODELS_CACHE_DIR" "$VENV_CACHE_DIR" "$PYTHON_INSTALL_DIR"

# Set UV environment variables for caching
export UV_CACHE_DIR
export UV_PYTHON_INSTALL_DIR="$PYTHON_INSTALL_DIR"

echo "=== RVC WebUI Docker Startup ==="
echo "Cache directory: $CACHE_DIR"

# ===== Python Environment Setup =====
# Check if venv exists in cache and is complete
VENV_MARKER="$VENV_CACHE_DIR/.venv-complete"

if [ -f "$VENV_MARKER" ] && [ -d "$VENV_CACHE_DIR/.venv" ]; then
    echo "[✓] Virtual environment found in cache"
else
    echo "[↓] Installing Python dependencies (first-time setup)..."
    echo "    This may take several minutes..."
    
    # Sync to cache directory
    cd /app
    UV_PROJECT_ENVIRONMENT="$VENV_CACHE_DIR/.venv" uv sync --locked
    
    # Mark as complete
    touch "$VENV_MARKER"
    echo "[✓] Python dependencies installed"
fi

# Symlink venv to expected location - always recreate to ensure correct target
rm -rf /app/.venv 2>/dev/null || true
ln -sf "$VENV_CACHE_DIR/.venv" /app/.venv

# Export the project environment for uv commands
export UV_PROJECT_ENVIRONMENT="$VENV_CACHE_DIR/.venv"

# ===== Model Download Setup =====
setup_model_cache() {
    local target_dir="$1"
    local cache_subdir="$2"
    local cache_path="$MODELS_CACHE_DIR/$cache_subdir"
    
    mkdir -p "$cache_path"
    
    # Create symlink if target doesn't exist or is not a symlink
    if [ ! -e "$target_dir" ]; then
        ln -sf "$cache_path" "$target_dir"
    elif [ ! -L "$target_dir" ]; then
        # Move existing files to cache if any
        if [ -d "$target_dir" ] && [ "$(ls -A "$target_dir" 2>/dev/null)" ]; then
            cp -rn "$target_dir"/* "$cache_path"/ 2>/dev/null || true
        fi
        rm -rf "$target_dir"
        ln -sf "$cache_path" "$target_dir"
    fi
}

# Setup model cache symlinks
echo "[*] Setting up model cache directories..."
setup_model_cache "/app/assets/hubert" "hubert"
setup_model_cache "/app/assets/rmvpe" "rmvpe"
setup_model_cache "/app/assets/pretrained" "pretrained"
setup_model_cache "/app/assets/pretrained_v2" "pretrained_v2"

# Check if models need to be downloaded
MODELS_MARKER="$MODELS_CACHE_DIR/.models-complete"

if [ -f "$MODELS_MARKER" ]; then
    echo "[✓] Models found in cache"
else
    echo "[↓] Downloading models (first-time setup)..."
    echo "    This may take several minutes..."
    
    uv run tools/download_models.py
    
    # Mark as complete
    touch "$MODELS_MARKER"
    echo "[✓] Models downloaded"
fi

echo "=== Starting RVC WebUI ==="
exec uv run python web_ui.py "$@"
