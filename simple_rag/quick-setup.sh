#!/bin/bash

# Quick setup script for Simple RAG Demo
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv_rag"

echo "========================================="
echo "🚀 Simple RAG Demo Quick Setup"
echo "========================================="

# Step 0: Check Python version
echo -e "\n[0/4] Checking Python version..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Please install Python 3.10 or higher."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
REQUIRED_VERSION="3.10"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "❌ Python $PYTHON_VERSION found. Python 3.10 or higher is required."
    exit 1
fi

echo "✓ Python $PYTHON_VERSION found"

# Step 1: Create and activate virtual environment
echo -e "\n[1/4] Creating virtual environment 'venv_rag'..."
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

# Step 2: Install dependencies
echo -e "\n[2/4] Installing dependencies..."
pip install --upgrade pip setuptools wheel -q
while IFS= read -r package; do
    [[ -z "$package" || "$package" =~ ^# ]] && continue
    
    if [[ "$package" == "sentence-transformers" ]]; then
        echo "  Installing: $package [800MB+ huge package, might take longer to install]"
    else
        echo "  Installing: $package"
    fi
    
    if ! pip install "$package" -q; then
        echo "❌ Failed to install $package"
        exit 1
    fi
done < "$SCRIPT_DIR/requirements.txt"

# Step 3: Check Ollama installation
echo -e "\n[3/4] Checking Ollama installation..."
if ! command -v ollama &> /dev/null; then
    echo "⚠️  Ollama not found. Please install Ollama from https://ollama.ai"
    echo "    After installation, ensure the Ollama service is running:"
    echo "    - On Linux: ollama serve"
    echo "    - On macOS/Windows: Ollama app should auto-start"
    echo ""
    echo "    Then pull the phi3 model:"
    echo "    ollama pull phi3"
    exit 1
fi

echo "✓ Ollama found"

# Step 4: Check if phi3 model is available
echo -e "\n[4/4] Checking for phi3 model..."
if ! ollama list | grep -q "phi3"; then
    echo "⚠️  phi3 model not found. Pulling it now..."
    echo "    (This may take a few minutes on first run)"
    ollama pull phi3
    echo "✓ phi3 model ready"
else
    echo "✓ phi3 model already available"
fi

echo -e "\n========================================="
echo "✅ Setup Complete!"
echo "========================================="
echo "✓ Python $PYTHON_VERSION verified"
echo "✓ Virtual environment created: $VENV_DIR"
echo "✓ Dependencies installed"
echo "✓ Ollama with phi3 model ready"
echo -e "\n========================================="
echo "💻 What you need to DO!"
echo "========================================="
echo "📝 To activate venv, run:"
echo "   source $VENV_DIR/bin/activate  # bash/zsh"
echo "   . $VENV_DIR/bin/activate.fish  # fish"
echo -e "\n🤖 Make sure Ollama service is running in another terminal:"
echo "   ollama serve"
echo -e "\n▶️  To run the RAG demo, execute:"
echo "   python $SCRIPT_DIR/rag_demo.py"
echo "========================================="
