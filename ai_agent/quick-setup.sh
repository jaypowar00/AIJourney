#!/bin/bash

# Quick setup script for AI Agent with SearxNG
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv_agent"
DOCKER_COMPOSE_FILE="$SCRIPT_DIR/searxng/docker-compose.yaml"

echo "========================================="
echo "🚀 AI Agent Quick Setup"
echo "========================================="

# Step 0: Check Python version
echo -e "\n[0/6] Checking Python version..."
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
echo -e "\n[1/6] Creating virtual environment 'venv_agent'..."
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

# Step 2: Install dependencies
echo -e "\n[2/6] Installing dependencies..."
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
done < "$SCRIPT_DIR/requirements-agent.txt"

# Step 3: Check and install Docker
echo -e "\n[3/6] Checking Docker installation..."
if ! command -v docker &> /dev/null; then
    echo "Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    rm get-docker.sh
    sudo usermod -aG docker "$USER"
    echo "⚠️  Please log out and log back in to use Docker without sudo"
fi

# Step 4: Stop existing SearxNG container
echo -e "\n[4/6] Stopping existing SearxNG containers(if any)..."
docker compose -f "$DOCKER_COMPOSE_FILE" down 2>/dev/null || true
docker ps -a | grep searxng | awk '{print $1}' | xargs -r docker rm -f 2>/dev/null || true

# Step 5: Start SearxNG
echo -e "\n[5/6] Starting a fresh SearxNG container..."
docker compose -f "$DOCKER_COMPOSE_FILE" up -d
sleep 5

# Step 6: Configure SearxNG settings
echo -e "\n[6/6] Configuring SearxNG settings..."

# Add JSON format support (after line 78 with "html")
docker exec searxng sed -i '/^[[:space:]]*- html$/a\    - json' /etc/searxng/settings.yml

# Add allow_public_api option (after line 93 with "public_instance: false")
docker exec searxng sed -i '/public_instance: false$/a\  allow_public_api: true' /etc/searxng/settings.yml

# Restart SearxNG to apply changes
echo "Restarting SearxNG..."
docker compose -f "$DOCKER_COMPOSE_FILE" restart searxng
sleep 3

echo -e "\n========================================="
echo "✅ Setup Complete!"
echo "========================================="
echo "✓ Python $PYTHON_VERSION verified"
echo "✓ Virtual environment created: $VENV_DIR"
echo "✓ Dependencies installed"
echo "✓ Docker and docker-compose ready"
echo "✓ SearxNG running at http://localhost:8080"
echo -e "\n========================================="
echo "💻 What you need to DO!"
echo "========================================="
echo "📝 To activate venv, run:"
echo "   source $VENV_DIR/bin/activate  # bash/zsh"
echo "   . $VENV_DIR/bin/activate.fish  # fish"
echo -e "\n▶️  To run the agent, execute:"
echo "   python $SCRIPT_DIR/agent_demo.py"
echo "========================================="