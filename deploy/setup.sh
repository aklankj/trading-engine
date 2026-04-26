#!/bin/bash
# ═══════════════════════════════════════════════════════════
# TRADING ENGINE — VPS Deployment Script
# Run: bash deploy/setup.sh
# ═══════════════════════════════════════════════════════════

set -e

echo "═══ Trading Engine Setup ═══"

# 1. Create virtual environment
echo "[1/6] Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
echo "[2/6] Installing Python packages..."
pip install --upgrade pip
pip install -r requirements.txt

# 3. Create directories
echo "[3/6] Creating data and log directories..."
mkdir -p data logs

# 4. Setup config
if [ ! -f config/.env ]; then
    echo "[4/6] Creating .env from template..."
    cp config/.env.example config/.env
    echo "  ⚠️  IMPORTANT: Edit config/.env with your API keys!"
else
    echo "[4/6] config/.env already exists, skipping"
fi

# 5. Setup systemd service
echo "[5/6] Setting up systemd service..."
if [ "$(id -u)" -eq 0 ]; then
    # Update paths in service file to match current directory
    INSTALL_DIR=$(pwd)
    USER=$(logname 2>/dev/null || echo $SUDO_USER || echo $USER)
    sed -e "s|/home/rex/trading-engine|${INSTALL_DIR}|g" \
        -e "s|User=rex|User=${USER}|g" \
        -e "s|Group=rex|Group=${USER}|g" \
        deploy/trading-engine.service > /etc/systemd/system/trading-engine.service
    systemctl daemon-reload
    systemctl enable trading-engine
    echo "  Service installed. Start with: sudo systemctl start trading-engine"
else
    echo "  Run as root to install systemd service, or manually copy:"
    echo "  sudo cp deploy/trading-engine.service /etc/systemd/system/"
fi

# 6. Verify
echo "[6/6] Verifying installation..."
python -c "from config.settings import cfg; issues = cfg.validate(); print(f'Config issues: {len(issues)}')"
python -c "import kiteconnect; print(f'kiteconnect version: {kiteconnect.__version__}')" 2>/dev/null || echo "  kiteconnect not installed (pip install kiteconnect)"
python -c "import schedule; print('schedule: OK')"
python -c "import feedparser; print('feedparser: OK')"
python -c "import loguru; print('loguru: OK')"

echo ""
echo "═══ Setup complete! ═══"
echo ""
echo "Next steps:"
echo "  1. Edit config/.env with your API keys"
echo "  2. Authenticate with Kite:"
echo "     python -m core.auth"
echo "  3. Test a morning scan:"
echo "     python main.py --morning"
echo "  4. Check system status:"
echo "     python main.py --status"
echo "  5. Start the engine:"
echo "     python main.py   (or: sudo systemctl start trading-engine)"
echo ""
