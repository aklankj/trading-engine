"""
Kite Connect authentication.

Access tokens expire daily at ~6:00 AM IST. This module handles:
1. Loading saved token from disk
2. Generating new session from request_token
3. Providing a ready-to-use KiteConnect instance

First-time setup requires manual browser login to get request_token.
After that, the token is persisted and reused until it expires.

For fully automated daily re-auth, consider using the enctoken approach
via the kitetrader library or a Selenium-based login flow.
"""

import json
from pathlib import Path
from datetime import datetime
from kiteconnect import KiteConnect

from config.settings import cfg
from utils.logger import log

TOKEN_FILE = cfg.DATA_DIR / "kite_token.json"


def _load_token() -> dict | None:
    """Load saved access token if it's from today."""
    if not TOKEN_FILE.exists():
        return None
    try:
        data = json.loads(TOKEN_FILE.read_text())
        saved_date = data.get("date", "")
        today = datetime.now().strftime("%Y-%m-%d")
        if saved_date == today and data.get("access_token"):
            return data
    except (json.JSONDecodeError, KeyError):
        pass
    return None


def _save_token(access_token: str, public_token: str = "") -> None:
    """Persist token with today's date."""
    TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "access_token": access_token,
        "public_token": public_token,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "saved_at": datetime.now().isoformat(),
    }
    TOKEN_FILE.write_text(json.dumps(data, indent=2))
    log.info("Kite access token saved for today")


def get_kite(request_token: str | None = None) -> KiteConnect:
    """
    Return an authenticated KiteConnect instance.

    Priority:
    1. If access_token is in .env and valid, use it
    2. If saved token from today exists, use it
    3. If request_token provided, generate new session
    4. Raise error with login URL

    Args:
        request_token: From the redirect URL after manual Kite login.
                       Only needed for first-time or daily re-auth.
    """
    kite = KiteConnect(api_key=cfg.KITE_API_KEY)

    # Try .env token first
    if cfg.KITE_ACCESS_TOKEN:
        kite.set_access_token(cfg.KITE_ACCESS_TOKEN)
        try:
            profile = kite.profile()
            log.info(f"Authenticated via .env token — {profile.get('user_name', 'OK')}")
            return kite
        except Exception:
            log.warning("Token from .env expired or invalid, trying saved token...")

    # Try saved token
    saved = _load_token()
    if saved:
        kite.set_access_token(saved["access_token"])
        try:
            profile = kite.profile()
            log.info(f"Authenticated via saved token — {profile.get('user_name', 'OK')}")
            return kite
        except Exception:
            log.warning("Saved token expired, need fresh login")

    # Generate from request_token
    if request_token:
        try:
            session = kite.generate_session(
                request_token, api_secret=cfg.KITE_API_SECRET
            )
            access_token = session["access_token"]
            kite.set_access_token(access_token)
            _save_token(access_token, session.get("public_token", ""))
            profile = kite.profile()
            log.info(f"New session created — {profile.get('user_name', 'OK')}")
            return kite
        except Exception as e:
            log.error(f"Session generation failed: {e}")
            raise

    # No valid token — try auto-login before giving up
    try:
        from core.auto_auth import auto_login
        log.info("No valid token — attempting auto-login...")
        if auto_login():
            # Reload the saved token
            saved = _load_token()
            if saved:
                kite.set_access_token(saved["access_token"])
                log.info("Auto-login succeeded, token loaded")
                return kite
    except Exception as e:
        log.warning(f"Auto-login attempt failed: {e}")

    # Auto-login failed — show manual login URL
    login_url = kite.login_url()
    msg = (
        f"\n{'='*60}\n"
        f"  KITE LOGIN REQUIRED\n"
        f"  Auto-login failed. Open this URL in your browser:\n\n"
        f"  {login_url}\n\n"
        f"  After login, copy the request_token from the redirect URL\n"
        f"  and run: python -m core.auth <request_token>\n"
        f"{'='*60}\n"
    )
    log.error(msg)
    # Don't crash the engine — return None so caller can handle it
    return None


# ── CLI entrypoint for initial auth ──────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        kite = KiteConnect(api_key=cfg.KITE_API_KEY)
        print(f"\nOpen this URL to login:\n{kite.login_url()}\n")
        print("Then run: python -m core.auth YOUR_REQUEST_TOKEN")
        sys.exit(1)

    token = sys.argv[1]
    kite = get_kite(request_token=token)
    print(f"\nAuthenticated successfully!")
    print(f"Profile: {kite.profile()}")
    print(f"Token saved to: {TOKEN_FILE}")
