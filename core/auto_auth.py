"""
Automated Kite login — no browser needed.
Uses direct HTTP requests + pyotp to generate TOTP.
Runs daily at 6:30 AM IST before market opens.
"""

import json
import requests
import pyotp
from pathlib import Path
from datetime import datetime
from kiteconnect import KiteConnect

from config.settings import cfg
from utils.logger import log

TOKEN_FILE = cfg.DATA_DIR / "kite_token.json"

LOGIN_URL = "https://kite.zerodha.com/api/login"
TWOFA_URL = "https://kite.zerodha.com/api/twofa"


def auto_login() -> bool:
    """
    Fully automated Kite login.
    Returns True if successful, False otherwise.
    """
    user_id = cfg.KITE_USER_ID
    password = cfg.KITE_PASSWORD
    totp_secret = cfg.KITE_TOTP_SECRET
    api_key = cfg.KITE_API_KEY
    api_secret = cfg.KITE_API_SECRET

    if not all([user_id, password, totp_secret, api_key, api_secret]):
        log.error("Auto-login: missing credentials in .env")
        return False

    session = requests.Session()

    try:
        # Step 1: Login with user_id and password
        log.info(f"Auto-login: logging in as {user_id}...")
        login_resp = session.post(LOGIN_URL, data={
            "user_id": user_id,
            "password": password,
        })

        if login_resp.status_code != 200:
            log.error(f"Auto-login: login failed with status {login_resp.status_code}")
            return False

        login_data = login_resp.json()
        if login_data.get("status") != "success":
            log.error(f"Auto-login: login failed — {login_data.get('message', 'unknown error')}")
            return False

        request_id = login_data["data"]["request_id"]
        log.info(f"Auto-login: step 1 passed, request_id obtained")

        # Step 2: 2FA with TOTP
        totp = pyotp.TOTP(totp_secret)
        totp_value = totp.now()

        twofa_resp = session.post(TWOFA_URL, data={
            "user_id": user_id,
            "request_id": request_id,
            "twofa_value": totp_value,
            "twofa_type": "totp",
        })

        if twofa_resp.status_code != 200:
            log.error(f"Auto-login: 2FA failed with status {twofa_resp.status_code}")
            return False

        log.info("Auto-login: step 2 passed, 2FA complete")

        # Step 3: Get request_token via Kite Connect login flow
        kite = KiteConnect(api_key=api_key)
        login_url = kite.login_url()

        # Follow the login URL with our authenticated session
        redirect_resp = session.get(login_url, allow_redirects=False)

        # Follow redirects manually to capture request_token
        while redirect_resp.status_code in (301, 302, 303, 307, 308):
            redirect_url = redirect_resp.headers.get("Location", "")
            if "request_token=" in redirect_url:
                break
            redirect_resp = session.get(redirect_url, allow_redirects=False)

        # Extract request_token from final redirect URL
        final_url = redirect_resp.headers.get("Location", "")
        if "request_token=" not in final_url:
            log.error(f"Auto-login: could not find request_token in redirect")
            log.debug(f"Final URL: {final_url}")
            return False

        request_token = final_url.split("request_token=")[1].split("&")[0]
        log.info(f"Auto-login: got request_token")

        # Step 4: Generate access_token
        session_data = kite.generate_session(request_token, api_secret=api_secret)
        access_token = session_data["access_token"]

        # Save token
        kite.set_access_token(access_token)
        profile = kite.profile()

        token_data = {
            "access_token": access_token,
            "public_token": session_data.get("public_token", ""),
            "date": datetime.now().strftime("%Y-%m-%d"),
            "saved_at": datetime.now().isoformat(),
        }
        TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
        TOKEN_FILE.write_text(json.dumps(token_data, indent=2))

        log.info(f"Auto-login SUCCESS — {profile.get('user_name', 'OK')}")

        # Send Telegram notification
        try:
            from execution.telegram_bot import send_message
            send_message(f"🔑 Kite auto-login successful — {profile.get('user_name', 'authenticated')}")
        except Exception:
            pass

        return True

    except Exception as e:
        log.error(f"Auto-login failed: {e}")

        try:
            from execution.telegram_bot import send_message
            send_message(f"⚠️ Kite auto-login FAILED: {str(e)[:100]}")
        except Exception:
            pass

        return False


if __name__ == "__main__":
    success = auto_login()
    if success:
        print("Auto-login successful!")
    else:
        print("Auto-login failed — check logs")
        exit(1)
