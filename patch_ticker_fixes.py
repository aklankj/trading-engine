"""
patch_ticker_fixes.py

Fixes yfinance ticker symbols that fail to download.
Run once: python patch_ticker_fixes.py

Fixes:
  - TATAMOTORS → TATAMTRDVR (ticker changed on yfinance)
  - GMRINFRA → GMRAIRPORT (company renamed)
  - ZOMATO: listed mid-2021, may need ZOMATO.NS check
"""

path = "factors/universe.py"
content = open(path).read()
changes = 0

# Fix TATAMOTORS
if '"TATAMOTORS"' in content:
    # Replace in NIFTY50 list - TATAMOTORS works on Kite but not yfinance
    # Keep the symbol but add a mapping in fetch function
    pass

# Better approach: add a yfinance symbol mapping
old_line = 'yf_symbols = [f"{s}.NS" for s in symbols]'
new_block = '''# Some NSE symbols have different tickers on yfinance
    YF_TICKER_MAP = {
        "TATAMOTORS": "TATAMTRDVR",
        "GMRINFRA": "GMRAIRPORT",
        "M&M": "M&M",
        "BAJAJ-AUTO": "BAJAJ-AUTO",
    }
    yf_symbols = [f"{YF_TICKER_MAP.get(s, s)}.NS" for s in symbols]'''

if old_line in content:
    content = content.replace(old_line, new_block)
    changes += 1
    print("1. Added YF_TICKER_MAP for failed tickers")

# Also fix the reverse mapping
old_mapping = 'yf_to_local = {f"{s}.NS": s for s in symbols}'
new_mapping = '''yf_to_local = {f"{YF_TICKER_MAP.get(s, s)}.NS": s for s in symbols}'''

if old_mapping in content:
    content = content.replace(old_mapping, new_mapping)
    changes += 1
    print("2. Fixed reverse mapping")

if changes > 0:
    open(path, "w").write(content)
    print(f"\n✅ {changes} fixes applied to {path}")
else:
    print("⚠️  No changes needed or markers not found")
    print("   Manually add YF_TICKER_MAP to factors/universe.py fetch_universe_prices()")
