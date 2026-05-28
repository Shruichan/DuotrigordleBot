#!/usr/bin/env bash
# Bundles the extension/ folder into dist/duotrigordle-bot.zip ready for the
# Chrome Web Store. Run after bumping the version in extension/manifest.json.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

mkdir -p dist
rm -f dist/duotrigordle-bot.zip

# Quick sanity: every file the manifest references must exist.
python3 - <<'PY'
import json, os, sys
m = json.load(open("extension/manifest.json"))
missing = []
for f in m["content_scripts"][0]["js"]: missing += [] if os.path.exists("extension/"+f) else [f]
for f in m["content_scripts"][0]["css"]: missing += [] if os.path.exists("extension/"+f) else [f]
for _, p in m["icons"].items(): missing += [] if os.path.exists("extension/"+p) else [p]
for r in m["web_accessible_resources"][0]["resources"]:
    missing += [] if os.path.exists("extension/"+r) else [r]
if missing:
    print("manifest references missing files:", missing); sys.exit(1)
print(f"manifest OK · version {m['version']}")
PY

( cd extension && zip -q -r ../dist/duotrigordle-bot.zip . -x ".*" )
echo "built dist/duotrigordle-bot.zip · $(du -h dist/duotrigordle-bot.zip | cut -f1)"
unzip -l dist/duotrigordle-bot.zip | tail -2
