#!/usr/bin/env python3
"""Download the original SVG benchmark files listed in manifest.json."""
import json, pathlib, urllib.request, time
ROOT = pathlib.Path(__file__).resolve().parent
manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
out = ROOT / "assets"
out.mkdir(exist_ok=True)
for item in manifest["assets"]:
    target = out / item["filename"]
    if target.exists() and target.stat().st_size:
        print(f"[skip] {target.name} already exists")
        continue
    req = urllib.request.Request(item["download_url"], headers={"User-Agent": "ImageToVectorBenchmark/1.0 (research/QA)"})
    try:
        with urllib.request.urlopen(req, timeout=45) as response:
            data = response.read()
        # Reject HTML error pages masquerading as downloads.
        head = data[:4096].lower()
        if b"<svg" not in head and b"<svg" not in data[:20000].lower():
            raise RuntimeError("Downloaded response does not appear to be SVG XML")
        target.write_bytes(data)
        print(f"[ok] {target.name}: {len(data):,} bytes")
        time.sleep(0.5)
    except Exception as exc:
        print(f"[failed] {target.name}: {exc}")
print("\nDone. See ATTRIBUTION.md before redistributing or publishing results.")
