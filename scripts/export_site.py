"""Assemble a portable copy of the E-Agent chat for any static host.

The ``web/`` directory is already a self-contained static site: every asset
reference is relative, so it works at a domain root (``enzyme.impf.ai``) or
under a sub-path (``user.github.io/E-Agent/``) without changes. This script
copies it into a clean output folder you can hand to Netlify, Vercel,
Cloudflare Pages, S3/CloudFront, nginx or GitHub Pages, and optionally pins a
Python-backend URL and writes a ``CNAME`` for a custom domain.

It does NOT rebuild the evidence bundle -- that is
``scripts/build_browser_bundle.py`` (run it, or its ``--check``, in CI). This
exporter only moves verified static files and checks their integrity.

    python scripts/export_site.py --out dist/enzyme-impf
    python scripts/export_site.py --cname enzyme.impf.ai
    python scripts/export_site.py --backend-url https://eagent-api.impf.ai

Browser + API mode needs no backend; leave ``--backend-url`` unset for it.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"

#: Local assets the page pulls in. Each must survive the copy, or the export is
#: broken in a way a host would only reveal at runtime.
REQUIRED = ("index.html", "styles.css", "app.js", "browser-agent.js",
            "reference-tools.js", "reference-data.json", "config.json")

_HOSTNAME = re.compile(
    r"^(?=.{1,253}$)(?!-)[A-Za-z0-9-]{1,63}(?<!-)"
    r"(?:\.(?!-)[A-Za-z0-9-]{1,63}(?<!-))+$")


def validate_backend_url(url: str) -> str:
    """Accept only a public HTTPS origin with a path, and no secrets in it."""
    parsed = urlsplit(url.strip())
    if parsed.scheme != "https" or not parsed.hostname:
        raise ValueError("the backend URL must be a public https:// URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("the backend URL cannot carry credentials, a query or a fragment")
    return url.strip().rstrip("/")


def validate_cname(host: str) -> str:
    host = host.strip().rstrip(".").lower()
    if host.startswith(("http://", "https://")) or "/" in host or not _HOSTNAME.match(host):
        raise ValueError(f"{host!r} is not a bare hostname such as enzyme.impf.ai")
    return host


def copy_site(out: Path) -> list[Path]:
    if out.resolve() == WEB.resolve():
        raise SystemExit("refusing to export onto the source web/ directory")
    if out.exists():
        shutil.rmtree(out)
    # Skip caches and editor cruft; copy everything else verbatim.
    ignore = shutil.ignore_patterns("__pycache__", ".DS_Store", "*.pyc")
    shutil.copytree(WEB, out, ignore=ignore)
    return sorted(p for p in out.rglob("*") if p.is_file())


def check_integrity(out: Path) -> None:
    missing = [name for name in REQUIRED if not (out / name).is_file()]
    if missing:
        raise SystemExit(f"export is missing required files: {', '.join(missing)}")
    for path in out.rglob("*.json"):
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise SystemExit(f"{path.relative_to(out)} is not valid JSON: {exc}")
    bundle = json.loads((out / "reference-data.json").read_text(encoding="utf-8"))
    if bundle.get("schema_version") != 1 or "system_prompt" not in bundle:
        raise SystemExit(
            "reference-data.json has an unexpected shape; rebuild it with "
            "scripts/build_browser_bundle.py before exporting")
    index = (out / "index.html").read_text(encoding="utf-8")
    for asset in ("styles.css", "app.js", "browser-agent.js", "reference-tools.js"):
        if asset not in index:
            raise SystemExit(f"index.html no longer references {asset}; the page would not load")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="dist/enzyme-impf",
                        help="output directory (relative to repo root; default dist/enzyme-impf)")
    parser.add_argument("--backend-url", default=None,
                        help="optional Python-backend HTTPS URL to pin into config.json "
                             "(omit for Browser + API mode, which needs no backend)")
    parser.add_argument("--cname", default=None,
                        help="optional custom domain to write as CNAME, e.g. enzyme.impf.ai "
                             "(only needed for GitHub Pages / Cloudflare-Pages-style hosts)")
    args = parser.parse_args(argv)

    out = (ROOT / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out)

    backend_url = ""
    if args.backend_url:
        try:
            backend_url = validate_backend_url(args.backend_url)
        except ValueError as exc:
            parser.error(str(exc))
    cname = ""
    if args.cname:
        try:
            cname = validate_cname(args.cname)
        except ValueError as exc:
            parser.error(str(exc))

    files = copy_site(out)

    # config.json: a pinned backend URL, or an empty one for browser-only mode.
    (out / "config.json").write_text(json.dumps({"backend_url": backend_url}) + "\n",
                                     encoding="utf-8")
    if cname:
        (out / "CNAME").write_text(cname + "\n", encoding="utf-8")
        files = sorted(set(files) | {out / "CNAME"})

    check_integrity(out)

    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"Exported the E-Agent chat to {out}")
    print(f"  files: {sum(1 for _ in out.rglob('*') if _.is_file())}   total: {total/1024:.0f} KiB")
    print(f"  run mode: {'Python backend pinned → ' + backend_url if backend_url else 'Browser + API (no backend)'}")
    print(f"  custom domain: {cname or '(none — set --cname for GitHub/Cloudflare Pages)'}")
    print("  next: upload the folder's contents to your host's web root for enzyme.impf.ai")
    print("        (details in docs/DEPLOY_ENZYME_IMPF.md)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
