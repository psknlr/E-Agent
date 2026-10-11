"""Assemble a portable copy of the E-Agent chat for any static host.

The ``web/`` directory is already a self-contained static site: every asset
reference is relative, so it works at a domain root (``enzyme.impf.ai``) or
under a sub-path (``user.github.io/E-Agent/``) without changes. This script
copies it into a clean output folder you can hand to Netlify, Vercel,
Cloudflare Pages, S3/CloudFront, nginx or GitHub Pages, and optionally pins a
Python-backend URL and writes a ``CNAME`` for a custom domain.

Two things beyond the plain files:

* The compute engine (``web/engine/``, Python as WebAssembly) is checked against
  its build manifest and copied with the site. Build it first with
  ``scripts/build_engine.py``; ``--without-engine`` exports the site without it,
  and the page then says the engine is not published and keeps working.
* ``--redirect-to URL`` writes no site at all, only what a host needs to send
  every visitor to ``URL`` -- for a domain that should simply lead to the GitHub
  Pages site, which already carries the whole application.

It does NOT rebuild the evidence bundle -- that is
``scripts/build_browser_bundle.py`` (run it, or its ``--check``, in CI). This
exporter only moves verified static files and checks their integrity.

    python scripts/export_site.py --out dist/enzyme-impf
    python scripts/export_site.py --cname enzyme.impf.ai
    python scripts/export_site.py --backend-url https://eagent-api.impf.ai
    python scripts/export_site.py --redirect-to https://psknlr.github.io/E-Agent/ --out dist/enzyme-redirect

Browser + API mode needs no backend; leave ``--backend-url`` unset for it.
"""

from __future__ import annotations

import argparse
import html
import importlib.util
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
            "endpoint.js", "reference-tools.js", "reference-data.json", "config.json",
            "agents.js", "agent-catalog.json", "compute-ui.js", "engine-client.js",
            "engine-install.js", "engine-worker.js")

#: What index.html must still reference, or the page would not load.
PAGE_ASSETS = ("styles.css", "app.js", "browser-agent.js", "endpoint.js", "reference-tools.js",
               "agents.js", "compute-ui.js", "engine-client.js", "engine-install.js")

#: A redirect target is embedded in HTML, JavaScript, JSON and several servers'
#: config languages, so it is held to the characters none of them treat as syntax.
_SAFE_TARGET = re.compile(r"^[A-Za-z0-9._~:/%-]+$")

#: The engine's loader files must be served as what they are. Netlify and
#: Cloudflare Pages read this file; for nginx and Apache see docs/DEPLOY_ENZYME_IMPF.md.
HEADERS = """\
/engine/pyodide/pyodide.mjs
  Content-Type: text/javascript; charset=utf-8
/engine/pyodide/pyodide.asm.mjs
  Content-Type: text/javascript; charset=utf-8
/engine/pyodide/pyodide.asm.wasm
  Content-Type: application/wasm
"""

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


def validate_redirect_target(url: str) -> str:
    """Accept only a public HTTPS URL made of characters no config language reads as syntax."""
    text = url.strip()
    parsed = urlsplit(text)
    if parsed.scheme != "https" or not parsed.hostname:
        raise ValueError("the redirect target must be a public https:// URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("the redirect target cannot carry credentials, a query or a fragment")
    if not _SAFE_TARGET.match(text) or not _HOSTNAME.match(parsed.hostname):
        raise ValueError("the redirect target may contain only letters, digits and . _ ~ : / % - "
                         "in a DNS host name")
    return text


def marker_for(out: Path) -> Path:
    """The file that says this script made ``out``: a sibling, not a member.

    It sits beside the directory so that what is inside it stays exactly what is
    to be uploaded. A directory that is already there is replaced only if its
    marker is, so the script never deletes a folder it did not make.
    """
    return out.parent / f".{out.name}.eagent-export"


def prepare_output(out: Path) -> None:
    """Make ``out`` a path this script may create, or stop.

    The exporter replaces its output directory, which means deleting it. That is
    only safe for a directory the exporter made: ``--out .`` once removed the
    repository's own ``scripts/`` and ``web/``, and ``--out ~/Documents`` would
    have removed those. So an existing directory is replaced only when it is empty
    or has the marker an earlier export left, and never when it is, or holds, the
    repository, the source ``web/``, the current directory or the home directory.
    """
    resolved = out.resolve()
    protected = [ROOT.resolve(), WEB.resolve(), Path.cwd().resolve(), Path.home().resolve(), Path(resolved.anchor)]
    for place in protected:
        if resolved == place or resolved in place.parents:
            raise SystemExit(f"refusing to use {out} as an output directory: it is, or contains, {place}. "
                             f"Choose a new directory, for example dist/enzyme-impf.")
    if WEB.resolve() in resolved.parents:
        raise SystemExit("refusing to write inside the source web/ directory")
    if resolved.exists():
        if not resolved.is_dir():
            raise SystemExit(f"{out} exists and is not a directory")
        if any(resolved.iterdir()) and not marker_for(resolved).is_file():
            raise SystemExit(f"{out} already exists and was not made by this script, so it will not be deleted. "
                             f"Remove it yourself (rm -r {out}) or choose another directory.")
        shutil.rmtree(resolved)


def mark_output(out: Path) -> None:
    marker_for(out.resolve()).write_text("made by scripts/export_site.py; the folder beside it is safe to replace\n",
                                         encoding="utf-8")


def copy_site(out: Path, *, with_engine: bool = True) -> list[Path]:
    prepare_output(out)
    # Skip caches and editor cruft; copy everything else verbatim.
    cruft = shutil.ignore_patterns("__pycache__", ".DS_Store", "*.pyc")

    def ignore(directory: str, names: list[str]) -> set[str]:
        skipped = set(cruft(directory, names))
        if not with_engine and Path(directory).resolve() == WEB.resolve():
            skipped.add("engine")
        return skipped

    shutil.copytree(WEB, out, ignore=ignore)
    mark_output(out)
    return sorted(p for p in out.rglob("*") if p.is_file())


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_engine(out: Path) -> None:
    """The copied engine must be a complete, consistent build of this checkout."""
    if not (out / "engine" / "manifest.json").is_file():
        raise SystemExit("the compute engine is not built; run python scripts/build_engine.py "
                         "first, or export with --without-engine")
    builder = _load_script("build_engine")
    problems = builder.check(out / "engine", builder.load_lock())
    if problems:
        raise SystemExit("the compute engine is not a consistent build:\n  " + "\n  ".join(problems)
                         + "\nrun python scripts/build_engine.py")


def check_integrity(out: Path, *, with_engine: bool = True) -> None:
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
    for asset in PAGE_ASSETS:
        if asset not in index:
            raise SystemExit(f"index.html no longer references {asset}; the page would not load")
    catalog = json.loads((out / "agent-catalog.json").read_text(encoding="utf-8"))
    if catalog.get("schema") != "eagent.engine.v1" or not catalog.get("agents"):
        raise SystemExit("agent-catalog.json has an unexpected shape; rebuild it with "
                         "scripts/build_browser_bundle.py before exporting")
    if with_engine:
        check_engine(out)


def write_redirect_site(out: Path, target: str, cname: str = "") -> list[str]:
    """What a host needs to send every visitor to ``target``, and nothing else.

    A page for hosts that only serve files (S3, GitHub Pages), a 404 page that does
    the same for any other path, and the redirect rule of each common host. The
    rules answer with a real 301, so crawlers and browsers remember the move.
    """
    prepare_output(out)
    out.mkdir(parents=True)
    mark_output(out)
    shown, attribute, script = html.escape(target), html.escape(target, quote=True), json.dumps(target)
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex">
<title>E-Agent has moved</title>
<link rel="canonical" href="{attribute}">
<meta http-equiv="refresh" content="0; url={attribute}">
<script>location.replace({script});</script>
<style>body{{font:16px/1.5 system-ui,sans-serif;margin:12vh auto;max-width:34em;padding:0 1.2em;color:#1c2a24}}a{{color:#1f7d58}}</style>
</head>
<body>
<p>E-Agent now lives at <a href="{attribute}">{shown}</a>. You are being taken there.</p>
</body>
</html>
"""
    files = {
        "index.html": page,
        "404.html": page,
        "_redirects": f"/*  {target}  301\n",
        "vercel.json": json.dumps({"redirects": [{"source": "/(.*)", "destination": target, "permanent": True}]}, indent=2) + "\n",
        ".htaccess": f"RewriteEngine On\nRewriteRule ^ {target} [R=301,L]\n",
        "nginx-redirect.conf.example": (
            "# Put this in the server block that answers for your domain.\n"
            f"return 301 {target};\n"),
    }
    if cname:
        files["CNAME"] = cname + "\n"
    for name, content in files.items():
        (out / name).write_text(content, encoding="utf-8")
    return sorted(files)


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
    parser.add_argument("--without-engine", action="store_true",
                        help="export the site without the compute engine (about 20 MB); the page "
                             "then reports the engine as not published")
    parser.add_argument("--redirect-to", default=None, metavar="URL",
                        help="write only a redirect to this https:// URL instead of a copy of the site")
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

    if args.redirect_to:
        if backend_url or args.without_engine:
            parser.error("--redirect-to writes no site, so --backend-url and --without-engine do not apply")
        try:
            target = validate_redirect_target(args.redirect_to)
        except ValueError as exc:
            parser.error(str(exc))
        names = write_redirect_site(out, target, cname)
        print(f"Wrote a redirect to {target} in {out}")
        print(f"  files: {', '.join(names)}")
        print("  next: upload the folder to a static host, or copy the rule your host understands")
        print("        (_redirects: Netlify and Cloudflare Pages; vercel.json; .htaccess: Apache;")
        print("        nginx-redirect.conf.example: nginx). Details in docs/DEPLOY_ENZYME_IMPF.md.")
        return 0

    with_engine = not args.without_engine
    files = copy_site(out, with_engine=with_engine)

    # config.json: a pinned backend URL, or an empty one for browser-only mode.
    (out / "config.json").write_text(json.dumps({"backend_url": backend_url}) + "\n",
                                     encoding="utf-8")
    if cname:
        (out / "CNAME").write_text(cname + "\n", encoding="utf-8")
        files = sorted(set(files) | {out / "CNAME"})
    if with_engine:
        (out / "_headers").write_text(HEADERS, encoding="utf-8")

    check_integrity(out, with_engine=with_engine)

    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"Exported the E-Agent chat to {out}")
    print(f"  files: {sum(1 for _ in out.rglob('*') if _.is_file())}   total: {total/1048576:.1f} MiB")
    print(f"  run mode: {'Python backend pinned → ' + backend_url if backend_url else 'Browser + API (no backend)'}")
    print(f"  compute engine: {'included (Python as WebAssembly, checked against its manifest)' if with_engine else 'not included (--without-engine)'}")
    print(f"  custom domain: {cname or '(none — set --cname for GitHub/Cloudflare Pages)'}")
    print("  next: upload the folder's contents to your host's web root for enzyme.impf.ai")
    print("        (details in docs/DEPLOY_ENZYME_IMPF.md)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
