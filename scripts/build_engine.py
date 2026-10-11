"""Assemble the in-browser engine from pinned, hash-verified parts.

The engine is the project's own Python, run on the visitor's CPU as WebAssembly.
This script builds everything it needs into one directory the static site serves:

    web/engine/
      manifest.json            what is here, with a SHA-256 for every file
      NOTICE.txt               what each component is and its licence
      pyodide/                 CPython compiled to WebAssembly (from npm)
      wheels/                  pydantic and its dependencies (from PyPI), PyYAML
      eagent-project.zip       src/eagent/**/*.py and the shipped task files

NOTHING BINARY IS COMMITTED
===========================
About twenty megabytes of assets, none of them in git. They are fetched at build
time (in CI on every deploy, or by hand for local work) from the registries they
are published on, and every download is checked against a SHA-256 recorded in
``scripts/engine.lock.json``. A mismatch aborts the build: a deploy that cannot
prove what it is shipping does not ship it. Upgrading anything is a change to
:data:`PINS` plus ``--update-lock``, which is a diff a reviewer can read.

WHY THE ZIPS ARE NOT COMPRESSED
===============================
Every zip written here is stored, with fixed timestamps and sorted entries, so
the same inputs give the same bytes on any machine. A deflate stream depends on
the zlib that wrote it, and a manifest hash that changes between a laptop and a
CI runner would make ``--check`` useless. The host compresses what it serves
anyway, so the transfer is unaffected.

    python scripts/build_engine.py                  # build into web/engine
    python scripts/build_engine.py --check          # verify an existing build
    python scripts/build_engine.py --update-lock    # re-resolve the pinned versions
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import re
import shutil
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Callable, Iterable

ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "scripts" / "engine.lock.json"
DEFAULT_OUT = ROOT / "web" / "engine"
DEFAULT_CACHE = ROOT / ".cache" / "engine"

LOCK_SCHEMA = "eagent.engine.lock.v1"
MANIFEST_SCHEMA = "eagent.engine.manifest.v1"

#: What is pinned. Each wheel is (import name, PyPI name, version, filename tag).
#: pydantic-core is the one that is not pure Python: its WebAssembly build is
#: published to PyPI under the emscripten platform tag Pyodide 314 uses.
PINS: dict[str, Any] = {
    "pyodide": "314.0.7",
    "wheels": [
        ("typing_extensions", "typing-extensions", "4.16.0", "py3-none-any"),
        ("annotated_types", "annotated-types", "0.8.0", "py3-none-any"),
        ("typing_inspection", "typing-inspection", "0.4.4", "py3-none-any"),
        ("pydantic_core", "pydantic-core", "2.50.0", "cp314-cp314-pyemscripten_2026_0_wasm32"),
        ("pydantic", "pydantic", "2.14.0", "py3-none-any"),
    ],
    "pyyaml": "6.0.3",
}

#: The files taken from the Pyodide npm tarball. The rest (consoles, source maps,
#: type declarations, the UMD build) is not needed by a module worker.
PYODIDE_FILES = ("pyodide.mjs", "pyodide.asm.mjs", "pyodide.asm.wasm",
                 "python_stdlib.zip", "pyodide-lock.json")

#: What the engine needs from this repository, and nothing else: the Python
#: package, the shipped task files its reports can read, and the template library
#: the controller loads (every threshold in a run comes from it). Reference data
#: products and docs are deliberately not shipped.
PROJECT_PATTERNS = (("src/eagent", "*.py"), ("configs/tasks", "*.yaml"), ("configs/templates", "*.yaml"))

#: Calls the page repeats before it will use the engine. The build computes each
#: natively and records the hash of the answer; the runtime in the visitor's
#: browser must produce the same hash, or it is not used. The inputs exercise
#: what could differ -- the alignment dynamic programme, the compiled pydantic
#: core, float formatting -- and stay on exact arithmetic (+ - * / and sqrt are
#: correctly rounded everywhere; cos, acos and exp are not, so no agent that
#: needs them is here).
SELFTEST = (
    ("sequence_distance", {"sequence_a": "ACDEFGHIKLMNPQRSTVWY",
                           "sequence_b": "ACDEFGHIKLMNPQRSTVWF", "method": "hamming"}),
    ("align_sequences", {"sequence_a": "MKTAYIAKQRQISFVKSHFSRQ",
                         "sequence_b": "MKTAYLAKQRQVSFVKSHFSRQ"}),
    ("wilson_interval", {"successes": "7", "trials": "20"}),
    ("variant_check", {"sequence": "MKTAYIAKQR", "variants": "K2A, T3S, Y9F"}),
)

FIXED_TIME = (1980, 1, 1, 0, 0, 0)

#: The shapes a lock entry's fields must have. The lock is a file in the
#: repository and the registries' replies are somebody else's input, and both
#: end up as paths (a cache entry named by a hash, an output file named by a
#: wheel's filename), so a field is checked before it is used as one.
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_WHEEL_FILENAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\.whl\Z")


class BuildError(RuntimeError):
    """The build cannot proceed without shipping something unverified."""


# ---------------------------------------------------------------------------
# hashing, fetching
# ---------------------------------------------------------------------------

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def fetch(url: str, sha256: str, cache: Path, *,
          opener: Callable[..., Any] = urllib.request.urlopen) -> bytes:
    """Bytes of ``url``, verified against ``sha256``. Cached by that hash.

    A cached file is re-verified, not trusted, and a bad one is deleted so a
    corrupt download cannot make every later build fail the same way.
    """
    if not url.startswith("https://"):
        raise BuildError(f"{url}\n  only https:// downloads are accepted")
    if not _SHA256.match(sha256):
        raise BuildError(f"{url}\n  {sha256!r} is not a SHA-256 digest (64 lowercase hex digits)")
    cache.mkdir(parents=True, exist_ok=True)
    cached = cache / sha256
    if cached.is_file():
        data = cached.read_bytes()
        if sha256_bytes(data) == sha256:
            return data
        cached.unlink()
    request = urllib.request.Request(url, headers={"User-Agent": "eagent-build-engine"})
    with opener(request, timeout=120) as response:
        data = response.read()
    actual = sha256_bytes(data)
    if actual != sha256:
        raise BuildError(f"{url}\n  expected sha256 {sha256}\n  received sha256 {actual}\n"
                         f"refusing to build from an artefact that is not the one that was pinned")
    cached.write_bytes(data)
    return data


# ---------------------------------------------------------------------------
# deterministic zips
# ---------------------------------------------------------------------------

def stored_zip(entries: Iterable[tuple[str, bytes]]) -> bytes:
    """A zip whose bytes depend only on the entries: stored, sorted, fixed time."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_STORED) as archive:
        for name, data in sorted(entries, key=lambda item: item[0]):
            info = zipfile.ZipInfo(name, date_time=FIXED_TIME)
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o644 << 16
            archive.writestr(info, data)
    return buffer.getvalue()


def project_entries(root: Path = ROOT) -> list[tuple[str, bytes]]:
    """The repository files the engine ships, as (archive path, bytes)."""
    entries: list[tuple[str, bytes]] = []
    for base, pattern in PROJECT_PATTERNS:
        for path in sorted((root / base).rglob(pattern)):
            if path.is_file() and "__pycache__" not in path.parts:
                entries.append((path.relative_to(root).as_posix(), path.read_bytes()))
    if not any(name.startswith("src/eagent/engine/") for name, _ in entries):
        raise BuildError("src/eagent/engine is missing from the project payload")
    return entries


def project_zip(root: Path = ROOT) -> bytes:
    return stored_zip(project_entries(root))


def pyyaml_pure_zip(sdist: bytes, version: str) -> bytes:
    """PyYAML's pure-Python package and licence, from its source distribution.

    PyYAML's C extension is optional and has no WebAssembly wheel on PyPI, so the
    pure package is used. It is the same code the C loader accelerates; the only
    cost is parse speed on files that are a few kilobytes.
    """
    entries: list[tuple[str, bytes]] = []
    with tarfile.open(fileobj=io.BytesIO(sdist), mode="r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            if "/lib/yaml/" in member.name and member.name.endswith(".py"):
                relative = member.name.split("/lib/yaml/", 1)[1]
                if relative.startswith("/") or ".." in relative.split("/"):
                    raise BuildError(f"the PyYAML source distribution holds an unsafe path {member.name!r}")
                entries.append(("yaml/" + relative, archive.extractfile(member).read()))
            elif member.name.count("/") == 1 and member.name.endswith("/LICENSE"):
                entries.append((f"PyYAML-{version}.dist-info/LICENSE",
                                archive.extractfile(member).read()))
    if not any(name.startswith("yaml/") for name, _ in entries):
        raise BuildError("the PyYAML source distribution holds no lib/yaml package")
    if not any(name.endswith("/LICENSE") for name, _ in entries):
        raise BuildError("the PyYAML source distribution holds no LICENSE to ship with it")
    return stored_zip(entries)


def pyodide_files(tarball: bytes) -> dict[str, bytes]:
    """The runtime files, taken from the npm tarball by exact name."""
    found: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(tarball), mode="r:gz") as archive:
        for name in PYODIDE_FILES:
            try:
                member = archive.getmember(f"package/{name}")
            except KeyError:
                raise BuildError(f"the Pyodide tarball has no package/{name}") from None
            found[name] = archive.extractfile(member).read()
    return found


def selftest_entries() -> list[dict[str, str]]:
    """What this checkout's agents answer natively for each self-test call."""
    source = str(ROOT / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    from eagent.engine import call  # imported here: the build needs no project import otherwise

    entries = []
    for agent, arguments in SELFTEST:
        text = json.dumps(arguments, sort_keys=True)
        reply = json.loads(call(agent, text))
        if not reply["ok"]:
            raise BuildError(f"the self-test call to {agent} was refused natively: {reply['refusal']}")
        entries.append({"agent": agent, "arguments": text, "sha256": reply["cite"]["sha256"]})
    return entries


# ---------------------------------------------------------------------------
# the lock
# ---------------------------------------------------------------------------

def load_lock(path: Path = LOCK_PATH) -> dict[str, Any]:
    if not path.is_file():
        raise BuildError(f"{path} does not exist; run with --update-lock to create it")
    lock = json.loads(path.read_text(encoding="utf-8"))
    if lock.get("schema") != LOCK_SCHEMA:
        raise BuildError(f"{path} is not a {LOCK_SCHEMA} file")
    for key in ("pyodide", "wheels", "pyyaml"):
        if key not in lock:
            raise BuildError(f"{path} has no {key!r} section")
    problems = lock_shape_problems(lock)
    if problems:
        raise BuildError(f"{path} is malformed:\n  " + "\n  ".join(problems))
    return lock


def lock_shape_problems(lock: dict[str, Any]) -> list[str]:
    """Why a lock cannot be trusted to name files and hashes, if it cannot."""
    problems: list[str] = []

    def download(label: str, entry: Any) -> None:
        if not isinstance(entry, dict):
            problems.append(f"{label} is not an object")
            return
        if not isinstance(entry.get("url"), str) or not entry["url"].startswith("https://"):
            problems.append(f"{label} has no https:// url")
        if not isinstance(entry.get("sha256"), str) or not _SHA256.match(entry["sha256"]):
            problems.append(f"{label} has no valid SHA-256")

    download("pyodide", lock.get("pyodide"))
    download("pyyaml", lock.get("pyyaml"))
    runtime = lock.get("pyodide") if isinstance(lock.get("pyodide"), dict) else {}
    per_file = runtime.get("file_sha256")
    if not isinstance(per_file, dict) or sorted(per_file) != sorted(PYODIDE_FILES) \
            or not all(isinstance(value, str) and _SHA256.match(value) for value in per_file.values()):
        problems.append("pyodide has no SHA-256 for each of its files (run --update-lock)")
    if not isinstance(lock.get("wheels"), list):
        problems.append("wheels is not a list")
        return problems
    for index, wheel in enumerate(lock["wheels"]):
        download(f"wheels[{index}]", wheel)
        if isinstance(wheel, dict) and not (isinstance(wheel.get("filename"), str)
                                            and _WHEEL_FILENAME.match(wheel["filename"])):
            problems.append(f"wheels[{index}] has a filename that is not a plain wheel name: {wheel.get('filename')!r}")
    return problems


def lock_matches_pins(lock: dict[str, Any]) -> list[str]:
    """Why the lock no longer matches the pins in this file, if it does not."""
    problems: list[str] = []
    if lock["pyodide"].get("version") != PINS["pyodide"]:
        problems.append(f"pyodide is pinned to {PINS['pyodide']} but the lock has "
                        f"{lock['pyodide'].get('version')}")
    locked = {w["name"]: w["version"] for w in lock["wheels"]}
    for import_name, _pypi, version, _tag in PINS["wheels"]:
        if locked.get(import_name) != version:
            problems.append(f"{import_name} is pinned to {version} but the lock has "
                            f"{locked.get(import_name)}")
    if lock["pyyaml"].get("version") != PINS["pyyaml"]:
        problems.append(f"pyyaml is pinned to {PINS['pyyaml']} but the lock has "
                        f"{lock['pyyaml'].get('version')}")
    return problems


def _json(url: str, opener: Callable[..., Any]) -> dict[str, Any]:
    with opener(urllib.request.Request(url, headers={"User-Agent": "eagent-build-engine"}),
                timeout=60) as response:
        return json.load(response)


def check_integrity(integrity: str | None, data: bytes) -> None:
    """The npm registry publishes an SRI string for each tarball; hold the download to it."""
    if not integrity:
        return
    for token in integrity.split():
        algorithm, _, expected = token.partition("-")
        if algorithm in ("sha256", "sha384", "sha512"):
            actual = base64.b64encode(hashlib.new(algorithm, data).digest()).decode("ascii")
            if actual != expected:
                raise BuildError(f"the Pyodide tarball does not match the {algorithm} integrity the npm "
                                 f"registry published for it; refusing to pin it")
            return
    raise BuildError(f"the npm integrity value {integrity!r} uses no algorithm this script can check")


def changed_hashes(previous: dict[str, Any], resolved: dict[str, Any]) -> list[str]:
    """Components whose version is unchanged but whose bytes are not.

    A version's file should never change. When a registry serves a different file
    under the same version, that is a republished or tampered release, and quietly
    re-pinning it would turn the lock into a rubber stamp.
    """
    changes: list[str] = []
    old_wheels = {wheel["name"]: wheel for wheel in previous.get("wheels", [])}
    for wheel in resolved["wheels"]:
        before = old_wheels.get(wheel["name"])
        if before and before.get("version") == wheel["version"] and before.get("sha256") != wheel["sha256"]:
            changes.append(f"{wheel['name']} {wheel['version']}: {before.get('sha256')} -> {wheel['sha256']}")
    for key in ("pyodide", "pyyaml"):
        before, now = previous.get(key) or {}, resolved[key]
        if before.get("version") == now["version"] and before.get("sha256") not in (None, now["sha256"]):
            changes.append(f"{key} {now['version']}: {before.get('sha256')} -> {now['sha256']}")
    return changes


def resolve_lock(cache: Path, *, opener: Callable[..., Any] = urllib.request.urlopen) -> dict[str, Any]:
    """Resolve :data:`PINS` to URLs and hashes from the registries they live on."""
    npm = _json(f"https://registry.npmjs.org/pyodide/{PINS['pyodide']}", opener)
    tarball_url = npm["dist"]["tarball"]
    with opener(urllib.request.Request(tarball_url), timeout=300) as response:
        tarball = response.read()
    check_integrity(npm["dist"].get("integrity"), tarball)
    runtime_files = pyodide_files(tarball)
    wheels = []
    for import_name, pypi_name, version, tag in PINS["wheels"]:
        release = _json(f"https://pypi.org/pypi/{pypi_name}/{version}/json", opener)
        chosen = [f for f in release["urls"] if f["packagetype"] == "bdist_wheel"
                  and f["filename"].endswith(f"-{tag}.whl")]
        if len(chosen) != 1:
            raise BuildError(f"{pypi_name} {version}: expected exactly one *-{tag}.whl, "
                             f"found {[f['filename'] for f in release['urls']]}")
        if not _WHEEL_FILENAME.match(chosen[0]["filename"]):
            raise BuildError(f"{pypi_name} {version}: the registry names the wheel {chosen[0]['filename']!r}, "
                             f"which is not a plain filename")
        wheels.append({"name": import_name, "version": version, "filename": chosen[0]["filename"],
                       "url": chosen[0]["url"], "sha256": chosen[0]["digests"]["sha256"],
                       "bytes": chosen[0]["size"]})
    yaml_release = _json(f"https://pypi.org/pypi/pyyaml/{PINS['pyyaml']}/json", opener)
    sdist = [f for f in yaml_release["urls"] if f["packagetype"] == "sdist"]
    if len(sdist) != 1:
        raise BuildError("expected exactly one PyYAML source distribution")
    return {
        "schema": LOCK_SCHEMA,
        "pyodide": {"version": PINS["pyodide"], "url": tarball_url, "sha256": sha256_bytes(tarball),
                    "integrity": npm["dist"].get("integrity"), "license": "MPL-2.0",
                    "files": list(PYODIDE_FILES),
                    "file_sha256": {name: sha256_bytes(runtime_files[name]) for name in PYODIDE_FILES}},
        "wheels": wheels,
        "pyyaml": {"version": PINS["pyyaml"], "url": sdist[0]["url"],
                   "sha256": sdist[0]["digests"]["sha256"], "license": "MIT"},
    }


# ---------------------------------------------------------------------------
# build and check
# ---------------------------------------------------------------------------

NOTICE_HEAD = """\
E-Agent browser engine: components and licences
================================================
This directory is built by scripts/build_engine.py from the pinned parts below.
Each is shipped unmodified, apart from PyYAML, of which only the pure-Python
package is used. Nothing here is a substitute for the licence of its component.

"""

NOTICE_TAIL = """\

eagent-project.zip holds the E-Agent Python package and the shipped task files
from this repository.

Every file's SHA-256 is in manifest.json, and the pinned sources and hashes are in
scripts/engine.lock.json.
"""


def notice(lock: dict[str, Any], python: str) -> str:
    """The component table, with every column lined up whatever the versions are."""
    versions = {w["name"]: w["version"] for w in lock["wheels"]}
    rows = [
        (f"Pyodide {lock['pyodide']['version']}", "MPL-2.0", "https://github.com/pyodide/pyodide"),
        (f"  CPython {python} (in pyodide.asm.wasm and python_stdlib.zip)", "PSF-2.0", ""),
        (f"pydantic {versions['pydantic']}", "MIT", "https://github.com/pydantic/pydantic"),
        (f"pydantic-core {versions['pydantic_core']}", "MIT", "https://github.com/pydantic/pydantic-core"),
        (f"typing-extensions {versions['typing_extensions']}", "PSF-2.0", "https://github.com/python/typing_extensions"),
        (f"annotated-types {versions['annotated_types']}", "MIT", "https://github.com/annotated-types/annotated-types"),
        (f"typing-inspection {versions['typing_inspection']}", "MIT", "https://github.com/pydantic/typing-inspection"),
        (f"PyYAML {lock['pyyaml']['version']} (its licence is in its zip)", "MIT", "https://github.com/yaml/pyyaml"),
    ]
    first = max(len(name) for name, _, _ in rows) + 2
    second = max(len(licence) for _, licence, _ in rows) + 2
    lines = [f"  {name.ljust(first)}{licence.ljust(second)}{url}".rstrip() for name, licence, url in rows]
    return NOTICE_HEAD + "\n".join(lines) + "\n" + NOTICE_TAIL


def build(out: Path, cache: Path, lock: dict[str, Any], *,
          opener: Callable[..., Any] = urllib.request.urlopen, root: Path = ROOT) -> dict[str, Any]:
    shape = lock_shape_problems(lock)
    if shape:
        raise BuildError("scripts/engine.lock.json is malformed:\n  " + "\n  ".join(shape))
    problems = lock_matches_pins(lock)
    if problems:
        raise BuildError("scripts/engine.lock.json is out of date:\n  " + "\n  ".join(problems)
                         + "\nrun: python scripts/build_engine.py --update-lock")
    if out.exists():
        shutil.rmtree(out)
    (out / "pyodide").mkdir(parents=True)
    (out / "wheels").mkdir()
    files: dict[str, bytes] = {}

    runtime = pyodide_files(fetch(lock["pyodide"]["url"], lock["pyodide"]["sha256"], cache, opener=opener))
    for name, data in runtime.items():
        if sha256_bytes(data) != lock["pyodide"]["file_sha256"][name]:
            raise BuildError(f"pyodide/{name} is not the file the lock recorded for it; "
                             f"refusing to build from a tarball whose contents changed")
        files[f"pyodide/{name}"] = data
    info = json.loads(runtime["pyodide-lock.json"])["info"]

    wheel_names = []
    for wheel in lock["wheels"]:
        files[f"wheels/{wheel['filename']}"] = fetch(wheel["url"], wheel["sha256"], cache, opener=opener)
        wheel_names.append(f"wheels/{wheel['filename']}")

    yaml_zip = f"wheels/pyyaml-{lock['pyyaml']['version']}-pure.zip"
    files[yaml_zip] = pyyaml_pure_zip(fetch(lock["pyyaml"]["url"], lock["pyyaml"]["sha256"], cache,
                                            opener=opener), lock["pyyaml"]["version"])
    files["eagent-project.zip"] = project_zip(root)

    files["NOTICE.txt"] = notice(lock, info["python"]).encode("utf-8")

    for name, data in files.items():
        target = out / name
        if not target.resolve().is_relative_to(out.resolve()):
            raise BuildError(f"{name} would be written outside {out}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

    listing = {name: {"bytes": len(data), "sha256": sha256_bytes(data)} for name, data in sorted(files.items())}
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "engine_schema": "eagent.engine.v1",
        "python": info["python"],
        "pyodide": lock["pyodide"]["version"],
        "abi": info.get("abi_version"),
        "install": {"wheels": wheel_names, "site_packages_zips": [yaml_zip],
                    "project": "eagent-project.zip"},
        "selftest": selftest_entries(),
        "files": listing,
        "total_bytes": sum(entry["bytes"] for entry in listing.values()),
        "lock_sha256": sha256_bytes(canonical(lock).encode("utf-8")),
    }
    manifest["bundle_sha256"] = sha256_bytes(canonical({"files": listing, "install": manifest["install"]}).encode("utf-8"))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def check(out: Path, lock: dict[str, Any], *, root: Path = ROOT) -> list[str]:
    """Everything wrong with an existing build, or an empty list."""
    manifest_path = out / "manifest.json"
    if not manifest_path.is_file():
        return [f"{manifest_path} does not exist; run scripts/build_engine.py"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    problems = [f"lock: {p}" for p in lock_matches_pins(lock)]
    if manifest.get("schema") != MANIFEST_SCHEMA:
        problems.append("manifest has an unexpected schema")
    if manifest.get("lock_sha256") != sha256_bytes(canonical(lock).encode("utf-8")):
        problems.append("the build was made from a different lock than scripts/engine.lock.json")
    for name, entry in manifest.get("files", {}).items():
        path = out / name
        if not path.is_file():
            problems.append(f"{name} is listed but missing")
        elif sha256_bytes(path.read_bytes()) != entry["sha256"]:
            problems.append(f"{name} does not match its recorded hash")
    # The manifest and the files can be changed together; the lock cannot be changed
    # by whoever changes them, so what it pins is held to its own hashes.
    listed = manifest.get("files", {})
    pinned = {f"wheels/{wheel['filename']}": wheel["sha256"] for wheel in lock["wheels"]}
    pinned.update({f"pyodide/{name}": digest for name, digest in lock["pyodide"]["file_sha256"].items()})
    for name, digest in sorted(pinned.items()):
        if name not in listed:
            problems.append(f"{name} is pinned by the lock but not in the manifest")
        elif listed[name].get("sha256") != digest:
            problems.append(f"{name} is not the file the lock pins")
    unlisted = sorted(p.relative_to(out).as_posix() for p in out.rglob("*")
                      if p.is_file() and p.name != "manifest.json"
                      and p.relative_to(out).as_posix() not in manifest.get("files", {}))
    problems += [f"{name} is present but not in the manifest" for name in unlisted]
    expected = manifest.get("files", {}).get("eagent-project.zip", {}).get("sha256")
    if expected and sha256_bytes(project_zip(root)) != expected:
        problems.append("eagent-project.zip is stale: the source tree has changed since this build")
    if manifest.get("selftest") != selftest_entries():
        problems.append("the recorded self-test answers differ from what this checkout computes")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output directory (default web/engine)")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE, help="download cache (default .cache/engine)")
    parser.add_argument("--lock", type=Path, default=LOCK_PATH)
    parser.add_argument("--check", action="store_true", help="verify an existing build and exit")
    parser.add_argument("--update-lock", action="store_true",
                        help="re-resolve the pinned versions to URLs and hashes, rewrite the lock and exit")
    parser.add_argument("--accept-changed-hashes", action="store_true",
                        help="with --update-lock: accept a registry serving different bytes for a version "
                             "the lock already pins (a version's file should never change)")
    args = parser.parse_args(argv)
    try:
        if args.update_lock:
            lock = resolve_lock(args.cache)
            if args.lock.is_file():
                try:
                    previous = json.loads(args.lock.read_text(encoding="utf-8"))
                except ValueError:
                    previous = {}
                changes = changed_hashes(previous, lock)
                if changes and not args.accept_changed_hashes:
                    raise BuildError("the registries now serve different bytes for versions the lock already pins:\n  "
                                     + "\n  ".join(changes)
                                     + "\nA version's file should never change. If you know why, rerun with "
                                       "--accept-changed-hashes.")
            args.lock.write_text(json.dumps(lock, indent=1, sort_keys=True) + "\n", encoding="utf-8")
            print(f"wrote {args.lock.relative_to(ROOT) if args.lock.is_relative_to(ROOT) else args.lock}")
            return 0
        lock = load_lock(args.lock)
        if args.check:
            problems = check(args.out, lock)
            for problem in problems:
                print("PROBLEM:", problem)
            print("engine build is consistent" if not problems else f"{len(problems)} problem(s)")
            return 1 if problems else 0
        manifest = build(args.out, args.cache, lock)
    except BuildError as error:
        print(f"build_engine: {error}", file=sys.stderr)
        return 1
    print(f"built the engine into {args.out}: {len(manifest['files'])} files, "
          f"{manifest['total_bytes'] / 1e6:.1f} MB, Python {manifest['python']} (Pyodide {manifest['pyodide']})")
    for name, entry in manifest["files"].items():
        print(f"  {entry['bytes'] / 1e6:7.2f} MB  {name}")
    print(f"bundle sha256 {manifest['bundle_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
