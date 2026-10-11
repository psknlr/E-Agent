"""The engine build ships only what it can prove, and the same bytes every time.

Nothing here touches the network: downloads are served by a fake opener that
returns small synthetic archives, so these tests cover the build's decisions --
what it verifies, what it refuses, what it records -- not the registries.
"""

from __future__ import annotations

import importlib.util
import io
import json
import tarfile
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("build_engine", ROOT / "scripts" / "build_engine.py")
build_engine = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_engine)

sha256 = build_engine.sha256_bytes
BuildError = build_engine.BuildError


# ---------------------------------------------------------------------------
# synthetic registries
# ---------------------------------------------------------------------------

def tar_gz(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


PYODIDE_TARBALL = tar_gz({
    **{f"package/{name}": f"contents of {name}".encode() for name in build_engine.PYODIDE_FILES},
    "package/pyodide-lock.json": json.dumps({"info": {"python": "3.14.2", "abi_version": "2026_0"}}).encode(),
    "package/README.md": b"not shipped",
})
PYYAML_SDIST = tar_gz({
    "pyyaml-6.0.3/lib/yaml/__init__.py": b"# yaml package\n",
    "pyyaml-6.0.3/lib/yaml/loader.py": b"# loader\n",
    "pyyaml-6.0.3/yaml/_yaml.pyx": b"# C extension source, not shipped\n",
    "pyyaml-6.0.3/LICENSE": b"MIT licence text\n",
    "pyyaml-6.0.3/tests/LICENSE": b"not the top-level licence\n",
})


class Response:
    def __init__(self, data: bytes):
        self._data = data

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self) -> bytes:
        return self._data


class Registry:
    """An opener that serves a fixed URL -> bytes map and records what was asked."""

    def __init__(self, files: dict[str, bytes]):
        self.files = files
        self.requested: list[str] = []

    def __call__(self, request, timeout=None):
        url = request.full_url if hasattr(request, "full_url") else request
        self.requested.append(url)
        if url not in self.files:
            raise OSError(f"no such url {url}")
        return Response(self.files[url])


def wheel_bytes(name: str) -> bytes:
    return f"wheel {name}".encode()


@pytest.fixture
def lock_and_registry():
    wheels = []
    files = {"https://registry.example/pyodide.tgz": PYODIDE_TARBALL,
             "https://files.example/pyyaml.tar.gz": PYYAML_SDIST}
    for import_name, _pypi, version, tag in build_engine.PINS["wheels"]:
        filename = f"{import_name}-{version}-{tag}.whl"
        url = f"https://files.example/{filename}"
        data = wheel_bytes(filename)
        files[url] = data
        wheels.append({"name": import_name, "version": version, "filename": filename, "url": url,
                       "sha256": sha256(data), "bytes": len(data)})
    lock = {
        "schema": build_engine.LOCK_SCHEMA,
        "pyodide": {"version": build_engine.PINS["pyodide"], "url": "https://registry.example/pyodide.tgz",
                    "sha256": sha256(PYODIDE_TARBALL), "license": "MPL-2.0",
                    "files": list(build_engine.PYODIDE_FILES)},
        "wheels": wheels,
        "pyyaml": {"version": build_engine.PINS["pyyaml"], "url": "https://files.example/pyyaml.tar.gz",
                   "sha256": sha256(PYYAML_SDIST), "license": "MIT"},
    }
    return lock, Registry(files)


@pytest.fixture
def project(tmp_path):
    """A stand-in repository holding exactly what the engine ships and some that it must not."""
    files = {
        "src/eagent/__init__.py": "VERSION = 1\n",
        "src/eagent/engine/__init__.py": "# engine\n",
        "src/eagent/engine/__pycache__/core.cpython-314.pyc": "compiled",
        "src/eagent/data.json": "{}",
        "configs/tasks/T1.yaml": "task_id: T1\n",
        "configs/tasks/notes.md": "not a task",
        "configs/templates/reaction/r.yaml": "template: r\n",
        "configs/templates/reaction/notes.txt": "not a template",
        "configs/datasources/s.yaml": "not shipped",
        "docs/guide.md": "not shipped",
        "tests/test_x.py": "not shipped",
    }
    for name, text in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return tmp_path


# ---------------------------------------------------------------------------
# deterministic zips
# ---------------------------------------------------------------------------

class TestStoredZip:
    def test_the_same_entries_give_the_same_bytes_in_any_order(self):
        entries = [("b/two.py", b"2"), ("a/one.py", b"1"), ("c.txt", b"3")]
        assert build_engine.stored_zip(entries) == build_engine.stored_zip(list(reversed(entries)))

    def test_entries_are_stored_sorted_and_untimed(self):
        data = build_engine.stored_zip([("z.py", b"z" * 500), ("a.py", b"a" * 500)])
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            infos = archive.infolist()
        assert [info.filename for info in infos] == ["a.py", "z.py"]
        assert {info.compress_type for info in infos} == {zipfile.ZIP_STORED}
        assert {info.date_time for info in infos} == {build_engine.FIXED_TIME}
        assert len(data) > 1000, "stored, not deflated: 1000 repeated bytes stay 1000 bytes"

    def test_different_content_gives_different_bytes(self):
        assert build_engine.stored_zip([("a", b"1")]) != build_engine.stored_zip([("a", b"2")])



# ---------------------------------------------------------------------------
# fetch
# ---------------------------------------------------------------------------

class TestFetch:
    def test_a_download_is_verified_and_cached_by_its_hash(self, tmp_path):
        data = b"payload"
        registry = Registry({"https://x.example/a": data})
        got = build_engine.fetch("https://x.example/a", sha256(data), tmp_path, opener=registry)
        assert got == data
        assert (tmp_path / sha256(data)).read_bytes() == data
        assert build_engine.fetch("https://x.example/a", sha256(data), tmp_path, opener=registry) == data
        assert registry.requested == ["https://x.example/a"], "the second call is served from the cache"

    @pytest.mark.parametrize("url", ["http://x.example/a", "file:///etc/passwd", "ftp://x.example/a", "//x.example/a"])
    def test_only_https_downloads_are_accepted(self, tmp_path, url):
        registry = Registry({url: b"payload"})
        with pytest.raises(BuildError, match="only https"):
            build_engine.fetch(url, sha256(b"payload"), tmp_path, opener=registry)
        assert registry.requested == []

    def test_a_download_that_is_not_the_pinned_file_aborts_the_build(self, tmp_path):
        registry = Registry({"https://x.example/a": b"something else"})
        with pytest.raises(BuildError, match="not the one that was pinned"):
            build_engine.fetch("https://x.example/a", sha256(b"payload"), tmp_path, opener=registry)
        assert list(tmp_path.iterdir()) == [], "a rejected download is never cached"

    def test_a_corrupt_cache_entry_is_replaced_not_trusted(self, tmp_path):
        data = b"payload"
        (tmp_path / sha256(data)).write_bytes(b"corrupt")
        registry = Registry({"https://x.example/a": data})
        assert build_engine.fetch("https://x.example/a", sha256(data), tmp_path, opener=registry) == data
        assert (tmp_path / sha256(data)).read_bytes() == data
        assert registry.requested == ["https://x.example/a"]



# ---------------------------------------------------------------------------
# what is shipped
# ---------------------------------------------------------------------------

class TestProjectPayload:
    def test_only_the_package_the_task_files_and_the_templates_are_shipped(self, project):
        names = sorted(name for name, _ in build_engine.project_entries(project))
        assert names == ["configs/tasks/T1.yaml", "configs/templates/reaction/r.yaml",
                         "src/eagent/__init__.py", "src/eagent/engine/__init__.py"]

    def test_a_tree_without_the_engine_is_refused(self, project):
        (project / "src/eagent/engine/__init__.py").unlink()
        with pytest.raises(BuildError, match="src/eagent/engine is missing"):
            build_engine.project_entries(project)

    def test_the_zip_is_a_function_of_the_files_alone(self, project):
        first = build_engine.project_zip(project)
        (project / "src/eagent/__init__.py").touch()
        assert build_engine.project_zip(project) == first
        (project / "src/eagent/__init__.py").write_text("VERSION = 2\n")
        assert build_engine.project_zip(project) != first



class TestRuntimeParts:
    def test_pyyaml_is_taken_as_its_pure_python_package_and_licence(self):
        data = build_engine.pyyaml_pure_zip(PYYAML_SDIST, "6.0.3")
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            assert archive.namelist() == ["PyYAML-6.0.3.dist-info/LICENSE", "yaml/__init__.py", "yaml/loader.py"]
            assert archive.read("PyYAML-6.0.3.dist-info/LICENSE") == b"MIT licence text\n"

    def test_a_pyyaml_source_distribution_with_a_traversing_path_is_refused(self):
        hostile = tar_gz({"pyyaml-6.0.3/lib/yaml/../../evil.py": b"x", "pyyaml-6.0.3/LICENSE": b"x"})
        with pytest.raises(BuildError, match="unsafe path"):
            build_engine.pyyaml_pure_zip(hostile, "6.0.3")

    @pytest.mark.parametrize("members,message", [
        ({"pyyaml-6.0.3/LICENSE": b"x"}, "no lib/yaml package"),
        ({"pyyaml-6.0.3/lib/yaml/__init__.py": b"x"}, "no LICENSE"),
    ])
    def test_a_pyyaml_source_distribution_missing_a_part_is_refused(self, members, message):
        with pytest.raises(BuildError, match=message):
            build_engine.pyyaml_pure_zip(tar_gz(members), "6.0.3")

    def test_the_runtime_files_are_taken_by_exact_name(self):
        found = build_engine.pyodide_files(PYODIDE_TARBALL)
        assert sorted(found) == sorted(build_engine.PYODIDE_FILES)
        assert found["pyodide.mjs"] == b"contents of pyodide.mjs"

    def test_a_runtime_tarball_missing_a_file_is_refused(self):
        partial = tar_gz({"package/pyodide.mjs": b"x"})
        with pytest.raises(BuildError, match="has no package/"):
            build_engine.pyodide_files(partial)



# ---------------------------------------------------------------------------
# the lock
# ---------------------------------------------------------------------------

class TestTheLock:
    def test_the_committed_lock_matches_the_pins_in_the_script(self):
        lock = build_engine.load_lock()
        assert build_engine.lock_matches_pins(lock) == []

    def test_every_locked_download_is_https_and_hashed(self):
        lock = build_engine.load_lock()
        entries = [lock["pyodide"], lock["pyyaml"], *lock["wheels"]]
        for entry in entries:
            assert entry["url"].startswith("https://"), entry["url"]
            assert len(entry["sha256"]) == 64 and set(entry["sha256"]) <= set("0123456789abcdef")

    def test_a_drifted_pin_is_named(self, lock_and_registry):
        lock, _ = lock_and_registry
        lock["wheels"][3]["version"] = "0.0.1"
        lock["pyyaml"]["version"] = "1.0"
        problems = build_engine.lock_matches_pins(lock)
        assert any("pydantic_core" in problem for problem in problems)
        assert any("pyyaml" in problem for problem in problems)

    def test_a_missing_or_foreign_lock_is_refused(self, tmp_path):
        with pytest.raises(BuildError, match="does not exist"):
            build_engine.load_lock(tmp_path / "none.json")
        wrong = tmp_path / "wrong.json"
        wrong.write_text(json.dumps({"schema": "something.else"}))
        with pytest.raises(BuildError, match="is not a"):
            build_engine.load_lock(wrong)

    def test_resolving_picks_exactly_the_pinned_wheel_and_records_its_hash(self, tmp_path):
        files = {"https://registry.npmjs.org/pyodide/" + build_engine.PINS["pyodide"]: json.dumps(
            {"dist": {"tarball": "https://registry.example/pyodide.tgz", "integrity": "sha512-x"}}).encode(),
            "https://registry.example/pyodide.tgz": PYODIDE_TARBALL}
        for import_name, pypi, version, tag in build_engine.PINS["wheels"]:
            wanted = f"{import_name}-{version}-{tag}.whl"
            files[f"https://pypi.org/pypi/{pypi}/{version}/json"] = json.dumps({"urls": [
                {"packagetype": "sdist", "filename": f"{pypi}-{version}.tar.gz", "url": "https://f.example/sdist",
                 "digests": {"sha256": "1" * 64}, "size": 1},
                {"packagetype": "bdist_wheel", "filename": f"{import_name}-{version}-cp39-cp39-win_amd64.whl",
                 "url": "https://f.example/win", "digests": {"sha256": "2" * 64}, "size": 2},
                {"packagetype": "bdist_wheel", "filename": wanted, "url": f"https://f.example/{wanted}",
                 "digests": {"sha256": "3" * 64}, "size": 3},
            ]}).encode()
        files[f"https://pypi.org/pypi/pyyaml/{build_engine.PINS['pyyaml']}/json"] = json.dumps({"urls": [
            {"packagetype": "bdist_wheel", "filename": "x.whl", "url": "https://f.example/w", "digests": {"sha256": "4" * 64}, "size": 4},
            {"packagetype": "sdist", "filename": "pyyaml.tar.gz", "url": "https://f.example/pyyaml.tar.gz",
             "digests": {"sha256": "5" * 64}, "size": 5}]}).encode()
        lock = build_engine.resolve_lock(tmp_path, opener=Registry(files))
        assert build_engine.lock_matches_pins(lock) == []
        assert {wheel["name"]: wheel["sha256"] for wheel in lock["wheels"]} == {n: "3" * 64 for n, *_ in build_engine.PINS["wheels"]}
        assert lock["pyyaml"]["sha256"] == "5" * 64
        assert lock["pyodide"]["sha256"] == sha256(PYODIDE_TARBALL)

    def test_resolving_refuses_an_ambiguous_release(self, tmp_path):
        files = {"https://registry.npmjs.org/pyodide/" + build_engine.PINS["pyodide"]: json.dumps(
            {"dist": {"tarball": "https://registry.example/pyodide.tgz"}}).encode(),
            "https://registry.example/pyodide.tgz": PYODIDE_TARBALL}
        import_name, pypi, version, tag = build_engine.PINS["wheels"][0]
        files[f"https://pypi.org/pypi/{pypi}/{version}/json"] = json.dumps({"urls": []}).encode()
        with pytest.raises(BuildError, match="expected exactly one"):
            build_engine.resolve_lock(tmp_path, opener=Registry(files))



# ---------------------------------------------------------------------------
# build and check
# ---------------------------------------------------------------------------

@pytest.fixture
def built(tmp_path, lock_and_registry, project):
    lock, registry = lock_and_registry
    out = tmp_path / "engine"
    manifest = build_engine.build(out, tmp_path / "cache", lock, opener=registry, root=project)
    return out, lock, registry, project, manifest


class TestBuild:
    def test_the_manifest_records_every_file_and_how_to_install_it(self, built):
        out, lock, _, _, manifest = built
        on_disk = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file() and p.name != "manifest.json")
        assert sorted(manifest["files"]) == on_disk
        for name, entry in manifest["files"].items():
            data = (out / name).read_bytes()
            assert (entry["bytes"], entry["sha256"]) == (len(data), sha256(data)), name
        assert manifest["python"] == "3.14.2"
        assert manifest["install"]["project"] == "eagent-project.zip"
        assert len(manifest["install"]["wheels"]) == len(build_engine.PINS["wheels"])
        assert manifest["install"]["site_packages_zips"] == ["wheels/pyyaml-6.0.3-pure.zip"]
        assert manifest["total_bytes"] == sum(entry["bytes"] for entry in manifest["files"].values())
        assert json.loads((out / "manifest.json").read_text()) == manifest

    def test_the_self_test_is_what_the_agents_answer_natively(self, built):
        from eagent.engine import call
        _, _, _, _, manifest = built
        assert [entry["agent"] for entry in manifest["selftest"]] == [agent for agent, _ in build_engine.SELFTEST]
        for entry in manifest["selftest"]:
            assert json.loads(call(entry["agent"], entry["arguments"]))["cite"]["sha256"] == entry["sha256"]

    def test_the_licences_are_shipped_beside_the_components(self, built):
        out, *_ = built
        notice = (out / "NOTICE.txt").read_text()
        for name in ("Pyodide 314.0.7", "pydantic 2.14.0", "PyYAML 6.0.3", "MPL-2.0", "3.14.2"):
            assert name in notice, name

    def test_two_builds_of_the_same_inputs_are_byte_identical(self, built, tmp_path):
        out, lock, registry, project, manifest = built
        again = build_engine.build(tmp_path / "engine2", tmp_path / "cache2", lock, opener=registry, root=project)
        assert again["bundle_sha256"] == manifest["bundle_sha256"]
        assert (tmp_path / "engine2" / "eagent-project.zip").read_bytes() == (out / "eagent-project.zip").read_bytes()
        assert (tmp_path / "engine2" / "manifest.json").read_bytes() == (out / "manifest.json").read_bytes()

    def test_a_build_replaces_what_was_there_rather_than_adding_to_it(self, built):
        out, lock, registry, project, _ = built
        (out / "left-over.bin").write_bytes(b"from an older build")
        build_engine.build(out, out.parent / "cache", lock, opener=registry, root=project)
        assert not (out / "left-over.bin").exists()

    def test_a_lock_that_disagrees_with_the_pins_builds_nothing(self, tmp_path, lock_and_registry, project):
        lock, registry = lock_and_registry
        lock["pyodide"]["version"] = "0.0.1"
        out = tmp_path / "engine"
        with pytest.raises(BuildError, match="out of date"):
            build_engine.build(out, tmp_path / "cache", lock, opener=registry, root=project)
        assert not out.exists()
        assert registry.requested == [], "nothing is downloaded for a lock that cannot be used"

    def test_a_download_that_is_not_the_pinned_file_builds_nothing_usable(self, tmp_path, lock_and_registry, project):
        lock, registry = lock_and_registry
        registry.files[lock["wheels"][0]["url"]] = b"a different wheel"
        with pytest.raises(BuildError, match="not the one that was pinned"):
            build_engine.build(tmp_path / "engine", tmp_path / "cache", lock, opener=registry, root=project)
        assert not (tmp_path / "engine" / "manifest.json").exists(), "no manifest, so nothing deployable"



class TestCheck:
    def test_a_fresh_build_is_consistent(self, built):
        out, lock, _, project, _ = built
        assert build_engine.check(out, lock, root=project) == []

    def test_a_missing_build_is_reported(self, tmp_path, lock_and_registry):
        lock, _ = lock_and_registry
        assert "does not exist" in build_engine.check(tmp_path / "none", lock)[0]

    def test_a_changed_file_is_reported_by_name(self, built):
        out, lock, _, project, _ = built
        target = out / "pyodide" / "pyodide.asm.wasm"
        target.write_bytes(target.read_bytes() + b"!")
        assert build_engine.check(out, lock, root=project) == ["pyodide/pyodide.asm.wasm does not match its recorded hash"]

    def test_a_missing_and_an_unlisted_file_are_both_reported(self, built):
        out, lock, _, project, _ = built
        (out / "NOTICE.txt").unlink()
        (out / "extra.js").write_text("alert(1)")
        problems = build_engine.check(out, lock, root=project)
        assert "NOTICE.txt is listed but missing" in problems
        assert "extra.js is present but not in the manifest" in problems

    def test_source_changes_since_the_build_make_the_project_zip_stale(self, built):
        out, lock, _, project, _ = built
        (project / "src/eagent/__init__.py").write_text("VERSION = 2\n")
        assert any("eagent-project.zip is stale" in problem for problem in build_engine.check(out, lock, root=project))

    def test_a_build_from_another_lock_is_reported(self, built):
        out, lock, _, project, _ = built
        lock["pyyaml"]["license"] = "changed"
        assert any("different lock" in problem for problem in build_engine.check(out, lock, root=project))

    def test_stale_self_test_answers_are_reported(self, built):
        out, lock, _, project, manifest = built
        manifest["selftest"][0]["sha256"] = "0" * 64
        (out / "manifest.json").write_text(json.dumps(manifest))
        assert any("self-test" in problem for problem in build_engine.check(out, lock, root=project))

