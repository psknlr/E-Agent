"""The site exporter ships a verified copy of the page, or a redirect, and nothing else."""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("export_site", ROOT / "scripts" / "export_site.py")
export_site = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(export_site)

TARGET = "https://psknlr.github.io/E-Agent/"


@pytest.fixture
def checked(monkeypatch):
    """The output directories whose copied engine the exporter was asked to verify."""
    seen: list[Path] = []
    monkeypatch.setattr(export_site, "check_engine", lambda out: seen.append(out))
    return seen


@pytest.fixture
def web(tmp_path, monkeypatch, checked):
    """A stand-in web/ directory that satisfies the exporter's checks."""
    site = tmp_path / "source-web"
    site.mkdir()
    index = "<!doctype html>" + "".join(f'<script src="./{name}"></script>' for name in export_site.PAGE_ASSETS)
    files = {
        "index.html": index,
        "styles.css": "body{}",
        "config.json": "{}",
        "reference-data.json": json.dumps({"schema_version": 1, "system_prompt": "x"}),
        "agent-catalog.json": json.dumps({"schema": "eagent.engine.v1", "agents": [{"name": "wilson_interval"}]}),
        "engine/manifest.json": "{}",
        "engine/pyodide/pyodide.mjs": "// loader",
        "engine/__pycache__/junk.pyc": "x",
    }
    for name in export_site.REQUIRED:
        files.setdefault(name, "// script")
    for name, text in files.items():
        path = site / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    monkeypatch.setattr(export_site, "WEB", site)
    return site


def run(*args):
    return export_site.main([str(a) for a in args])


class TestTheSiteCopy:
    def test_a_full_export_carries_the_engine_the_headers_and_the_domain(self, web, checked, tmp_path, capsys):
        out = tmp_path / "out"
        assert run("--out", out, "--cname", "Enzyme.Impf.AI") == 0
        assert (out / "engine" / "pyodide" / "pyodide.mjs").is_file()
        assert (out / "CNAME").read_text() == "enzyme.impf.ai\n"
        assert json.loads((out / "config.json").read_text()) == {"backend_url": ""}
        headers = (out / "_headers").read_text()
        assert "/engine/pyodide/pyodide.mjs\n  Content-Type: text/javascript" in headers
        assert "/engine/pyodide/pyodide.asm.wasm\n  Content-Type: application/wasm" in headers
        assert checked == [out], "the copied engine is checked against its manifest"
        assert not list(out.rglob("*.pyc"))
        assert "compute engine: included" in capsys.readouterr().out

    def test_without_the_engine_nothing_of_it_is_copied_or_checked(self, web, checked, tmp_path, capsys):
        out = tmp_path / "slim"
        assert run("--out", out, "--without-engine") == 0
        assert not (out / "engine").exists()
        assert not (out / "_headers").exists()
        assert checked == []
        assert (out / "agent-catalog.json").is_file(), "the page still describes its agents"
        assert "not included" in capsys.readouterr().out

    def test_a_pinned_backend_goes_into_config_json(self, web, tmp_path):
        out = tmp_path / "out"
        run("--out", out, "--backend-url", "https://eagent-api.impf.ai/")
        assert json.loads((out / "config.json").read_text()) == {"backend_url": "https://eagent-api.impf.ai"}

    def test_it_will_not_overwrite_the_source(self, web):
        with pytest.raises(SystemExit, match="refusing to use .* as an output directory"):
            export_site.copy_site(web)

    @pytest.mark.parametrize("missing", ["engine-worker.js", "agents.js", "agent-catalog.json", "reference-data.json"])
    def test_a_missing_required_file_stops_the_export(self, web, tmp_path, missing):
        (web / missing).unlink()
        with pytest.raises(SystemExit, match=re.escape(missing)):
            run("--out", tmp_path / "out")

    def test_an_index_that_stops_loading_a_script_stops_the_export(self, web, tmp_path):
        text = (web / "index.html").read_text().replace('<script src="./compute-ui.js"></script>', "")
        (web / "index.html").write_text(text)
        with pytest.raises(SystemExit, match="no longer references compute-ui.js"):
            run("--out", tmp_path / "out", "--without-engine")

    def test_invalid_json_and_a_foreign_catalog_stop_the_export(self, web, tmp_path):
        (web / "agent-catalog.json").write_text("{not json")
        with pytest.raises(SystemExit, match="agent-catalog.json is not valid JSON"):
            run("--out", tmp_path / "a", "--without-engine")
        (web / "agent-catalog.json").write_text(json.dumps({"schema": "other", "agents": []}))
        with pytest.raises(SystemExit, match="agent-catalog.json has an unexpected shape"):
            run("--out", tmp_path / "b", "--without-engine")

    def test_a_missing_engine_is_a_clear_instruction(self, tmp_path):
        out = tmp_path / "bare"
        out.mkdir()
        with pytest.raises(SystemExit, match="build_engine.py"):
            export_site.check_engine(out)


class TestTheRealSite:
    def test_the_repositorys_own_web_directory_passes_its_own_checks(self, tmp_path):
        out = tmp_path / "real"
        export_site.copy_site(out, with_engine=False)
        export_site.check_integrity(out, with_engine=False)

    def test_every_script_the_page_loads_is_exported_and_checked(self):
        index = (ROOT / "web" / "index.html").read_text()
        loaded = re.findall(r'<script src="\./([^"?]+)', index)
        assert loaded, "index.html loads scripts"
        for name in loaded:
            assert (ROOT / "web" / name).is_file(), name
            assert name in export_site.PAGE_ASSETS, f"{name} is loaded by the page but not checked by the exporter"
            assert name in export_site.REQUIRED, name
        assert "engine-worker.js" in export_site.REQUIRED, "the worker is started by code, not by a script tag"

    def test_the_engine_is_never_part_of_the_slim_copy(self, tmp_path):
        out = tmp_path / "real"
        names = {p.name for p in export_site.copy_site(out, with_engine=False)}
        assert "manifest.json" not in names
        assert not (out / "engine").exists()


class TestTheRedirect:
    def test_it_writes_only_what_a_host_needs_to_send_visitors_on(self, tmp_path, capsys):
        out = tmp_path / "redirect"
        assert run("--redirect-to", TARGET, "--cname", "enzyme.impf.ai", "--out", out) == 0
        assert sorted(p.name for p in out.iterdir()) == [".htaccess", "404.html", "CNAME", "_redirects", "index.html",
                                                          "nginx-redirect.conf.example", "vercel.json"]
        for page in ("index.html", "404.html"):
            text = (out / page).read_text()
            assert f'<meta http-equiv="refresh" content="0; url={TARGET}">' in text
            assert f'location.replace("{TARGET}");' in text
            assert f'<link rel="canonical" href="{TARGET}">' in text
            assert 'name="robots" content="noindex"' in text
        assert (out / "_redirects").read_text() == f"/*  {TARGET}  301\n"
        assert json.loads((out / "vercel.json").read_text()) == {"redirects": [{"source": "/(.*)", "destination": TARGET, "permanent": True}]}
        assert f"RewriteRule ^ {TARGET} [R=301,L]" in (out / ".htaccess").read_text()
        assert f"return 301 {TARGET};" in (out / "nginx-redirect.conf.example").read_text()
        assert (out / "CNAME").read_text() == "enzyme.impf.ai\n"
        assert "Wrote a redirect" in capsys.readouterr().out

    def test_it_copies_none_of_the_site(self, tmp_path):
        out = tmp_path / "redirect"
        run("--redirect-to", TARGET, "--out", out)
        assert not (out / "app.js").exists() and not (out / "engine").exists() and not (out / "CNAME").exists()

    @pytest.mark.parametrize("target", [
        "javascript:alert(1)", "http://example.org/", "ftp://example.org/", "//example.org/", "https://",
        "https://user:pw@example.org/", "https://example.org/?x=1", "https://example.org/#x",
        'https://example.org/"><script>alert(1)</script>', "https://example.org/<x>", "https://exa mple.org/",
        "https://example.org/a b", "https://example.org/\nSet-Cookie: x=1", "https://example.org/a;b",
        "https://example.org/a&b", "https://example.org/'x", "https://localhost/", "https://-bad.example.org/",
        "https://example.org/\\evil",
    ])
    def test_a_target_that_is_not_a_plain_https_url_is_refused(self, tmp_path, target):
        out = tmp_path / "redirect"
        with pytest.raises(SystemExit):
            run("--redirect-to", target, "--out", out)
        assert not out.exists(), "nothing is written for a refused target"

    @pytest.mark.parametrize("target", [TARGET, "https://enzyme.example.org/", "https://example.org:8443/a/b-c_d~e/", "https://example.org"])
    def test_ordinary_targets_are_accepted_unchanged(self, target):
        assert export_site.validate_redirect_target(f"  {target}  ") == target

    def test_a_redirect_cannot_be_combined_with_what_it_does_not_use(self, tmp_path):
        for extra in (["--backend-url", "https://eagent-api.impf.ai"], ["--without-engine"]):
            with pytest.raises(SystemExit):
                run("--redirect-to", TARGET, "--out", tmp_path / "x", *extra)

    def test_it_will_not_write_into_the_source_directory(self, web):
        with pytest.raises(SystemExit, match="refusing to write inside the source web/ directory"):
            export_site.write_redirect_site(web / "nested", TARGET)


class TestItOnlyDeletesWhatItMade:
    """The exporter replaces its output directory, which means deleting it.

    ``--out .`` removed the repository's own ``scripts/`` and ``web/`` in a scratch
    copy, and the same flag with ``~/Documents`` would have removed those. These run
    against a stand-in repository so the proof that nothing is deleted is a listing.
    """

    @pytest.fixture
    def repo(self, tmp_path, web, monkeypatch):
        repo = tmp_path / "repo"
        for name in ("scripts/build.py", "docs/readme.md", "src/code.py"):
            (repo / name).parent.mkdir(parents=True, exist_ok=True)
            (repo / name).write_text("precious")
        monkeypatch.setattr(export_site, "ROOT", repo)
        monkeypatch.chdir(repo)
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        (tmp_path / "home").mkdir()
        return repo

    def untouched(self, repo):
        assert sorted(p.relative_to(repo).as_posix() for p in repo.rglob("*") if p.is_file()) == [
            "docs/readme.md", "scripts/build.py", "src/code.py"]

    @pytest.mark.parametrize("flags", [[], ["--redirect-to", TARGET]])
    def test_the_repository_its_parent_the_current_directory_and_home_are_refused(self, repo, tmp_path, flags):
        for place in (repo, ".", repo / "..", tmp_path, tmp_path / "home", "/"):
            with pytest.raises(SystemExit, match="refusing to use .* as an output directory"):
                run("--out", place, *flags)
        self.untouched(repo)
        assert (tmp_path / "home").is_dir()

    @pytest.mark.parametrize("flags", [[], ["--redirect-to", TARGET]])
    def test_a_folder_it_did_not_make_is_never_deleted(self, repo, tmp_path, flags):
        other = tmp_path / "Documents"
        other.mkdir()
        (other / "thesis.txt").write_text("years of work")
        with pytest.raises(SystemExit, match="was not made by this script, so it will not be deleted"):
            run("--out", other, *flags)
        assert (other / "thesis.txt").read_text() == "years of work"

    def test_a_file_is_not_an_output_directory(self, repo, tmp_path):
        (tmp_path / "file").write_text("x")
        with pytest.raises(SystemExit, match="is not a directory"):
            run("--out", tmp_path / "file")

    @pytest.mark.parametrize("flags", [[], ["--redirect-to", TARGET]])
    def test_its_own_earlier_export_is_replaced_and_an_empty_folder_is_used(self, repo, tmp_path, flags):
        out = tmp_path / "dist" / "site"
        assert run("--out", out, *flags) == 0
        (out / "stale.txt").write_text("left over")
        assert run("--out", out, *flags) == 0
        assert not (out / "stale.txt").exists(), "an earlier export is replaced, not added to"
        assert (tmp_path / "dist" / ".site.eagent-export").is_file()
        empty = tmp_path / "empty"
        empty.mkdir()
        assert run("--out", empty, *flags) == 0

    def test_the_marker_is_beside_the_folder_so_the_upload_is_exactly_the_deliverable(self, repo, tmp_path):
        out = tmp_path / "redirect"
        assert run("--redirect-to", TARGET, "--out", out) == 0
        assert not any(p.name.endswith("eagent-export") for p in out.rglob("*"))
        assert marker_exists(out)


def marker_exists(out):
    return export_site.marker_for(out.resolve()).is_file()
