"""The research package a run hands over, and the check that it is intact.

One module today, :mod:`eagent.deliverables.bundle`. It is a package rather
than a single file because the deliverables are the part of this system that
outlives the run: the format will grow new items, and a reader of a bundle
written a year ago needs the rules it was written under to still be findable
by name.

The names are re-exported lazily for the same reason they are in
:mod:`eagent.tools`: importing a bundler pulls in the structure reader used
to validate converted coordinate files, and a caller that only wants
:func:`verify_bundle` -- typically the recipient of a package, checking it on
a machine that never ran the pipeline -- should not pay for that.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

_LAZY_EXPORTS: dict[str, str] = {
    "BATCH_ITEM_NAME": "bundle",
    "BUNDLE_FORMAT": "bundle",
    "BUNDLE_ITEMS": "bundle",
    "BUNDLE_MANIFEST_NAME": "bundle",
    "BundleEntry": "bundle",
    "BundleItem": "bundle",
    "BundleManifest": "bundle",
    "BundleResult": "bundle",
    "BundleVerification": "bundle",
    "CONFIDENCE_POLICY": "bundle",
    "DirectoryPolicy": "bundle",
    "FileEntry": "bundle",
    "ItemStatus": "bundle",
    "RESEARCH_REPORT_NAME": "bundle",
    "RUN_MANIFEST_NAME": "bundle",
    "STRUCTURE_POLICY": "bundle",
    "assemble_bundle": "bundle",
    "bundle_summary_lines": "bundle",
    "render_research_report": "bundle",
    "verify_bundle": "bundle",
}

__all__ = sorted(_LAZY_EXPORTS)

if TYPE_CHECKING:  # pragma: no cover - type checkers only, never at runtime
    from .bundle import (
        BATCH_ITEM_NAME, BUNDLE_FORMAT, BUNDLE_ITEMS, BUNDLE_MANIFEST_NAME,
        BundleEntry, BundleItem, BundleManifest, BundleResult,
        BundleVerification, CONFIDENCE_POLICY, DirectoryPolicy, FileEntry,
        ItemStatus, RESEARCH_REPORT_NAME, RUN_MANIFEST_NAME, STRUCTURE_POLICY,
        assemble_bundle, bundle_summary_lines, render_research_report,
        verify_bundle,
    )


def __getattr__(name: str) -> Any:
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(
            f"module 'eagent.deliverables' has no attribute '{name}'; exported "
            f"names are {', '.join(__all__)}")
    module = importlib.import_module(f".{module_name}", __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
