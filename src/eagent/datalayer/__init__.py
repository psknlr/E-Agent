"""Six-layer data architecture and the connector capability registry.

The package answers two questions that a single "database config" cannot:

* *Which question does this record answer, and what can it not substitute for?*
  -- :mod:`eagent.datalayer.layers`, which also fixes the closed set of
  identifier-based joins and refuses every resemblance-based one.
* *What is this resource documented to do, what must it never be used to claim,
  and what has nobody verified yet?* -- :mod:`eagent.datalayer.registry`, loaded
  from ``configs/datasources/*.yaml``.

Nothing in the registry has been connectivity-tested in this environment. Every
entry carries ``connectivity_verified=false``, uncertain specifics are ``null``
with ``needs_curation: true``, and the capability flags describe documentation
rather than proven behaviour.
"""

from __future__ import annotations

from .layers import (
    DataLayer,
    JoinError,
    JoinFieldMissingError,
    JoinKey,
    JoinKeySpec,
    JoinResult,
    LAYER_ORDER,
    LayerCoverage,
    SIMILARITY_PSEUDO_KEY_STEMS,
    SimilarityJoinRefusedError,
    UnknownJoinKeyError,
    is_similarity_pseudo_key,
    join_key_spec,
    permitted_keys,
    refuse_similarity_join,
    validate_join,
)
from .registry import (
    AccessMode,
    CAPABILITY_NAMES,
    CapabilityFlags,
    CapabilityState,
    DataSource,
    DuplicateSourceError,
    EVIDENCE_STRENGTH_IS_CANONICAL,
    EvidenceStrength,
    IndependenceReport,
    RegistryError,
    RegistryIntegrityError,
    SourceRegistry,
    UnknownSourceError,
    default_datasource_dir,
)

__all__ = [
    "AccessMode",
    "CAPABILITY_NAMES",
    "CapabilityFlags",
    "CapabilityState",
    "DataLayer",
    "DataSource",
    "DuplicateSourceError",
    "EVIDENCE_STRENGTH_IS_CANONICAL",
    "EvidenceStrength",
    "IndependenceReport",
    "JoinError",
    "JoinFieldMissingError",
    "JoinKey",
    "JoinKeySpec",
    "JoinResult",
    "LAYER_ORDER",
    "LayerCoverage",
    "RegistryError",
    "RegistryIntegrityError",
    "SIMILARITY_PSEUDO_KEY_STEMS",
    "SimilarityJoinRefusedError",
    "SourceRegistry",
    "UnknownJoinKeyError",
    "UnknownSourceError",
    "default_datasource_dir",
    "is_similarity_pseudo_key",
    "join_key_spec",
    "permitted_keys",
    "refuse_similarity_join",
    "validate_join",
]
