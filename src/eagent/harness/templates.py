"""Loading, indexing and interrogating the five template types.

The templates are where every threshold in this pipeline is allowed to come
from. Nothing else may invent one: a distance window hard-coded in a scoring
function would become this project's private definition of catalysis, and no
reviewer could trace it back to the systems it was fitted on. So the library
is strict about three things that a plain ``yaml.safe_load`` into a dict would
let through.

*Unsourced templates are refused at load time, with the file named.* The
pydantic models already reject a provenance block with no source type or no
identifiers, but they raise without saying which of eleven files was at fault,
and a run that dies with "template provenance is unsourced" sends a curator
hunting. :class:`TemplateLibrary` catches that and re-raises naming the path.

*Uncalibrated geometry is counted, not hidden.* A window whose
``calibrated_on`` is empty was written from general chemistry, not fitted to
complexes of known activity. Such a window can still be measured against, but
a run has to be able to say how much of its geometry rests on them --
:meth:`TemplateLibrary.calibration_report` is what lets a report state "nine of
nine constraints uncalibrated" instead of presenting a geometric verdict as
though it were grounded.

*Theoretical-model templates are flagged distinctly.* A theozyme is a
hypothesis about an arrangement of atoms; an experimental structure is an
observation of one. Both are admissible sources, and collapsing them into
"sourced" would let a sketched active site reject real enzymes. The library
keeps the distinction reachable through :meth:`window_authority` so the
evaluation layer can downgrade confidence instead of silently trusting it.

The attribute names on the library (``catalytic_templates`` keyed by template
id, ``engineering_templates`` keyed by family name) are not arbitrary: the
scientific interfaces duck-type ``ctx.templates`` through exactly those names.
See :func:`eagent.science.scorecard.resolve_catalytic_template` and
``ProposeMutations._engineering_template_for``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import yaml
from pydantic import BaseModel, ValidationError

from ..errors import TemplateError
from ..schemas.templates import (
    AssayTemplate,
    CatalyticTemplate,
    EngineeringTemplate,
    FamilyTemplate,
    GeometryConstraint,
    ReactionTemplate,
    TemplateSourceType,
)

__all__ = [
    "CalibrationReport",
    "ConstraintRecord",
    "LoadedTemplate",
    "TEMPLATE_KINDS",
    "TemplateLibrary",
    "WINDOW_AUTHORITIES",
    "default_template_dir",
    "normalise_family",
]

#: Sub-directory name -> model. The directory layout is the declaration of
#: kind: a catalytic template in ``family/`` would otherwise validate against
#: the wrong model and fail with a confusing field error.
TEMPLATE_KINDS: dict[str, type[BaseModel]] = {
    "reaction": ReactionTemplate,
    "family": FamilyTemplate,
    "catalytic": CatalyticTemplate,
    "engineering": EngineeringTemplate,
    "assay": AssayTemplate,
}

#: The three authority levels a geometry window can carry. These strings are
#: the values of :class:`eagent.tools.evaluate_catalysis.WindowAuthority`; they
#: are repeated here rather than imported so that loading templates does not
#: drag in the evaluation stack, and ``test_templates.py`` asserts the two
#: vocabularies have not drifted apart.
WINDOW_AUTHORITIES: tuple[str, str, str] = (
    "calibrated", "uncalibrated", "theoretical_model",
)


def default_template_dir() -> Path:
    """Where the shipped templates live, overridable with ``EAGENT_TEMPLATE_DIR``.

    Resolved from this file so a source checkout works uninstalled, and
    overridable so a run can be pinned to a curated copy rather than to
    whatever happens to be in the working tree.
    """
    env = os.environ.get("EAGENT_TEMPLATE_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[3] / "configs" / "templates"


def normalise_family(name: str | None) -> str:
    """Comparison key for a family name.

    ``MDR/ADH``, ``mdr_adh`` and ``MDR-ADH`` are the same family written three
    ways, and a dict keyed on the literal string would answer "no such family"
    for two of them -- which downstream means a candidate silently loses its
    mechanistic hypothesis rather than raising.
    """
    if not name:
        return ""
    return "".join(ch for ch in str(name).upper() if ch.isalnum())


@dataclass(frozen=True)
class LoadedTemplate:
    """One template with the file it came from.

    The path travels with the object because every integrity complaint this
    module can make -- unsourced, duplicate id, two families claiming one name
    -- is only actionable if it names the file a curator must open.
    """

    kind: str
    path: Path
    template: Any

    @property
    def template_id(self) -> str:
        return str(self.template.template_id)

    @property
    def is_theoretical(self) -> bool:
        return bool(self.template.provenance.is_theoretical)


@dataclass(frozen=True)
class ConstraintRecord:
    """One geometry constraint, with the template context needed to judge it."""

    template_id: str
    family_name: str
    constraint: GeometryConstraint
    source_type: TemplateSourceType
    path: Path

    @property
    def name(self) -> str:
        return self.constraint.name

    @property
    def severity(self) -> str:
        return self.constraint.severity

    @property
    def authority(self) -> str:
        """One of :data:`WINDOW_AUTHORITIES`.

        A theoretical template outranks calibration: coordinates computed from
        a model of a transition state are a hypothesis however many systems
        were used to tune the window around them.
        """
        if self.source_type is TemplateSourceType.THEORETICAL_MODEL:
            return "theoretical_model"
        return "calibrated" if self.constraint.is_calibrated else "uncalibrated"

    @property
    def may_reject(self) -> bool:
        """Whether a measurement against this window may disqualify a candidate."""
        return self.authority == "calibrated"


@dataclass(frozen=True)
class CalibrationReport:
    """How much of a library's geometry rests on uncalibrated windows.

    Reported as counts plus the offending names, never as a single "quality
    score": a run has to be able to print which windows a curator must fit,
    and a percentage cannot be acted on.
    """

    total: int
    calibrated: int
    uncalibrated: tuple[ConstraintRecord, ...]
    theoretical: tuple[ConstraintRecord, ...]
    gating_uncalibrated: tuple[ConstraintRecord, ...]

    @property
    def uncalibrated_count(self) -> int:
        return len(self.uncalibrated)

    @property
    def fully_uncalibrated(self) -> bool:
        """True when no window in the library was fitted to anything."""
        return self.total > 0 and self.calibrated == 0

    def summary(self) -> str:
        if self.total == 0:
            return "no geometry constraints are defined in this library"
        return (f"{self.uncalibrated_count} of {self.total} geometry "
                f"constraint(s) carry no calibration set; "
                f"{len(self.theoretical)} come from a theoretical-model "
                f"template")

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "calibrated": self.calibrated,
            "uncalibrated": [
                {"template_id": c.template_id, "constraint": c.name,
                 "severity": c.severity, "authority": c.authority,
                 "file": str(c.path)}
                for c in self.uncalibrated
            ],
            "theoretical": [
                {"template_id": c.template_id, "constraint": c.name}
                for c in self.theoretical
            ],
            "gating_uncalibrated": [
                {"template_id": c.template_id, "constraint": c.name}
                for c in self.gating_uncalibrated
            ],
            "summary": self.summary(),
        }


class TemplateLibrary:
    """The five template types, loaded, indexed and cross-checked.

    Indexed three ways because the run asks three different questions: by
    template id (a candidate records the template it was annotated against),
    by family name (a mechanistic hypothesis), and by reaction class (what the
    task is trying to do).
    """

    def __init__(self, loaded: Iterable[LoadedTemplate] = ()) -> None:
        self._loaded: list[LoadedTemplate] = []
        self.by_id: dict[str, Any] = {}
        self._paths: dict[str, Path] = {}
        self._kind_of: dict[str, str] = {}

        # Duck-typed surfaces the interfaces reach for through ``ctx.templates``.
        self.reaction_templates: dict[str, ReactionTemplate] = {}
        self.family_templates: dict[str, FamilyTemplate] = {}
        self.catalytic_templates: dict[str, CatalyticTemplate] = {}
        self.engineering_templates: dict[str, EngineeringTemplate] = {}
        self.assay_templates: dict[str, AssayTemplate] = {}

        self._family_by_key: dict[str, FamilyTemplate] = {}
        self._engineering_by_key: dict[str, EngineeringTemplate] = {}
        self._catalytic_by_family: dict[str, list[CatalyticTemplate]] = {}
        self._by_reaction_class: dict[str, list[ReactionTemplate]] = {}

        for item in loaded:
            self._add(item)

    # -- construction ------------------------------------------------------
    @classmethod
    def load(cls, root: str | Path | None = None) -> "TemplateLibrary":
        """Load every ``*.yaml`` under the five kind directories.

        A file that does not validate aborts the load. Skipping it would start
        a run with a silently smaller library, and "no catalytic template for
        this family" then reads as a fact about the enzyme rather than as a
        broken file.
        """
        base = Path(root) if root is not None else default_template_dir()
        if not base.is_dir():
            raise TemplateError(
                f"template directory {base} does not exist; the geometry "
                f"windows, catalytic roles and assay criteria all come from "
                f"there and none of them has a defensible default"
            )
        items: list[LoadedTemplate] = []
        for kind, model in TEMPLATE_KINDS.items():
            subdir = base / kind
            if not subdir.is_dir():
                continue
            for path in sorted(subdir.glob("*.yaml")):
                items.append(cls._load_one(kind, path, model))
        lib = cls(items)
        if not lib._loaded:
            raise TemplateError(
                f"no template files found under {base}; an empty library would "
                f"let every downstream step report 'no template' as though that "
                f"were a scientific finding"
            )
        return lib

    @staticmethod
    def _load_one(kind: str, path: Path, model: type[BaseModel]) -> LoadedTemplate:
        """Parse and validate one file, naming it in every failure mode."""
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise TemplateError(f"{path}: not valid YAML: {exc}") from exc
        if not isinstance(raw, Mapping):
            raise TemplateError(
                f"{path}: did not parse into a mapping; a template file holds "
                f"exactly one template"
            )
        try:
            template = model(**dict(raw))
        except TemplateError as exc:
            # The provenance and geometry validators raise this for an unsourced
            # template, a template with no identifiers, and a constraint with no
            # window. Re-raised with the path because the model cannot know it.
            raise TemplateError(f"{path}: {exc}") from exc
        except ValidationError as exc:
            raise TemplateError(
                f"{path}: does not validate as a {model.__name__}: {exc}"
            ) from exc
        return LoadedTemplate(kind=kind, path=path, template=template)

    def _add(self, item: LoadedTemplate) -> None:
        """Index one template, refusing every collision rather than overwriting."""
        tid = item.template_id
        if tid in self.by_id:
            raise TemplateError(
                f"{item.path}: template_id '{tid}' is already defined by "
                f"{self._paths[tid]}; a lookup would return one of them at "
                f"random and the manifest would not say which"
            )
        prov = item.template.provenance
        if not prov.source_type.admissible:   # defence in depth; the model refuses first
            raise TemplateError(f"{item.path}: template '{tid}' is unsourced")

        self._loaded.append(item)
        self.by_id[tid] = item.template
        self._paths[tid] = item.path
        self._kind_of[tid] = item.kind

        if item.kind == "reaction":
            tpl: Any = item.template
            self.reaction_templates[tid] = tpl
            self._by_reaction_class.setdefault(str(tpl.reaction_class), []).append(tpl)
        elif item.kind == "family":
            tpl = item.template
            key = normalise_family(tpl.family_name)
            previous = self._family_by_key.get(key)
            if previous is not None:
                raise TemplateError(
                    f"{item.path}: family '{tpl.family_name}' is already "
                    f"described by '{previous.template_id}'; two family "
                    f"templates for one family are two different mechanistic "
                    f"hypotheses and must not share a name"
                )
            self.family_templates[tpl.family_name] = tpl
            self._family_by_key[key] = tpl
        elif item.kind == "catalytic":
            tpl = item.template
            self.catalytic_templates[tid] = tpl
            self._catalytic_by_family.setdefault(
                normalise_family(tpl.family_name), []).append(tpl)
        elif item.kind == "engineering":
            tpl = item.template
            key = normalise_family(tpl.family_name)
            previous_eng = self._engineering_by_key.get(key)
            if previous_eng is not None:
                raise TemplateError(
                    f"{item.path}: engineering template for family "
                    f"'{tpl.family_name}' is already defined by "
                    f"'{previous_eng.template_id}'. The frozen roles and the "
                    f"round-1 mutation cap come from this file; silently "
                    f"keeping one of two would unfreeze a catalytic residue"
                )
            # Keyed by family, not by id: ProposeMutations looks the engineering
            # template up by the parent's family name.
            self.engineering_templates[tpl.family_name] = tpl
            self._engineering_by_key[key] = tpl
        elif item.kind == "assay":
            self.assay_templates[tid] = item.template

    # -- inventory ---------------------------------------------------------
    def __len__(self) -> int:
        return len(self._loaded)

    def __iter__(self) -> Iterator[LoadedTemplate]:
        return iter(self._loaded)

    def __contains__(self, template_id: object) -> bool:
        return template_id in self.by_id

    def ids(self) -> list[str]:
        return sorted(self.by_id)

    def family_names(self) -> list[str]:
        """Family names as the templates spell them (``MDR/ADH``, not ``MDRADH``)."""
        return sorted(self.family_templates)

    def reaction_classes(self) -> list[str]:
        return sorted(self._by_reaction_class)

    def path_of(self, template_id: str) -> Path | None:
        """The file a template came from, for an error message a curator can act on."""
        return self._paths.get(template_id)

    def kind_of(self, template_id: str) -> str | None:
        return self._kind_of.get(template_id)

    # -- lenient lookups (the shape the interfaces duck-type) --------------
    def get(self, template_id: str) -> Any | None:
        """Any template by id, or ``None``. Never raises: callers branch on None."""
        return self.by_id.get(template_id)

    def catalytic(self, template_id: str) -> CatalyticTemplate | None:
        return self.catalytic_templates.get(template_id)

    def reaction(self, template_id: str) -> ReactionTemplate | None:
        return self.reaction_templates.get(template_id)

    def assay(self, template_id: str) -> AssayTemplate | None:
        return self.assay_templates.get(template_id)

    def family(self, family_name: str) -> FamilyTemplate | None:
        return self._family_by_key.get(normalise_family(family_name))

    def engineering(self, family_name: str) -> EngineeringTemplate | None:
        return self._engineering_by_key.get(normalise_family(family_name))

    def reaction_class(self, reaction_class: str) -> list[ReactionTemplate]:
        """Every reaction template declaring this class, in load order."""
        return list(self._by_reaction_class.get(str(reaction_class), ()))

    def catalytic_for_family(self, family_name: str) -> list[CatalyticTemplate]:
        return list(self._catalytic_by_family.get(normalise_family(family_name), ()))

    # -- strict lookups ----------------------------------------------------
    def require(self, template_id: str) -> Any:
        if template_id not in self.by_id:
            raise TemplateError(
                f"no template '{template_id}' in this library; loaded ids: "
                f"{', '.join(self.ids())}"
            )
        return self.by_id[template_id]

    def require_family(self, family_name: str) -> FamilyTemplate:
        found = self.family(family_name)
        if found is None:
            raise TemplateError(
                f"no family template for '{family_name}'; known families: "
                f"{', '.join(self.family_names()) or 'none'}"
            )
        return found

    def resolve_catalytic(self, reaction_class: str,
                          family_name: str) -> CatalyticTemplate:
        """The catalytic template for a (reaction class, family) pair.

        The link runs reaction class -> "is this transformation one we have a
        template for" -> family -> ``catalytic_template_ids``. Ambiguity is an
        error rather than a choice: picking one of two catalytic templates by
        name similarity would attach a mechanism to a candidate that no curator
        ever asserted, and every geometry window and catalytic residue
        afterwards would be measured against the wrong one.
        """
        if not self.reaction_class(reaction_class):
            raise TemplateError(
                f"no reaction template declares reaction class "
                f"'{reaction_class}'; known classes: "
                f"{', '.join(self.reaction_classes()) or 'none'}. The reaction "
                f"a run is trying to do must be described before a mechanism "
                f"is attached to it"
            )
        fam = self.require_family(family_name)
        declared = list(fam.catalytic_template_ids)
        if not declared:
            raise TemplateError(
                f"family template '{fam.template_id}' ({fam.family_name}) "
                f"declares no catalytic_template_ids, so there is no "
                f"mechanism to evaluate candidates of this family against"
            )
        resolved: list[CatalyticTemplate] = []
        for tid in declared:
            found = self.catalytic_templates.get(tid)
            if found is None:
                raise TemplateError(
                    f"family '{fam.family_name}' names catalytic template "
                    f"'{tid}', which is not loaded (see "
                    f"{self.path_of(fam.template_id)})"
                )
            resolved.append(found)
        if len(resolved) > 1:
            raise TemplateError(
                f"family '{fam.family_name}' declares {len(resolved)} catalytic "
                f"templates ({', '.join(t.template_id for t in resolved)}) and "
                f"nothing in the files says which one serves reaction class "
                f"'{reaction_class}'. A curator must declare the mapping; this "
                f"library will not guess it from the identifiers"
            )
        return resolved[0]

    # -- calibration and authority ----------------------------------------
    def constraints(self) -> list[ConstraintRecord]:
        """Every geometry constraint in the library, with its template context."""
        out: list[ConstraintRecord] = []
        for item in self._loaded:
            if item.kind != "catalytic":
                continue
            tpl: CatalyticTemplate = item.template
            for constraint in tpl.geometry_constraints:
                out.append(ConstraintRecord(
                    template_id=tpl.template_id,
                    family_name=tpl.family_name,
                    constraint=constraint,
                    source_type=tpl.provenance.source_type,
                    path=item.path,
                ))
        return out

    def uncalibrated_constraints(self) -> list[ConstraintRecord]:
        """Constraints with no ``calibrated_on`` set.

        These are usable for measurement and for lowering confidence. They may
        not reject a candidate, which is what :attr:`ConstraintRecord.may_reject`
        and the verifier's uncalibrated-disqualification check enforce.
        """
        return [c for c in self.constraints() if not c.constraint.is_calibrated]

    def calibration_report(self) -> CalibrationReport:
        """How much of this library's geometry is uncalibrated, in countable form."""
        all_c = self.constraints()
        uncal = [c for c in all_c if not c.constraint.is_calibrated]
        theoretical = [c for c in all_c
                       if c.source_type is TemplateSourceType.THEORETICAL_MODEL]
        gating_uncal = [c for c in uncal if c.severity == "gating"]
        return CalibrationReport(
            total=len(all_c),
            calibrated=len(all_c) - len(uncal),
            uncalibrated=tuple(uncal),
            theoretical=tuple(theoretical),
            gating_uncalibrated=tuple(gating_uncal),
        )

    def theoretical_template_ids(self) -> list[str]:
        """Templates whose source is an explicitly labelled theoretical model."""
        return sorted(i.template_id for i in self._loaded if i.is_theoretical)

    def is_theoretical(self, template_id: str) -> bool:
        """Whether this template's geometry came from a model rather than an observation.

        Downstream this must *downgrade* confidence rather than exclude the
        template: a theozyme is a legitimate hypothesis to measure against, and
        an illegitimate reason to reject an enzyme.
        """
        tpl = self.by_id.get(template_id)
        if tpl is None:
            return False
        return bool(tpl.provenance.is_theoretical)

    def window_authority(self, template_id: str, constraint_name: str) -> str:
        """Authority of one window: one of :data:`WINDOW_AUTHORITIES`.

        Mirrors :class:`eagent.tools.evaluate_catalysis.WindowAuthority` so the
        evaluation layer and a report generated straight from the library agree
        about which windows are allowed to reject a candidate.
        """
        for record in self.constraints():
            if record.template_id == template_id and record.name == constraint_name:
                return record.authority
        raise TemplateError(
            f"catalytic template '{template_id}' has no geometry constraint "
            f"named '{constraint_name}'"
        )

    def confidence_caveats(self, template_id: str) -> list[str]:
        """Lines a report must carry when it quotes a verdict from this template.

        Returned as text rather than as a numeric penalty on purpose: there is
        no measured conversion from "the window was never fitted" to a number
        of confidence points, and inventing one would dress a known gap up as a
        quantity.
        """
        tpl = self.by_id.get(template_id)
        if tpl is None:
            raise TemplateError(f"no template '{template_id}' in this library")
        notes: list[str] = []
        if tpl.provenance.is_theoretical:
            notes.append(
                f"{template_id} is a labelled theoretical model; its windows "
                f"may not drive a rejection and passes against them are "
                f"provisional")
        uncal = [c.name for c in self.uncalibrated_constraints()
                 if c.template_id == template_id]
        if uncal:
            notes.append(
                f"{template_id}: {len(uncal)} uncalibrated window(s) "
                f"({', '.join(uncal)}); a pass against them is not evidence of "
                f"catalytic competence and a failure may not disqualify")
        if isinstance(tpl, CatalyticTemplate) and not tpl.reference_structures:
            notes.append(
                f"{template_id} names no reference structure, which is why its "
                f"windows could not be calibrated")
        return notes

    # -- integrity ---------------------------------------------------------
    def integrity_problems(self) -> list[str]:
        """Gaps a run should report up front, as sentences a curator can act on.

        Not raised: every one of these is a legitimate state for a library that
        is honest about being incomplete. They are returned so the controller
        can put them in the manifest instead of a run discovering them one
        candidate at a time.
        """
        problems: list[str] = []
        for fam in self.family_templates.values():
            for tid in fam.catalytic_template_ids:
                if tid not in self.catalytic_templates:
                    problems.append(
                        f"family '{fam.family_name}' names catalytic template "
                        f"'{tid}', which is not loaded")
            if not fam.catalytic_template_ids:
                problems.append(
                    f"family '{fam.family_name}' declares no catalytic template")
        for key, fam_tpl in self._family_by_key.items():
            if key not in self._engineering_by_key:
                problems.append(
                    f"family '{fam_tpl.family_name}' has no engineering "
                    f"template; variants for it cannot be proposed")
        for cat in self.catalytic_templates.values():
            if normalise_family(cat.family_name) not in self._family_by_key:
                problems.append(
                    f"catalytic template '{cat.template_id}' claims family "
                    f"'{cat.family_name}', for which no family template is "
                    f"loaded")
        report = self.calibration_report()
        for record in report.gating_uncalibrated:
            problems.append(
                f"{record.template_id}: constraint '{record.name}' is gating "
                f"but uncalibrated; an unfitted window may not reject a "
                f"candidate")
        return problems
