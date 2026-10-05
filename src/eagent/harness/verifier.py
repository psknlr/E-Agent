"""The independent verifier.

Independent means one specific thing: this module does not read the producing
step's verdict. It re-derives what it checks from the primary material -- the
candidate's own sequence string, the coordinate file on disk, the template's
own windows, the pose's recorded restraint list -- and compares that with what
the report says. A verifier that reads
:attr:`~eagent.schemas.candidate.GeometryReport.gating_passed` and agrees with
it has verified nothing; it has restated the claim in a second voice, which is
worse than no check because it reads like corroboration.

The fields it deliberately ignores are listed in
:data:`IGNORED_PRODUCER_CONCLUSIONS` and the list is asserted in the tests, so
a later edit that "simplifies" a check by trusting one of them fails.

What it checks, and the failure each check exists for:

sequence / structure pairing
    A report that pairs candidate C with structure S when S is a different
    protein. Everything downstream -- pocket residues, mutation numbering,
    geometry -- is then measured on the wrong molecule and looks perfectly
    self-consistent.

ligand identity and chirality
    The docked molecule is not the one in the spec, or carries a different
    configuration. The run then optimises an enzyme for a compound the project
    is not about.

cofactor type and oxidation state
    NADP+ where the template requires NADPH. An oxidised cofactor cannot
    donate a hydride, so a "competent" complex built around one is a picture
    of a reaction that cannot occur.

residue numbering round-trip
    An off-by-a-His-tag mapping. The variant expresses, folds, shows nothing,
    and the conclusion drawn is "the hypothesis was wrong" rather than "we
    built the wrong protein".

claims without artifacts or provenance
    A sentence in a report that nothing on disk supports. It cannot be
    rechecked, and it will outlive the run.

numeric ee without a calibration source
    An ee is a measurement. A predicted one is a number from a model that must
    name what it was calibrated on, or it is a guess formatted as data.

disqualification on an uncalibrated window alone
    Throwing away candidates against a window nobody fitted. The pool shrinks,
    the shrinkage looks principled, and the criterion was never evidence.

restrained constraints counted as independent evidence
    A distance that was enforced during modelling cannot afterwards
    corroborate the model that enforced it.

an experimental-negative label on a computational failure
    A crashed docking run written into the record layer as "no activity
    detected" is a fabricated experimental result.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..context import RunContext
from ..envelope import Artifact, Provenance, QCFlag, Severity, Status, ToolResult
from ..provenance import utc_now
from ..schemas.candidate import Candidate, ComplexPose, GeometryReport, StructureRecord
from ..schemas.chem import (
    CofactorState, Stereochemistry, cofactor_state_from_ligand_code,
)
from ..schemas.record import ExperimentRecord, OutcomeClass
from ..schemas.templates import CatalyticTemplate

__all__ = [
    "Claim",
    "Finding",
    "IGNORED_PRODUCER_CONCLUSIONS",
    "IndependentVerifier",
    "LigandDeclaration",
    "VERIFIER_NAME",
    "VerificationReport",
]

VERIFIER_NAME = "independent_verifier"

#: Producer-authored verdicts this module must never read. Each one is a
#: conclusion of the step under review: using it would turn verification into
#: agreement. The recomputation that replaces each is named alongside.
IGNORED_PRODUCER_CONCLUSIONS: dict[str, str] = {
    "GeometryReport.gating_passed":
        "recomputed from the template's own windows and the measurements",
    "GeometryReport.independent_satisfied":
        "recomputed with science.robustness.CircularityGuard from the pose's "
        "restrained_constraints",
    "GeometryReport.circular_constraints":
        "recomputed from the pose's restrained_constraints",
    "StructureRecord.sequence_identity_to_candidate":
        "recomputed by aligning the candidate sequence to the coordinates",
    "StructureRecord.is_mutant_relative_to_candidate":
        "recomputed from the alignment; used only as the report's declaration "
        "to be checked against",
    "Candidate.passes_gates":
        "not consulted; the verifier checks inputs and claims, not rankings",
    "CatalyticMapping.is_complete":
        "recomputed by re-deriving each role's residue from the sequence",
}

#: An ee written into prose. ``98% ee``, ``ee of 98%``, ``ee = -12``.
_EE_PATTERN = re.compile(
    r"(?:(-?\d+(?:\.\d+)?)\s*%?\s*ee\b)|(?:\bee\b\s*(?:of|=|:)?\s*(-?\d+(?:\.\d+)?)\s*%?)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Finding:
    """One problem, at one severity, about one subject."""

    code: str
    severity: Severity
    message: str
    subject: str | None = None
    recomputed_from: str = ""

    def to_flag(self) -> QCFlag:
        suffix = f" [recomputed from {self.recomputed_from}]" \
            if self.recomputed_from else ""
        return QCFlag(self.code, self.severity, self.message + suffix,
                      self.subject)

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "severity": self.severity.value,
                "message": self.message, "subject": self.subject,
                "recomputed_from": self.recomputed_from}


@dataclass(frozen=True)
class Claim:
    """One statement a report makes, with what is supposed to back it."""

    claim_id: str
    text: str = ""
    subject: str | None = None            # candidate id, pose id, family
    artifact_key: str | None = None
    step_id: str | None = None
    interface: str | None = None
    metric: str | None = None             # ee_target_pct, conversion_pct, ...
    value: float | None = None
    unit: str | None = None
    calibration_source: str | None = None
    evidence_refs: tuple[str, ...] = ()
    is_key: bool = True

    @property
    def is_ee(self) -> bool:
        """Whether this claim carries an enantiomeric excess."""
        metric = (self.metric or "").lower()
        return "ee" in re.split(r"[^a-z]+", metric) or metric.startswith("ee_")


@dataclass(frozen=True)
class LigandDeclaration:
    """What a pose says it actually contains.

    Separate from :class:`~eagent.schemas.candidate.ComplexPose` because the
    pose records presence and provenance of a ligand, not its chemical
    identity. The identity has to come from whatever built the complex, and it
    is exactly the thing that silently drifts from the spec.
    """

    pose_id: str
    substrate_smiles: str | None = None
    substrate_inchikey: str | None = None
    product_configuration: Stereochemistry | None = None
    cofactor_name: str | None = None
    cofactor_ligand_code: str | None = None
    cofactor_state: CofactorState | None = None
    metals: tuple[str, ...] = ()


@dataclass
class VerificationReport:
    """Everything the verifier concluded, and everything it could not check."""

    findings: list[Finding] = field(default_factory=list)
    unverifiable: list[str] = field(default_factory=list)
    checks_run: list[str] = field(default_factory=list)
    at: str = field(default_factory=utc_now)

    @property
    def blockers(self) -> list[Finding]:
        return [f for f in self.findings if f.severity is Severity.BLOCKER]

    @property
    def clean(self) -> bool:
        return not self.blockers

    def add(self, code: str, severity: Severity, message: str,
            subject: str | None = None, recomputed_from: str = "") -> None:
        self.findings.append(Finding(code, severity, message, subject,
                                     recomputed_from))

    def cannot_check(self, what: str) -> None:
        """Record something that could not be verified.

        Not silence: an unverifiable claim is a different state from a verified
        one, and a report that omits the distinction reads as a clean bill of
        health.
        """
        self.unverifiable.append(what)

    def to_dict(self) -> dict[str, Any]:
        return {
            "at": self.at,
            "checks_run": list(self.checks_run),
            "findings": [f.to_dict() for f in self.findings],
            "unverifiable": list(self.unverifiable),
            "n_blockers": len(self.blockers),
            "ignored_producer_conclusions": dict(IGNORED_PRODUCER_CONCLUSIONS),
        }


class IndependentVerifier:
    """Re-derives, then compares. Never agrees with a step by reading its verdict."""

    name = VERIFIER_NAME
    version = "0.1.0"

    # -- entry point -------------------------------------------------------
    def verify(
        self,
        ctx: RunContext,
        *,
        candidates: Sequence[Candidate] = (),
        claims: Sequence[Claim] = (),
        records: Sequence[ExperimentRecord] = (),
        catalytic_templates: Mapping[str, CatalyticTemplate] | None = None,
        ligands: Mapping[str, LigandDeclaration] | None = None,
        structure_chains: Mapping[str, str] | None = None,
        disqualifying_constraints: Mapping[str, Sequence[str]] | None = None,
        computational_failures: Iterable[str] = (),
        artifacts: Sequence[Artifact] = (),
        write_report: bool = True,
    ) -> ToolResult:
        """Run every check and return an envelope whose blockers stop the run."""
        report = VerificationReport()
        templates = self._template_index(ctx, catalytic_templates)
        ligand_map = dict(ligands or {})
        chains = dict(structure_chains or {})
        failures = {str(c) for c in computational_failures}

        self._check_sequence_structure_pairing(report, candidates, chains)
        self._check_ligand_identity_and_chirality(
            ctx, report, candidates, ligand_map)
        self._check_cofactor_against_template(
            report, candidates, templates, ligand_map)
        self._check_numbering_round_trip(report, candidates)
        self._check_claims_backed(ctx, report, claims, artifacts)
        self._check_numeric_ee(report, candidates, claims, records)
        self._check_uncalibrated_disqualification(
            report, candidates, templates, disqualifying_constraints or {})
        self._check_restrained_excluded(report, candidates)
        self._check_no_experimental_negative_for_computational_failure(
            report, records, failures)

        return self._envelope(ctx, report, write_report)

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _template_index(
        ctx: RunContext,
        supplied: Mapping[str, CatalyticTemplate] | None,
    ) -> dict[str, CatalyticTemplate]:
        """Catalytic templates by id and by ``family:<name>``.

        Taken from the explicit argument first and from ``ctx.templates``
        second. A template that is simply absent makes a check unverifiable,
        never passed: "no template to compare against" is not agreement.
        """
        index: dict[str, CatalyticTemplate] = {}

        def put(tpl: Any) -> None:
            if isinstance(tpl, CatalyticTemplate):
                index[tpl.template_id] = tpl
                index.setdefault(f"family:{tpl.family_name}", tpl)

        if isinstance(supplied, Mapping):
            for value in supplied.values():
                put(value)
        elif supplied is not None:
            for value in supplied:          # type: ignore[union-attr]
                put(value)
        library = getattr(ctx, "templates", None)
        store = getattr(library, "catalytic_templates", None)
        if isinstance(store, Mapping):
            for value in store.values():
                put(value)
        return index

    @staticmethod
    def _template_for(cand: Candidate,
                      templates: Mapping[str, CatalyticTemplate],
                      ) -> CatalyticTemplate | None:
        tid = cand.catalytic_mapping.catalytic_template_id
        if tid and tid in templates:
            return templates[tid]
        family = cand.family.family_name
        if family:
            return templates.get(f"family:{family}")
        return None

    # -- 1. sequence / structure pairing -----------------------------------
    def _check_sequence_structure_pairing(
        self, report: VerificationReport, candidates: Sequence[Candidate],
        chains: Mapping[str, str],
    ) -> None:
        """Re-align every candidate to its own coordinates.

        The structure is read from disk and aligned here rather than trusting
        ``sequence_identity_to_candidate``: that number is the pairing step's
        own conclusion, and a step that picked the wrong file reports a
        confident identity for the wrong protein.
        """
        report.checks_run.append("sequence_structure_pairing")
        from ..science import numbering as numbering_mod
        from ..science import structure_io

        for cand in candidates:
            for record in cand.structures:
                subject = f"{cand.candidate_id}/{record.structure_id}"
                chain = self._chain_for(report, record, chains, subject,
                                        structure_io)
                if chain is None:
                    continue
                try:
                    rmap = numbering_mod.build_map(cand.sequence, chain)
                except Exception as exc:
                    report.add(
                        "structure_alignment_failed", Severity.WARN,
                        f"{subject}: the candidate sequence could not be "
                        f"aligned to the coordinates ({exc}); the pairing is "
                        f"unverified rather than confirmed", subject)
                    report.cannot_check(f"{subject}: alignment failed")
                    continue
                self._compare_alignment(report, cand, record, rmap, subject)

    @staticmethod
    def _chain_for(report: VerificationReport, record: StructureRecord,
                   chains: Mapping[str, str], subject: str, structure_io: Any):
        """The chain to align against, or ``None`` with the reason recorded."""
        if not record.path:
            report.cannot_check(
                f"{subject}: the structure record names no file, so the "
                f"pairing cannot be rechecked")
            return None
        path = Path(record.path)
        if not path.exists():
            report.add("structure_file_missing", Severity.WARN,
                       f"{subject}: {path} is not on disk; the sequence to "
                       f"structure pairing could not be rechecked", subject)
            report.cannot_check(f"{subject}: {path} missing")
            return None
        try:
            structure = structure_io.read_structure(path, record.structure_id)
        except Exception as exc:
            report.add("structure_unreadable", Severity.WARN,
                       f"{subject}: {path} could not be parsed ({exc})", subject)
            report.cannot_check(f"{subject}: unparseable coordinates")
            return None
        wanted = chains.get(record.structure_id)
        if wanted:
            chain = structure.chain(wanted)
            if chain is None:
                report.add(
                    "structure_chain_absent", Severity.BLOCKER,
                    f"{subject}: chain {wanted!r} was named for this record "
                    f"and the file has chains "
                    f"{[c.chain_id for c in structure.chains]}; the report is "
                    f"paired with coordinates it does not contain", subject,
                    "the coordinate file")
                return None
            return chain
        polymer = [c for c in structure.chains if c.polymer_residues()]
        if len(polymer) == 1:
            return polymer[0]
        report.add(
            "structure_chain_ambiguous", Severity.WARN,
            f"{subject}: {len(polymer)} polymer chains and no chain recorded "
            f"for this structure, so the verifier will not pick one; supply "
            f"structure_chains to make this checkable", subject)
        report.cannot_check(f"{subject}: chain not specified")
        return None

    @staticmethod
    def _compare_alignment(report: VerificationReport, cand: Candidate,
                           record: StructureRecord, rmap: Any,
                           subject: str) -> None:
        """Mismatches the report did not declare are a wrong-protein pairing."""
        declared = {m.strip().upper() for m in record.mutations_in_structure}
        undeclared: list[str] = []
        for index, cand_letter, struct_letter, author in rmap.mismatches:
            token = f"{cand_letter}{index + 1}{struct_letter}"
            alt = f"{cand_letter}{author.token}{struct_letter}"
            if token.upper() in declared or alt.upper() in declared:
                continue
            undeclared.append(f"{token} (author {author})")
        if undeclared:
            report.add(
                "sequence_structure_mismatch", Severity.BLOCKER,
                f"{subject}: the coordinates disagree with the candidate "
                f"sequence at {len(undeclared)} position(s) that the record "
                f"does not declare ({', '.join(undeclared[:6])}). Either the "
                f"structure belongs to a different protein or it is a mutant "
                f"nobody recorded; every pocket residue and every mutation "
                f"numbered on it is suspect", subject,
                "re-alignment of the candidate sequence to the coordinates")
        if rmap.unmapped_structure_positions and not undeclared:
            report.add(
                "structure_has_extra_residues", Severity.WARN,
                f"{subject}: {len(rmap.unmapped_structure_positions)} observed "
                f"residue(s) are absent from the candidate sequence (a tag, a "
                f"fusion, or the wrong chain)", subject,
                "re-alignment of the candidate sequence to the coordinates")

    # -- 2. ligand identity and chirality ----------------------------------
    def _check_ligand_identity_and_chirality(
        self, ctx: RunContext, report: VerificationReport,
        candidates: Sequence[Candidate],
        ligands: Mapping[str, LigandDeclaration],
    ) -> None:
        """The modelled molecule must be the one the spec names, stereo included."""
        report.checks_run.append("ligand_identity_and_chirality")
        spec = ctx.task.reaction
        want_smiles = (spec.substrate.isomeric_smiles or "").strip()
        want_key = (spec.substrate.inchikey or "").strip().upper()
        target_config = spec.product.target_stereochemistry

        for cand in candidates:
            for pose in cand.poses:
                decl = ligands.get(pose.pose_id)
                subject = f"{cand.candidate_id}/{pose.pose_id}"
                if decl is None:
                    if pose.substrate_present:
                        report.cannot_check(
                            f"{subject}: no ligand declaration, so the "
                            f"modelled substrate's identity was not checked "
                            f"against the spec")
                    continue
                self._compare_substrate(report, subject, decl, want_smiles,
                                        want_key)
                self._compare_configuration(report, subject, decl,
                                            target_config)

    @staticmethod
    def _compare_substrate(report: VerificationReport, subject: str,
                           decl: LigandDeclaration, want_smiles: str,
                           want_key: str) -> None:
        got_smiles = (decl.substrate_smiles or "").strip()
        got_key = (decl.substrate_inchikey or "").strip().upper()

        if want_key and got_key:
            if got_key != want_key:
                same_skeleton = got_key[:14] == want_key[:14]
                code = ("ligand_chirality_mismatch" if same_skeleton
                        else "ligand_identity_mismatch")
                what = ("the same constitution with a different "
                        "stereochemistry" if same_skeleton
                        else "a different compound")
                report.add(
                    code, Severity.BLOCKER,
                    f"{subject}: the modelled substrate has InChIKey "
                    f"{got_key} and the spec asks for {want_key} -- {what}. "
                    f"The run would be optimising an enzyme for a molecule "
                    f"the project is not about", subject,
                    "the spec's substrate block")
                return
        elif want_smiles and got_smiles:
            if got_smiles != want_smiles:
                stereo_only = (got_smiles.replace("@", "").replace("/", "")
                               .replace("\\", "")
                               == want_smiles.replace("@", "").replace("/", "")
                               .replace("\\", ""))
                code = ("ligand_chirality_mismatch" if stereo_only
                        else "ligand_identity_mismatch")
                report.add(
                    code, Severity.BLOCKER,
                    f"{subject}: the modelled substrate SMILES "
                    f"{got_smiles!r} is not the spec's {want_smiles!r}"
                    + (" -- the difference is stereochemical" if stereo_only
                       else ""), subject, "the spec's substrate block")
                return
        else:
            report.cannot_check(
                f"{subject}: the spec or the pose gives no structure for the "
                f"substrate, so identity could not be compared")
            return
        if want_smiles and got_smiles and got_smiles != want_smiles:
            report.add(
                "ligand_smiles_differs_from_inchikey_match", Severity.WARN,
                f"{subject}: the InChIKeys agree but the SMILES strings "
                f"differ; without a chemistry toolkit this verifier compares "
                f"strings and cannot say whether they are the same molecule "
                f"written two ways", subject)

    @staticmethod
    def _compare_configuration(report: VerificationReport, subject: str,
                               decl: LigandDeclaration,
                               target: Stereochemistry) -> None:
        got = decl.product_configuration
        if got is None or target in (Stereochemistry.UNSPECIFIED,
                                     Stereochemistry.ACHIRAL):
            return
        if target in (Stereochemistry.R, Stereochemistry.S) and got != target:
            report.add(
                "product_configuration_mismatch", Severity.BLOCKER,
                f"{subject}: the modelled product is {got.value} and the task "
                f"asks for {target.value}. A variant that is excellent for "
                f"one enantiomer is the worst possible choice for the other",
                subject, "the task's product block")

    # -- 3. cofactor against the template ----------------------------------
    def _check_cofactor_against_template(
        self, report: VerificationReport, candidates: Sequence[Candidate],
        templates: Mapping[str, CatalyticTemplate],
        ligands: Mapping[str, LigandDeclaration],
    ) -> None:
        """NAD(P)+ and NAD(P)H are not interchangeable, and nor are NADH and NADPH."""
        report.checks_run.append("cofactor_type_and_oxidation_state")
        for cand in candidates:
            template = self._template_for(cand, templates)
            if template is None:
                if cand.poses:
                    report.cannot_check(
                        f"{cand.candidate_id}: no catalytic template was "
                        f"available, so the cofactor could not be checked "
                        f"against one")
                continue
            for pose in cand.poses:
                self._check_one_pose_cofactor(
                    report, cand, pose, template, ligands.get(pose.pose_id))

    @staticmethod
    def _check_one_pose_cofactor(
        report: VerificationReport, cand: Candidate, pose: ComplexPose,
        template: CatalyticTemplate, decl: LigandDeclaration | None,
    ) -> None:
        subject = f"{cand.candidate_id}/{pose.pose_id}"
        required_state = template.required_cofactor_state
        state = (decl.cofactor_state if decl and decl.cofactor_state is not None
                 else pose.cofactor_state)
        name = decl.cofactor_name if decl else None
        code = decl.cofactor_ligand_code if decl else None

        if template.required_cofactor and not pose.cofactor_present:
            report.add(
                "required_cofactor_absent", Severity.BLOCKER,
                f"{subject}: the template requires {template.required_cofactor} "
                f"and the pose contains no cofactor. A hydride-transfer "
                f"complex without the hydride donor is not the mechanism this "
                f"template describes", subject, template.template_id)
            return
        if not pose.cofactor_present:
            return

        if name and template.required_cofactor and \
                name.strip().upper() != template.required_cofactor.strip().upper():
            report.add(
                "cofactor_identity_mismatch", Severity.BLOCKER,
                f"{subject}: the pose carries {name} and template "
                f"{template.template_id} requires "
                f"{template.required_cofactor}; the 2'-phosphate changes which "
                f"enzymes can use it, so this is not a substitution the model "
                f"may make", subject, template.template_id)

        if required_state is not CofactorState.UNKNOWN and \
                state is not CofactorState.UNKNOWN and state != required_state:
            report.add(
                "cofactor_oxidation_state_mismatch", Severity.BLOCKER,
                f"{subject}: the cofactor is {state.value} and template "
                f"{template.template_id} requires {required_state.value}. An "
                f"oxidised nicotinamide cannot donate a hydride, so a complex "
                f"built around one models a reaction that cannot occur",
                subject, template.template_id)
        elif state is CofactorState.UNKNOWN and \
                required_state is not CofactorState.UNKNOWN:
            report.add(
                "cofactor_state_unknown", Severity.WARN,
                f"{subject}: the cofactor's oxidation state is not recorded "
                f"and template {template.template_id} requires "
                f"{required_state.value}; the complex cannot be called "
                f"competent until a curator supplies the state", subject,
                template.template_id)
            report.cannot_check(f"{subject}: cofactor oxidation state unrecorded")

        if code:
            implied = cofactor_state_from_ligand_code(code)
            if implied is not CofactorState.UNKNOWN and \
                    state is not CofactorState.UNKNOWN and implied != state:
                report.add(
                    "cofactor_state_contradicts_ligand_code", Severity.BLOCKER,
                    f"{subject}: ligand code {code} is {implied.value} and the "
                    f"pose declares {state.value}; one of the two is wrong and "
                    f"the geometry was measured on whichever the file actually "
                    f"contains", subject, "the PDB chemical component code")
            if implied is not CofactorState.UNKNOWN and \
                    required_state is not CofactorState.UNKNOWN and \
                    implied != required_state:
                report.add(
                    "cofactor_oxidation_state_mismatch", Severity.BLOCKER,
                    f"{subject}: ligand code {code} is {implied.value} and "
                    f"template {template.template_id} requires "
                    f"{required_state.value}", subject, template.template_id)

        missing_metals = [m for m in template.metals
                          if m.strip().upper() not in
                          {x.strip().upper() for x in pose.metals_present}]
        if missing_metals:
            report.add(
                "required_metal_absent", Severity.BLOCKER,
                f"{subject}: template {template.template_id} requires "
                f"{', '.join(template.metals)} and the pose contains "
                f"{pose.metals_present or 'no metal'}; the catalytic metal is "
                f"part of the mechanism, not a decoration", subject,
                template.template_id)

    # -- 4. numbering round-trip -------------------------------------------
    def _check_numbering_round_trip(self, report: VerificationReport,
                                    candidates: Sequence[Candidate]) -> None:
        """Re-derive each catalytic role's residue from the sequence itself.

        ``role_to_residue`` is 1-based candidate numbering and ``role_to_index``
        is the 0-based index of the same residue (see
        ``eagent.tools.annotate_family``). Re-deriving the token from the
        sequence and comparing is the whole check: if they disagree, some
        downstream step is naming one residue and measuring another.
        """
        report.checks_run.append("residue_numbering_round_trip")
        for cand in candidates:
            mapping = cand.catalytic_mapping
            sequence = cand.sequence
            for role, token in sorted(mapping.role_to_residue.items()):
                subject = f"{cand.candidate_id}/{role}"
                parsed = _parse_residue_token(token)
                if parsed is None:
                    report.add(
                        "residue_token_unparseable", Severity.BLOCKER,
                        f"{subject}: residue token {token!r} is not "
                        f"<letter><1-based position>; nothing downstream can "
                        f"map it back to a residue", subject)
                    continue
                letter, author_number = parsed
                index = mapping.role_to_index.get(role)
                if index is None:
                    report.add(
                        "role_without_index", Severity.BLOCKER,
                        f"{subject}: the role names residue {token} but "
                        f"carries no sequence index, so no step can locate it "
                        f"in the candidate", subject)
                    continue
                if not 0 <= index < len(sequence):
                    report.add(
                        "residue_index_out_of_range", Severity.BLOCKER,
                        f"{subject}: index {index} is outside a sequence of "
                        f"{len(sequence)} residues", subject)
                    continue
                round_trip = f"{sequence[index]}{index + 1}"
                if round_trip.upper() != f"{letter}{author_number}".upper():
                    report.add(
                        "numbering_round_trip_failed", Severity.BLOCKER,
                        f"{subject}: the report names {token} but index "
                        f"{index} of this candidate is {sequence[index]}"
                        f"{index + 1}. A proposal written against this "
                        f"numbering would mutate a different residue, the "
                        f"variant would express and fold, and the flat assay "
                        f"would be read as a refuted hypothesis", subject,
                        "the candidate sequence")

    # -- 5. claims need artifacts and provenance ---------------------------
    def _check_claims_backed(self, ctx: RunContext,
                             report: VerificationReport,
                             claims: Sequence[Claim],
                             artifacts: Sequence[Artifact]) -> None:
        """Every key claim must point at something on disk that was recorded."""
        report.checks_run.append("claims_have_artifacts_and_provenance")
        known_artifacts: set[str] = {a.key for a in artifacts}
        provenance_steps: dict[str, dict[str, Any]] = {}
        provenance_tools: set[str] = set()
        for rec in ctx.manifest.steps:
            provenance_steps[rec.step_id] = rec.to_dict()
            provenance_tools.add(rec.interface)
            for art in rec.artifacts:
                key = art.get("key")
                if key:
                    known_artifacts.add(str(key))

        for claim in claims:
            if not claim.is_key:
                continue
            if not claim.artifact_key:
                report.add(
                    "claim_without_artifact", Severity.BLOCKER,
                    f"claim {claim.claim_id} names no artifact: "
                    f"{claim.text[:120]!r}. A statement nothing on disk "
                    f"supports cannot be rechecked and will outlive the run",
                    claim.subject)
            elif claim.artifact_key not in known_artifacts:
                report.add(
                    "claim_artifact_missing", Severity.BLOCKER,
                    f"claim {claim.claim_id} cites artifact "
                    f"{claim.artifact_key!r}, which no recorded step produced",
                    claim.subject, "the run manifest's artifact list")
            step_ok = (claim.step_id in provenance_steps
                       if claim.step_id else
                       (claim.interface in provenance_tools
                        if claim.interface else False))
            if not step_ok:
                report.add(
                    "claim_without_provenance", Severity.BLOCKER,
                    f"claim {claim.claim_id} names step "
                    f"{claim.step_id or claim.interface!r}, for which the "
                    f"manifest holds no provenance entry; the claim cannot be "
                    f"recomputed", claim.subject,
                    "the run manifest's step records")

    # -- 6. numeric ee needs a calibration source --------------------------
    def _check_numeric_ee(self, report: VerificationReport,
                          candidates: Sequence[Candidate],
                          claims: Sequence[Claim],
                          records: Sequence[ExperimentRecord]) -> None:
        """An ee is either measured, or predicted by something that named its
        calibration set. There is no third option.

        The candidate check does not rely on
        :class:`~eagent.schemas.candidate.StereoCall` having validated itself:
        a model constructed with ``model_construct`` or mutated after
        validation carries the field anyway, and this is the layer that is
        supposed to notice.
        """
        report.checks_run.append("numeric_ee_has_calibration_source")
        measured: set[str] = set()
        for rec in records:
            if rec.ee_target_pct is not None and rec.outcome.is_experimental \
                    and rec.detection.chiral_method_validated:
                for key in (rec.record_id, rec.accession, rec.sequence_sha256):
                    if key:
                        measured.add(str(key))

        for cand in candidates:
            stereo = cand.stereo
            if stereo.predicted_ee_pct is None:
                continue
            if not (stereo.calibration_source or "").strip():
                report.add(
                    "uncited_numeric_ee", Severity.BLOCKER,
                    f"{cand.candidate_id}: a predicted ee of "
                    f"{stereo.predicted_ee_pct} is reported with no "
                    f"calibration source. A number in percent that nothing "
                    f"was fitted to reads as a measurement and is not one",
                    cand.candidate_id, "the candidate's StereoCall fields")

        for claim in claims:
            value_is_ee = claim.is_ee and claim.value is not None
            prose = _EE_PATTERN.search(claim.text or "")
            if not value_is_ee and prose is None:
                continue
            if (claim.calibration_source or "").strip():
                continue
            if claim.subject and claim.subject in measured:
                continue
            quoted = (claim.value if value_is_ee
                      else (prose.group(1) or prose.group(2)))
            report.add(
                "uncited_numeric_ee", Severity.BLOCKER,
                f"claim {claim.claim_id} quotes an ee of {quoted} with no "
                f"calibration source and no validated chiral measurement "
                f"behind it", claim.subject,
                "the claim's own fields and the experimental records")

        for rec in records:
            if rec.ee_target_pct is None:
                continue
            if rec.detection.chiral_method_validated:
                continue
            report.add(
                "ee_without_validated_chiral_method", Severity.BLOCKER,
                f"record {rec.record_id} carries ee {rec.ee_target_pct} but "
                f"its detection block does not record a validated chiral "
                f"method; an ee from an unvalidated separation is a number, "
                f"not a configuration", rec.record_id,
                "the record's Detection block")

    # -- 7. uncalibrated-only disqualification -----------------------------
    def _check_uncalibrated_disqualification(
        self, report: VerificationReport, candidates: Sequence[Candidate],
        templates: Mapping[str, CatalyticTemplate],
        disqualifying: Mapping[str, Sequence[str]],
    ) -> None:
        """A window nobody fitted may lower confidence; it may not delete a candidate."""
        report.checks_run.append("no_uncalibrated_only_disqualification")
        for cand in candidates:
            if not cand.disqualified:
                continue
            template = self._template_for(cand, templates)
            if template is None:
                continue
            windows = {c.name: c for c in template.geometry_constraints}
            cited = [str(n) for n in disqualifying.get(cand.candidate_id, ())]
            if not cited:
                reason = cand.disqualification_reason or ""
                cited = [name for name in windows if name and name in reason]
            if not cited:
                continue
            theoretical = template.provenance.is_theoretical
            unfitted = [n for n in cited
                        if n in windows
                        and (theoretical or not windows[n].is_calibrated)]
            if len(unfitted) == len(cited):
                report.add(
                    "disqualified_by_uncalibrated_window", Severity.BLOCKER,
                    f"{cand.candidate_id} was disqualified only on "
                    f"{', '.join(cited)}, "
                    + ("whose template is a labelled theoretical model"
                       if theoretical else
                       "which carry no calibration set")
                    + ". An unfitted window may downgrade confidence; it may "
                      "not remove a candidate from the pool",
                    cand.candidate_id,
                    f"{template.template_id}.geometry_constraints.calibrated_on")

    # -- 8. restrained constraints excluded --------------------------------
    def _check_restrained_excluded(self, report: VerificationReport,
                                   candidates: Sequence[Candidate]) -> None:
        """Recompute the circular/independent split from the pose's restraints.

        Uses :class:`~eagent.science.robustness.CircularityGuard` on the pose's
        own ``restrained_constraints`` rather than reading the report's
        ``circular_constraints``, because the number under review is exactly
        the one the producing step wrote.
        """
        report.checks_run.append("restrained_constraints_excluded")
        from ..science.robustness import CircularityGuard

        for cand in candidates:
            poses = {p.pose_id: p for p in cand.poses}
            for geometry in cand.geometry:
                pose = poses.get(geometry.pose_id)
                if pose is None:
                    report.add(
                        "geometry_report_without_pose", Severity.BLOCKER,
                        f"{cand.candidate_id}: geometry report cites pose "
                        f"{geometry.pose_id!r}, which this candidate does not "
                        f"carry; the measurements belong to something else",
                        cand.candidate_id)
                    continue
                self._compare_circularity(report, cand, pose, geometry,
                                          CircularityGuard)

    @staticmethod
    def _compare_circularity(report: VerificationReport, cand: Candidate,
                             pose: ComplexPose, geometry: GeometryReport,
                             guard_cls: Any) -> None:
        subject = f"{cand.candidate_id}/{geometry.pose_id}"
        guard = guard_cls.for_report(pose, geometry)
        evidence = guard.independent_evidence(geometry)
        recomputed_circular = set(evidence.circular_all)
        declared_circular = set(geometry.circular_constraints)

        missed = sorted(recomputed_circular - declared_circular)
        if missed:
            report.add(
                "restrained_counted_as_independent", Severity.BLOCKER,
                f"{subject}: constraint(s) {', '.join(missed)} were restrained "
                f"while the pose was built and are not listed as circular. A "
                f"distance that was enforced cannot afterwards corroborate the "
                f"model that enforced it", subject,
                "the pose's restrained_constraints")
        if geometry.independent_satisfied > evidence.n_independent_satisfied:
            report.add(
                "independent_evidence_overcounted", Severity.BLOCKER,
                f"{subject}: the report counts "
                f"{geometry.independent_satisfied} independent satisfied "
                f"constraint(s); recomputing from the restraint list gives "
                f"{evidence.n_independent_satisfied}", subject,
                "the pose's restrained_constraints")
        if evidence.is_entirely_circular:
            report.add(
                "evidence_entirely_circular", Severity.BLOCKER,
                f"{subject}: every satisfied constraint "
                f"({', '.join(evidence.circular_satisfied)}) was one that had "
                f"been imposed; this pose carries no evidence that was not put "
                f"into it", subject, "the pose's restrained_constraints")

    # -- 9. no experimental negative for a computational failure -----------
    def _check_no_experimental_negative_for_computational_failure(
        self, report: VerificationReport, records: Sequence[ExperimentRecord],
        computational_failures: set[str],
    ) -> None:
        """Keep "the model crashed" out of the experimental record layer.

        Three ways the confusion gets written down, each checked: an
        experimental outcome on a subject whose only event was a computational
        failure; a negative with no detection method behind it; and an
        experimental outcome with no evidence reference at all.
        """
        report.checks_run.append("no_experimental_negative_for_computation")
        for rec in records:
            keys = {str(k) for k in
                    (rec.record_id, rec.accession, rec.sequence_sha256)
                    if k}
            if not rec.outcome.is_experimental:
                continue
            overlap = keys & computational_failures
            if overlap:
                report.add(
                    "experimental_label_on_computational_failure",
                    Severity.BLOCKER,
                    f"record {rec.record_id} is labelled "
                    f"'{rec.outcome.value}' -- an experimental claim -- for a "
                    f"subject whose only recorded event is a computational "
                    f"failure ({', '.join(sorted(overlap))}). A modelling "
                    f"failure says nothing about the enzyme", rec.record_id,
                    "the run's recorded computational failures")
            if rec.outcome is OutcomeClass.NO_TARGET_PRODUCT_DETECTED and \
                    not (rec.detection.method or "").strip():
                report.add(
                    "negative_without_detection_method", Severity.BLOCKER,
                    f"record {rec.record_id} claims no product was detected "
                    f"but names no detection method; a negative with no assay "
                    f"behind it is a computational failure wearing an "
                    f"experimental label", rec.record_id,
                    "the record's Detection block")
            if not rec.evidence:
                report.add(
                    "experimental_record_without_evidence", Severity.BLOCKER,
                    f"record {rec.record_id} carries the experimental outcome "
                    f"'{rec.outcome.value}' with no evidence reference; "
                    f"nothing says where the measurement came from",
                    rec.record_id, "the record's evidence list")

    # -- envelope ----------------------------------------------------------
    def _envelope(self, ctx: RunContext, report: VerificationReport,
                  write_report: bool) -> ToolResult:
        """Wrap the findings so the controller can stop on the blockers."""
        result = ToolResult(
            status=Status.SUCCESS,
            provenance=Provenance(
                tool=self.name, tool_version=self.version,
                parameters={"checks": list(report.checks_run),
                            "ignores_producer_conclusions":
                                sorted(IGNORED_PRODUCER_CONCLUSIONS)},
                started_at=report.at, finished_at=utc_now()),
        )
        for finding in report.findings:
            result.qc_flags.append(finding.to_flag())
        for item in report.unverifiable:
            result.add_uncertainty(
                "not_verifiable", item, resolvable_by=
                "supply the missing file, template or declaration")
        result.data["verification"] = report.to_dict()

        if write_report:
            path = ctx.path("verification", "verification_report.json")
            path.write_text(json.dumps(report.to_dict(), indent=2,
                                       ensure_ascii=False, default=str),
                            encoding="utf-8")
            result.artifacts.append(Artifact(
                key="verification_report", path=str(path), kind="file",
                n_records=len(report.findings),
                summary=("every check run, every finding, and everything that "
                         "could not be checked")))

        blockers = [f for f in report.findings if f.severity is Severity.BLOCKER]
        if blockers:
            result.status = Status.FAILED
            result.message = (
                f"verification failed: {len(blockers)} blocking problem(s). "
                f"The run must stop here; these are not trade-offs")
            result.add_next(
                "repair_and_reverify",
                "Fix the inputs the verifier names, then run it again",
                {"codes": sorted({f.code for f in blockers})},
                requires_human=True)
        elif report.unverifiable:
            result.status = Status.PARTIAL
            result.message = (
                f"no blocking problem found, but {len(report.unverifiable)} "
                f"check(s) could not be run; an unverified claim is not a "
                f"verified one")
        else:
            result.message = (
                f"{len(report.checks_run)} independent check(s) passed")
        return result


def _parse_residue_token(token: str) -> tuple[str, int] | None:
    """Split ``Y155`` into ``('Y', 155)``, or ``None`` when it is not one.

    Returns ``None`` rather than guessing: a token the verifier cannot parse is
    a token no downstream step can map either, and that is the finding.
    """
    match = re.fullmatch(r"\s*([A-Za-z])\s*(\d+)\s*", str(token))
    if match is None:
        return None
    return match.group(1).upper(), int(match.group(2))
