"""Tests for :mod:`eagent.science.diversity`.

The cases target the ways a batch quietly stops being an experiment: three
near-duplicates winning three diversity slots, one over-sequenced clade
occupying the plate, control genes being discovered after the construct cap was
already spent, and a short batch being padded back to 96 so the plan looks
complete.
"""

from __future__ import annotations

import unittest

from eagent.schemas import (
    BatchRole,
    Budget,
    Candidate,
    CatalyticMapping,
    ConfidenceLevel,
    ControlItem,
    FamilyAnnotation,
    ScoreDimension,
    SequenceRecord,
)
from eagent.science.diversity import (
    DEFAULT_ROLE_TARGETS,
    SignatureBasis,
    candidate_distance,
    compose_batch,
    facility_location_gain,
    family_quota_select,
    greedy_submodular_select,
    hamming_distance,
    is_model_uncertain,
    jaccard_distance,
    kmer_set,
    pocket_distance,
    pocket_signature,
    rank_utility,
    reserve_control_slots,
    sequence_distance,
)

BASE = ("MKAAVLYEFGKPLEIKEVEVAPPKAHEVRIKIAYTGVCHTDAYTLSGADPEGLFPVILGHEGAGIVESVGEGVT"
        "NVKPGDHVIPLYTPECGECKFCKSGKTNLCQKIRATQGKGLMPDGTSRFTCKGKEILHYMGCSTFSEYTVVAD")


def mutate(sequence: str, positions: dict[int, str]) -> str:
    chars = list(sequence)
    for index, letter in positions.items():
        chars[index] = letter
    return "".join(chars)


def candidate(
    cid: str,
    *,
    sequence: str = BASE,
    family: str = "SDR",
    cluster: str | None = None,
    pocket: dict[str, str] | None = None,
    gates_pass: bool | None = True,
    uncertainty: ConfidenceLevel = ConfidenceLevel.STRONG,
    evidence_value: float = 1.0,
    input_errors: list[str] | None = None,
) -> Candidate:
    """A candidate with just enough scorecard for selection to be legal."""
    cand = Candidate(
        candidate_id=cid,
        sequence_record=SequenceRecord(candidate_id=cid, sequence=sequence,
                                       is_fragment=False, percent_identity=60.0,
                                       search_method="mmseqs2"),
        family=FamilyAnnotation(family_name=family, sequence_cluster_id=cluster),
        catalytic_mapping=CatalyticMapping(
            catalytic_template_id="cat:sdr:kred",
            role_to_residue=dict(pocket or {"catalytic_Tyr": "Y155"}),
        ),
        input_errors=list(input_errors or []),
    )
    cand.set_dimension(ScoreDimension(
        name="catalytic_machinery_mappable", is_gate=True,
        gate_passed=gates_pass, direction="categorical",
        level=(ConfidenceLevel.STRONG if gates_pass
               else ConfidenceLevel.CONTRADICTORY if gates_pass is False
               else ConfidenceLevel.INSUFFICIENT)))
    cand.set_dimension(ScoreDimension(
        name="functional_literature_evidence", level=ConfidenceLevel.MODERATE,
        value=evidence_value))
    cand.set_dimension(ScoreDimension(
        name="model_uncertainty", level=uncertainty, direction="categorical"))
    return cand


# --------------------------------------------------------------------------
# Distances
# --------------------------------------------------------------------------

class TestDistances(unittest.TestCase):

    def test_jaccard_of_an_empty_set_is_none_not_one(self) -> None:
        """An uncharacterised candidate must not look maximally diverse."""
        self.assertIsNone(jaccard_distance(set(), {"a"}))
        self.assertIsNone(jaccard_distance({"a"}, set()))
        self.assertEqual(jaccard_distance({"a"}, {"a"}), 0.0)
        self.assertEqual(jaccard_distance({"a"}, {"b"}), 1.0)

    def test_hamming_refuses_unequal_lengths(self) -> None:
        with self.assertRaises(ValueError):
            hamming_distance("ACDE", "ACD")

    def test_hamming_is_normalised(self) -> None:
        self.assertEqual(hamming_distance("ACDE", "ACDE"), 0.0)
        self.assertEqual(hamming_distance("ACDE", "ACDF"), 0.25)

    def test_kmer_set_is_a_set(self) -> None:
        self.assertEqual(kmer_set("AAAAA", 3), frozenset({"AAA"}))
        self.assertEqual(len(kmer_set("ACDEF", 3)), 3)

    def test_identical_sequences_have_zero_distance(self) -> None:
        self.assertEqual(sequence_distance(BASE, BASE), 0.0)

    def test_distant_sequences_score_higher_than_near_duplicates(self) -> None:
        near = mutate(BASE, {10: "W"})
        far = BASE[::-1]
        self.assertLess(sequence_distance(BASE, near),
                        sequence_distance(BASE, far))

    def test_unknown_method_raises(self) -> None:
        with self.assertRaises(ValueError):
            sequence_distance(BASE, BASE, method="cosine")


class TestPocketSignature(unittest.TestCase):

    def test_signature_from_catalytic_roles_is_flagged_as_such(self) -> None:
        sig = pocket_signature(candidate("a"))
        self.assertIs(sig.basis, SignatureBasis.CATALYTIC_ROLES_ONLY)
        self.assertTrue(sig.tokens)

    def test_bare_shell_tokens_are_compared_by_composition(self) -> None:
        """A list of W110-style tokens states no frame, so it is not positional.

        ``W110`` in one protein and ``W111`` in another say nothing about each
        other: author numbering shifts with construct boundaries and tags.
        Treating the tokens as positions made two identical pockets maximally
        distant, so a bare list falls back to composition and says so.
        """
        sig = pocket_signature(candidate("a"),
                               extra_pocket_residues={"a": ["W110", "F147"]})
        self.assertIs(sig.basis, SignatureBasis.POCKET_COMPOSITION)
        self.assertIn("pocket_aa:W#1", sig.tokens)
        self.assertNotIn("pocket:W110", sig.tokens)

    def test_framed_shell_residues_are_positional(self) -> None:
        from eagent.science.pocket import pocket_residues_from_tokens
        shell = pocket_residues_from_tokens("a", ["W110", "F147"],
                                            frame="SDR/scheme:sdr-v1")
        sig = pocket_signature(candidate("a"), extra_pocket_residues={"a": shell})
        self.assertIs(sig.basis, SignatureBasis.ALIGNED_POCKET_RESIDUES)
        self.assertIn("pocket:W110", sig.tokens)
        self.assertEqual(sig.frame, "SDR/scheme:sdr-v1")

    def test_unmapped_candidate_has_an_empty_incomparable_signature(self) -> None:
        bare = candidate("bare")
        bare.catalytic_mapping = CatalyticMapping()
        sig = pocket_signature(bare)
        self.assertIs(sig.basis, SignatureBasis.EMPTY)
        self.assertIsNone(pocket_distance(bare, candidate("a")))

    def test_pocket_distance_sees_pocket_change_a_sequence_distance_misses(self) -> None:
        """Two near-identical sequences with different pockets are far apart."""
        a = candidate("a", pocket={"catalytic_Tyr": "Y155", "shell_1": "W110"})
        b = candidate("b", sequence=mutate(BASE, {5: "W"}),
                      pocket={"catalytic_Tyr": "Y155", "shell_1": "A110"})
        self.assertLess(sequence_distance(a, b), 0.2)
        dist = pocket_distance(a, b)
        assert dist is not None
        self.assertGreater(dist, 0.5)

    def test_candidate_distance_falls_back_to_sequence_when_unmapped(self) -> None:
        bare = candidate("bare")
        bare.catalytic_mapping = CatalyticMapping()
        self.assertGreaterEqual(candidate_distance(bare, candidate("a")), 0.0)
        with self.assertRaises(ValueError):
            candidate_distance(bare, candidate("a"), require_pocket=True)


# --------------------------------------------------------------------------
# Submodular selection
# --------------------------------------------------------------------------

class TestGreedySelection(unittest.TestCase):

    def setUp(self) -> None:
        # Three near-duplicates and two genuinely different pockets.
        self.items = [
            candidate("dup1", pocket={"s": "A1", "t": "A2"}),
            candidate("dup2", pocket={"s": "A1", "t": "A2"}),
            candidate("dup3", pocket={"s": "A1", "t": "A2"}),
            candidate("far1", pocket={"s": "W9", "t": "W8"}),
            candidate("far2", pocket={"s": "K3", "t": "K4"}),
        ]

    def distance(self, a: Candidate, b: Candidate) -> float:
        return candidate_distance(a, b)

    def test_coverage_term_prefers_a_spread_over_near_duplicates(self) -> None:
        chosen = greedy_submodular_select(
            self.items, 3, lambda _c: 0.0, self.distance, lambda_weight=1.0)
        ids = {c.candidate_id for c in chosen}
        self.assertIn("far1", ids)
        self.assertIn("far2", ids)
        self.assertEqual(len([i for i in ids if i.startswith("dup")]), 1,
                         "only one representative of the duplicate cluster is useful")

    def test_zero_lambda_collapses_to_pure_utility(self) -> None:
        utility = {"dup1": 0.9, "dup2": 0.8, "dup3": 0.7, "far1": 0.1, "far2": 0.0}
        chosen = greedy_submodular_select(
            self.items, 3, utility, self.distance, lambda_weight=0.0)
        self.assertEqual([c.candidate_id for c in chosen],
                         ["dup1", "dup2", "dup3"])

    def test_selection_is_deterministic_under_reordering(self) -> None:
        first = [c.candidate_id for c in greedy_submodular_select(
            self.items, 3, lambda _c: 0.0, self.distance)]
        second = [c.candidate_id for c in greedy_submodular_select(
            list(reversed(self.items)), 3, lambda _c: 0.0, self.distance)]
        self.assertEqual(first, second)

    def test_pool_smaller_than_k_returns_the_pool_without_padding(self) -> None:
        chosen = greedy_submodular_select(
            self.items[:2], 10, lambda _c: 0.0, self.distance)
        self.assertEqual(len(chosen), 2)

    def test_negative_lambda_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            greedy_submodular_select(self.items, 2, lambda _c: 0.0,
                                     self.distance, lambda_weight=-1.0)

    def test_facility_location_gain_falls_for_an_already_covered_pool(self) -> None:
        pool = self.items
        empty_coverage = {c.candidate_id: 0.0 for c in pool}
        gain_fresh = facility_location_gain(
            pool[0], pool, empty_coverage, lambda a, b: 1.0 - self.distance(a, b))
        full_coverage = {c.candidate_id: 1.0 for c in pool}
        gain_covered = facility_location_gain(
            pool[0], pool, full_coverage, lambda a, b: 1.0 - self.distance(a, b))
        self.assertGreater(gain_fresh, 0.0)
        self.assertEqual(gain_covered, 0.0)

    def test_rank_utility_is_monotone_and_bounded(self) -> None:
        utilities = rank_utility(self.items)
        values = [utilities[c.candidate_id] for c in self.items]
        self.assertEqual(values, sorted(values, reverse=True))
        self.assertLessEqual(max(values), 1.0)
        self.assertGreater(min(values), 0.0)


# --------------------------------------------------------------------------
# Quotas
# --------------------------------------------------------------------------

class TestQuotas(unittest.TestCase):

    def test_family_cap_stops_one_clade_taking_the_batch(self) -> None:
        pool = [candidate(f"sdr{i}", family="SDR") for i in range(10)]
        pool += [candidate("akr1", family="AKR"), candidate("akr2", family="AKR")]
        chosen = family_quota_select(pool, {"SDR": 3}, 8)
        families = [c.family.family_name for c in chosen]
        self.assertEqual(families.count("SDR"), 3)
        self.assertEqual(families.count("AKR"), 2)
        self.assertEqual(len(chosen), 5, "capped, never substituted up to k")

    def test_wildcard_quota_applies_to_unnamed_families(self) -> None:
        pool = ([candidate(f"a{i}", family="A") for i in range(4)]
                + [candidate(f"b{i}", family="B") for i in range(4)])
        chosen = family_quota_select(pool, {"*": 2}, 8)
        self.assertEqual(len(chosen), 4)

    def test_cluster_cap_is_finer_than_the_family_cap(self) -> None:
        pool = [candidate(f"c{i}", family="SDR", cluster="clade-1")
                for i in range(5)]
        pool += [candidate("other", family="SDR", cluster="clade-2")]
        chosen = family_quota_select(pool, {"SDR": 10}, 6, cluster_cap=2)
        self.assertEqual(len(chosen), 3)

    def test_input_order_is_respected(self) -> None:
        pool = [candidate("z", family="A"), candidate("a", family="A")]
        self.assertEqual([c.candidate_id for c in family_quota_select(pool, None, 1)],
                         ["z"])


# --------------------------------------------------------------------------
# Controls
# --------------------------------------------------------------------------

class TestControlReservation(unittest.TestCase):

    def controls(self) -> list[ControlItem]:
        return [
            ControlItem(name="no enzyme", kind="no_enzyme",
                        demonstrates="the background reaction rate"),
            ControlItem(name="known KRED", kind="positive_enzyme",
                        demonstrates="the assay system works on a known substrate",
                        requires_new_construct=True),
            ControlItem(name="empty vector", kind="empty_vector",
                        demonstrates="the host contributes no activity",
                        requires_new_construct=True),
        ]

    def test_control_genes_reduce_the_candidate_slots(self) -> None:
        budget = Budget(new_constructs_round_1=96, constructs_include_controls=True)
        self.assertEqual(budget.candidate_slots, 96)
        reservation = reserve_control_slots(budget, self.controls())
        self.assertEqual(reservation.reserved_slots, 2)
        self.assertEqual(reservation.candidate_slots, 94)
        self.assertEqual(reservation.budget.candidate_slots, 94)

    def test_new_construct_controls_are_marked_as_occupying_slots(self) -> None:
        budget = Budget(new_constructs_round_1=96, constructs_include_controls=True)
        reservation = reserve_control_slots(budget, self.controls())
        for control in reservation.controls:
            if control.requires_new_construct:
                self.assertTrue(control.occupies_batch_slot)

    def test_separate_budget_line_reserves_nothing_but_says_so(self) -> None:
        budget = Budget(new_constructs_round_1=96, constructs_include_controls=False)
        reservation = reserve_control_slots(budget, self.controls())
        self.assertEqual(reservation.reserved_slots, 0)
        self.assertEqual(reservation.candidate_slots, 96)
        self.assertIn("separate synthesis budget", reservation.note)

    def test_a_control_plan_bigger_than_the_round_is_refused(self) -> None:
        budget = Budget(new_constructs_round_1=1, detailed_complex_target=300,
                        constructs_include_controls=True)
        with self.assertRaises(ValueError):
            reserve_control_slots(budget, self.controls())


# --------------------------------------------------------------------------
# compose_batch
# --------------------------------------------------------------------------

class TestComposeBatch(unittest.TestCase):

    def budget(self, slots: int = 12) -> Budget:
        return Budget(new_constructs_round_1=slots, detailed_complex_target=300)

    def targets(self, high: int, diversity: int, probe: int) -> dict[BatchRole, int]:
        return {BatchRole.HIGH_EVIDENCE: high,
                BatchRole.DIVERSITY: diversity,
                BatchRole.UNCERTAINTY_PROBE: probe}

    def pool(self, n: int, **kwargs) -> list[Candidate]:
        return [candidate(f"c{i:02d}", pocket={"s": f"R{i}", "t": f"Q{i}"},
                          evidence_value=float(n - i), **kwargs)
                for i in range(n)]

    def test_default_role_targets_are_the_documented_48_24_24(self) -> None:
        self.assertEqual(DEFAULT_ROLE_TARGETS[BatchRole.HIGH_EVIDENCE], 48)
        self.assertEqual(DEFAULT_ROLE_TARGETS[BatchRole.DIVERSITY], 24)
        self.assertEqual(DEFAULT_ROLE_TARGETS[BatchRole.UNCERTAINTY_PROBE], 24)
        self.assertEqual(sum(DEFAULT_ROLE_TARGETS.values()), 96)

    def test_a_full_batch_fills_every_role(self) -> None:
        pool = self.pool(8, uncertainty=ConfidenceLevel.STRONG)
        pool += [candidate(f"u{i}", pocket={"s": f"Z{i}"},
                           uncertainty=ConfidenceLevel.WEAK) for i in range(4)]
        plan = compose_batch(pool, self.budget(12), self.targets(6, 3, 3), None)
        self.assertEqual(plan.n_candidates, 12)
        self.assertIsNone(plan.shortfall_reason)
        counts = plan.role_counts()
        self.assertEqual(counts[BatchRole.HIGH_EVIDENCE.value], 6)
        self.assertEqual(counts[BatchRole.UNCERTAINTY_PROBE.value], 3)
        self.assertEqual(counts[BatchRole.DIVERSITY.value], 3)

    def test_short_batch_is_reported_not_padded(self) -> None:
        """Four qualifying candidates cannot become twelve."""
        pool = self.pool(4)
        pool += [candidate(f"bad{i}", gates_pass=False) for i in range(20)]
        plan = compose_batch(pool, self.budget(12), self.targets(6, 3, 3), None)
        self.assertEqual(plan.n_candidates, 4)
        self.assertTrue(plan.is_short)
        self.assertIsNotNone(plan.shortfall_reason)
        assert plan.shortfall_reason is not None
        self.assertIn("4 of 12", plan.shortfall_reason)
        self.assertIn("failed a gate", plan.shortfall_reason)
        selected = {m.candidate_id for m in plan.members}
        self.assertFalse(any(cid.startswith("bad") for cid in selected))

    def test_undecided_gates_are_counted_separately_from_rejections(self) -> None:
        pool = self.pool(2)
        pool += [candidate(f"und{i}", gates_pass=None) for i in range(5)]
        plan = compose_batch(pool, self.budget(12), self.targets(6, 3, 3), None)
        self.assertEqual(plan.n_candidates, 2)
        assert plan.shortfall_reason is not None
        self.assertIn("5 had an undecided gate", plan.shortfall_reason)
        self.assertIn("cheapest way to lengthen", plan.shortfall_reason)

    def test_disqualified_candidates_never_enter(self) -> None:
        pool = self.pool(3)
        pool.append(candidate("defect", input_errors=["wrong ligand supplied"]))
        plan = compose_batch(pool, self.budget(12), self.targets(6, 3, 3), None)
        self.assertNotIn("defect", {m.candidate_id for m in plan.members})

    def test_quota_caps_the_over_represented_clade(self) -> None:
        pool = [candidate(f"sdr{i:02d}", family="SDR", cluster="clade-1",
                          pocket={"s": f"R{i}"}, evidence_value=100.0 - i)
                for i in range(10)]
        pool += [candidate(f"akr{i}", family="AKR", cluster="clade-2",
                           pocket={"s": f"K{i}"}, evidence_value=1.0)
                 for i in range(2)]
        plan = compose_batch(pool, self.budget(12), self.targets(6, 3, 3),
                             {"SDR": 4})
        families = [m.family for m in plan.members]
        self.assertEqual(families.count("SDR"), 4)
        self.assertEqual(families.count("AKR"), 2)
        self.assertTrue(plan.is_short)

    def test_probes_draw_only_from_gate_passing_candidates(self) -> None:
        pool = self.pool(3, uncertainty=ConfidenceLevel.STRONG)
        pool += [candidate(f"uncertain_failed{i}", gates_pass=False,
                           uncertainty=ConfidenceLevel.CONTRADICTORY)
                 for i in range(5)]
        pool += [candidate(f"uncertain_ok{i}", pocket={"s": f"W{i}"},
                           uncertainty=ConfidenceLevel.INSUFFICIENT)
                 for i in range(2)]
        plan = compose_batch(pool, self.budget(12), self.targets(2, 0, 4), None)
        probes = [m.candidate_id for m in plan.members
                  if m.role is BatchRole.UNCERTAINTY_PROBE]
        self.assertTrue(probes)
        self.assertTrue(all(p.startswith("uncertain_ok") for p in probes))

    def test_a_confident_model_is_not_a_probe(self) -> None:
        confident = candidate("sure", uncertainty=ConfidenceLevel.STRONG)
        unsure = candidate("unsure", uncertainty=ConfidenceLevel.WEAK)
        self.assertFalse(is_model_uncertain(confident))
        self.assertTrue(is_model_uncertain(unsure))

    def test_controls_reduce_the_requested_slots_of_the_plan(self) -> None:
        budget = Budget(new_constructs_round_1=12, detailed_complex_target=300,
                        constructs_include_controls=True)
        controls = [ControlItem(name="positive", kind="positive_enzyme",
                                demonstrates="the assay system works",
                                requires_new_construct=True),
                    ControlItem(name="no enzyme", kind="no_enzyme",
                                demonstrates="the background rate")]
        pool = self.pool(16)
        pool += [candidate(f"u{i}", pocket={"s": f"Z{i}"},
                           uncertainty=ConfidenceLevel.WEAK) for i in range(4)]
        plan = compose_batch(pool, budget, self.targets(6, 3, 3), None,
                             controls=controls)
        self.assertEqual(plan.requested_slots, 11,
                         "one control needs a gene, so 12 constructs buy 11 candidates")
        self.assertEqual(plan.n_candidates, 11)
        self.assertIsNone(plan.shortfall_reason)
        self.assertEqual(len(plan.controls), 2)

    def test_an_empty_probe_pool_leaves_slots_unfilled_by_default(self) -> None:
        """A role's slots are not quietly repurposed: that changes the experiment."""
        pool = self.pool(20, uncertainty=ConfidenceLevel.STRONG)
        plan = compose_batch(pool, self.budget(12), self.targets(6, 3, 3), None)
        self.assertEqual(plan.n_candidates, 9)
        assert plan.shortfall_reason is not None
        self.assertIn("uncertainty_probe=3", plan.shortfall_reason)
        self.assertIn("not reallocated", plan.shortfall_reason)

    def test_reallocation_is_opt_in_and_relabels_the_slots(self) -> None:
        pool = self.pool(20, uncertainty=ConfidenceLevel.STRONG)
        plan = compose_batch(pool, self.budget(12), self.targets(6, 3, 3), None,
                             reallocate_unfilled_roles=True)
        self.assertEqual(plan.n_candidates, 12)
        self.assertIsNone(plan.shortfall_reason)
        absorbed = [m for m in plan.members if "absorbed a slot" in m.selection_reason]
        self.assertEqual(len(absorbed), 3)
        self.assertTrue(all(m.role is BatchRole.DIVERSITY for m in absorbed))

    def test_role_targets_are_rescaled_to_the_reserved_budget(self) -> None:
        budget = Budget(new_constructs_round_1=96, detailed_complex_target=300,
                        constructs_include_controls=True)
        controls = [ControlItem(name=f"ctl{i}", kind="positive_enzyme",
                                demonstrates="the assay system works",
                                requires_new_construct=True) for i in range(2)]
        pool = self.pool(120, uncertainty=ConfidenceLevel.WEAK)
        plan = compose_batch(pool, budget, None, None, controls=controls)
        self.assertEqual(plan.requested_slots, 94)
        self.assertEqual(plan.n_candidates, 94)
        counts = plan.role_counts()
        self.assertEqual(sum(counts.values()), 94)
        self.assertGreater(counts[BatchRole.HIGH_EVIDENCE.value],
                           counts[BatchRole.DIVERSITY.value])

    def test_ungated_candidates_raise_rather_than_being_admitted(self) -> None:
        bare = Candidate(candidate_id="bare",
                         sequence_record=SequenceRecord(candidate_id="bare",
                                                        sequence=BASE))
        with self.assertRaises(ValueError):
            compose_batch([bare], self.budget(12), self.targets(6, 3, 3), None)

    def test_controls_cannot_be_requested_as_a_candidate_role(self) -> None:
        with self.assertRaises(ValueError):
            compose_batch(self.pool(3), self.budget(12),
                          {BatchRole.CONTROL: 2}, None)

    def test_measurement_footprint_counts_wells_not_genes(self) -> None:
        plan = compose_batch(self.pool(6), self.budget(12),
                             self.targets(6, 0, 0), None)
        self.assertEqual(plan.measurement_units,
                         plan.n_candidates * plan.cofactor_conditions
                         * plan.replicates)

    def test_composition_is_deterministic(self) -> None:
        pool = self.pool(10) + [candidate(f"u{i}", pocket={"s": f"Z{i}"},
                                          uncertainty=ConfidenceLevel.WEAK)
                                for i in range(4)]
        first = compose_batch(pool, self.budget(12), self.targets(6, 3, 3), None)
        second = compose_batch(list(reversed(pool)), self.budget(12),
                               self.targets(6, 3, 3), None)
        self.assertEqual([(m.candidate_id, m.role) for m in first.members],
                         [(m.candidate_id, m.role) for m in second.members])


if __name__ == "__main__":
    unittest.main(verbosity=2)
