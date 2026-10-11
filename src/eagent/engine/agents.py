"""Compute agents: the project's own science functions, called with strings.

Each agent here is a few lines around a function that already exists, is already
tested, and is documented at length where it is defined. What this file adds is
only the part a model-facing surface needs: strict parsing of string arguments,
the project's own refusals passed through unchanged, and a result shaped so
every number has a field name it can be cited by.

THE SHAPE OF A VALUE
====================
Quantities sit at the top level of the value, under names that carry their
meaning (``identity_percent_over_aligned_columns``, not ``identity``). That is
not decoration. A citation names a field, the numeric guard checks the number
against that field, and a nested or ambiguous name is a way for a correct
number to be cited against the wrong thing. Lists and notes ride alongside and
are not citable.

WHY THESE SEVEN
===============
They are the ones whose answer a person would otherwise take on trust from a
model: an alignment identity, a sequence distance, a binomial interval, a
distance or torsion, and whether a list of substitutions is consistent with the
parent sequence. Models are fluent and unreliable at exactly these. The last two
agents (in :mod:`eagent.engine.pipeline`) say what the pipeline needs from an
operator and where each of its ten interfaces can run, so a model asked "what do
I have to decide next" reads the project's own answer instead of inventing one.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import ValidationError

from ..schemas.variant import Mutation
from ..science.diversity import DEFAULT_KMER_SIZE, sequence_distance
from ..science.geometry import angle, centroid, dihedral, distance
from ..science.numbering import alignment_score, needleman_wunsch
from ..science.robustness import DEFAULT_WILSON_Z, wilson_interval
from .core import (
    AgentRefusal, AgentSpec, choice_arg, integer_arg, number_arg, register,
    sequence_arg, text_arg,
)

#: Residues per sequence. The aligner is O(n*m) in time and in pointer-matrix
#: memory; 1200 x 1200 is comfortable on a phone and about a second on a laptop.
MAX_SEQUENCE = 1200

#: Alignment columns shown as text. Longer alignments report their numbers only,
#: because the scalars are the finding and the strings are a courtesy.
MAX_ALIGNMENT_TEXT = 400

#: Substitutions checked in one call.
MAX_VARIANTS = 40

#: Two-sided normal quantiles for the confidences the interval agent offers. The
#: 95% value is the project's own default, so this agent agrees with every other
#: place the project computes a Wilson interval.
_CONFIDENCE_Z = {"0.90": 1.6448536269514722, "0.95": DEFAULT_WILSON_Z,
                 "0.99": 2.5758293035489004}

_COORDINATE = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", re.ASCII)
#: Between two coordinates: one comma with any spaces around it, or spaces alone.
#: The first draft of this grammar was one regular expression with
#: ``\s*[,\s]\s*`` between the coordinates, whose three parts can share a run of
#: spaces in as many ways as it is long, so a failed match took cubic time (a
#: 1,400-character argument held the worker for 15 s). Splitting first and
#: checking each piece is linear, and accepts and refuses exactly the same text.
_SEPARATOR = re.compile(r"\s*,\s*|\s+", re.ASCII)
_ASCII_SPACE = " \t\n\r\f\v"
_VARIANT = re.compile(r"([A-Za-z])(\d{1,5})([A-Za-z])", re.ASCII)


def _echo(text: str, limit: int) -> str:
    """At most ``limit`` printable ASCII characters of ``text``, for a message.

    A refusal may quote what it refused, but never more than a few characters and
    never a character JSON would have to escape at six bytes apiece: the reply has
    a fixed size, and what the caller sent is the one thing in it the caller controls.
    """
    text = re.sub(r"\s+", " ", text, flags=re.ASCII)
    shown = "".join(char if " " <= char <= "~" and char not in "\"\\" else "?" for char in text[:limit])
    return shown + ("..." if len(text) > limit else "")


# ---------------------------------------------------------------------------
# align_sequences
# ---------------------------------------------------------------------------

def _align_sequences(args: dict[str, str]) -> dict[str, Any]:
    a = sequence_arg(args, "sequence_a", maximum=MAX_SEQUENCE)
    b = sequence_arg(args, "sequence_b", maximum=MAX_SEQUENCE)
    match = number_arg(args, "match", 2.0, minimum=-100, maximum=100)
    mismatch = number_arg(args, "mismatch", -1.0, minimum=-100, maximum=100)
    gap_open = number_arg(args, "gap_open", 10.0, minimum=0, maximum=100)
    gap_extend = number_arg(args, "gap_extend", 0.5, minimum=0, maximum=100)
    alignment = needleman_wunsch(a, b, match, mismatch, gap_open, gap_extend)
    value: dict[str, Any] = {
        "method": "global alignment with affine gaps (Gotoh)",
        "length_a": len(a),
        "length_b": len(b),
        "alignment_columns": alignment.length,
        "aligned_pair_columns": alignment.n_aligned_columns,
        "identical_columns": alignment.n_identities,
        "gap_columns": alignment.n_gaps,
        "identity_over_aligned_columns": alignment.identity,
        "identity_percent_over_aligned_columns": alignment.identity * 100.0,
        "coverage_a": alignment.coverage_a(),
        "coverage_b": alignment.coverage_b(),
        "coverage_a_percent": alignment.coverage_a() * 100.0,
        "coverage_b_percent": alignment.coverage_b() * 100.0,
        "score": alignment_score(alignment.aligned_a, alignment.aligned_b, match,
                                 mismatch, gap_open, gap_extend),
        "match": match, "mismatch": mismatch,
        "gap_open": gap_open, "gap_extend": gap_extend,
        "note": ("Identity is counted over aligned columns only, so a high identity over "
                 "few aligned columns is not a good alignment: read it with the coverage "
                 "and the gap count. match and mismatch are scores; gap_open and "
                 "gap_extend are penalty magnitudes."),
    }
    if alignment.length <= MAX_ALIGNMENT_TEXT:
        value["aligned_a"] = alignment.aligned_a
        value["aligned_b"] = alignment.aligned_b
    else:
        value["alignment_text"] = (f"omitted: {alignment.length} columns exceeds "
                                   f"{MAX_ALIGNMENT_TEXT}; the numbers above are complete")
    return value


register(AgentSpec(
    name="align_sequences",
    description=("Globally align two protein sequences (affine gaps) and report identity "
                 "over aligned columns, coverage, gaps and score. Runs the project's own "
                 "aligner on the user's CPU."),
    parameters={
        "sequence_a": "one-letter residue codes; whitespace is ignored",
        "sequence_b": "one-letter residue codes; whitespace is ignored",
        "match": "optional: score for identical residues (default 2)",
        "mismatch": "optional: score for different residues (default -1)",
        "gap_open": "optional: penalty magnitude to open a gap (default 10)",
        "gap_extend": "optional: penalty magnitude per gap residue (default 0.5)",
    },
    required=("sequence_a", "sequence_b"),
    max_chars={"sequence_a": MAX_SEQUENCE * 2, "sequence_b": MAX_SEQUENCE * 2},
    run=_align_sequences,
    wraps="eagent.science.numbering.needleman_wunsch, alignment_score",
))


# ---------------------------------------------------------------------------
# sequence_distance
# ---------------------------------------------------------------------------

def _sequence_distance(args: dict[str, str]) -> dict[str, Any]:
    a = sequence_arg(args, "sequence_a", maximum=MAX_SEQUENCE)
    b = sequence_arg(args, "sequence_b", maximum=MAX_SEQUENCE)
    method = choice_arg(args, "method", ("kmer_jaccard", "hamming"), "kmer_jaccard")
    k = integer_arg(args, "k", DEFAULT_KMER_SIZE, minimum=1, maximum=8)
    distance_value = sequence_distance(a, b, method=method, k=k)
    value: dict[str, Any] = {
        "method": method,
        "distance": distance_value,
        "similarity": 1.0 - distance_value,
        "distance_percent": distance_value * 100.0,
        "similarity_percent": (1.0 - distance_value) * 100.0,
        "length_a": len(a),
        "length_b": len(b),
        "note": ("A global distance is dominated by scaffold variation. kmer_jaccard needs "
                 "no alignment; hamming needs equal-length, already aligned sequences."),
    }
    if method == "kmer_jaccard":
        value["k"] = k
    return value


register(AgentSpec(
    name="sequence_distance",
    description=("Distance between two sequences in [0, 1], by k-mer Jaccard (no alignment "
                 "needed) or Hamming (equal-length aligned sequences)."),
    parameters={
        "sequence_a": "one-letter residue codes; whitespace is ignored",
        "sequence_b": "one-letter residue codes; whitespace is ignored",
        "method": "optional: kmer_jaccard (default) or hamming",
        "k": "optional: k-mer size for kmer_jaccard, 1 to 8 (default 3)",
    },
    required=("sequence_a", "sequence_b"),
    max_chars={"sequence_a": MAX_SEQUENCE * 2, "sequence_b": MAX_SEQUENCE * 2},
    run=_sequence_distance,
    wraps="eagent.science.diversity.sequence_distance",
))


# ---------------------------------------------------------------------------
# wilson_interval
# ---------------------------------------------------------------------------

def _wilson(args: dict[str, str]) -> dict[str, Any]:
    successes = integer_arg(args, "successes", None, minimum=0, maximum=10**9)
    trials = integer_arg(args, "trials", None, minimum=0, maximum=10**9)
    confidence = choice_arg(args, "confidence", tuple(_CONFIDENCE_Z), "0.95")
    z = _CONFIDENCE_Z[confidence]
    lower, upper = wilson_interval(successes, trials, z)
    return {
        "successes": successes,
        "trials": trials,
        "proportion": successes / trials,
        "lower": lower,
        "upper": upper,
        "confidence": float(confidence),
        # The same quantities as percentages, because that is how they are written
        # in prose, and a quoted "95%" must be citable against a field that says 95.
        "proportion_percent": successes / trials * 100.0,
        "lower_percent": lower * 100.0,
        "upper_percent": upper * 100.0,
        "confidence_percent": float(confidence) * 100.0,
        "z": z,
        "note": ("A sampling interval for a binomial proportion. It does not model error in "
                 "the method that produced the successes, which is usually larger."),
    }


register(AgentSpec(
    name="wilson_interval",
    description=("Wilson score interval for successes out of trials, honest at 0 and all "
                 "successes where the textbook interval collapses to zero width."),
    parameters={
        "successes": "whole number, 0 up to trials",
        "trials": "whole number, at least 1",
        "confidence": "optional: 0.90, 0.95 (default) or 0.99",
    },
    required=("successes", "trials"),
    run=_wilson,
    wraps="eagent.science.robustness.wilson_interval",
))


# ---------------------------------------------------------------------------
# geometry_measure
# ---------------------------------------------------------------------------

_POINTS_NEEDED = {"distance": 2, "angle": 3, "dihedral": 4}
_MAX_CENTROID_POINTS = 20


def _points(raw: str) -> list[tuple[float, float, float]]:
    chunks = [chunk for chunk in re.split(r"[;\n]", raw) if chunk.strip()]
    points: list[tuple[float, float, float]] = []
    for chunk in chunks:
        parts = _SEPARATOR.split(chunk.strip(_ASCII_SPACE))
        if len(parts) != 3 or not all(_COORDINATE.fullmatch(part) for part in parts):
            raise AgentRefusal(f"could not read {_echo(chunk.strip(), 60)!r} as three coordinates "
                               f"'x, y, z'; separate points with ';'")
        point = tuple(float(part) for part in parts)
        if any(abs(part) > 1e6 for part in point):
            raise AgentRefusal("coordinates must be within 1e6 angstroms of the origin")
        points.append(point)  # type: ignore[arg-type]
    return points


def _geometry(args: dict[str, str]) -> dict[str, Any]:
    measure = choice_arg(args, "measure", ("distance", "angle", "dihedral", "centroid"), None)
    points = _points(text_arg(args, "points"))
    if measure == "centroid":
        if not 1 <= len(points) <= _MAX_CENTROID_POINTS:
            raise AgentRefusal(f"centroid needs 1 to {_MAX_CENTROID_POINTS} points, "
                               f"got {len(points)}")
        x, y, z = centroid(points)
        return {"measure": "centroid", "n_points": len(points),
                "centroid_x_angstrom": x, "centroid_y_angstrom": y, "centroid_z_angstrom": z,
                "note": "Unweighted mean position. Coordinates are taken as angstroms; no unit conversion happens."}
    needed = _POINTS_NEEDED[measure]
    if len(points) != needed:
        raise AgentRefusal(f"{measure} needs exactly {needed} points, got {len(points)}")
    if measure == "distance":
        return {"measure": "distance", "distance_angstrom": distance(*points),
                "note": "Coordinates are taken as angstroms; no unit conversion happens."}
    if measure == "angle":
        return {"measure": "angle", "angle_degrees": angle(*points),
                "note": "Angle a-b-c with the second point at the vertex, 0 to 180 degrees."}
    return {"measure": "dihedral", "dihedral_degrees": dihedral(*points),
            "note": ("Torsion a-b-c-d, IUPAC sign convention, in (-180, 180]. The sign is "
                     "kept: it is the difference between two prochiral faces.")}


register(AgentSpec(
    name="geometry_measure",
    description=("Distance, angle, torsion or centroid from explicit coordinates, using the "
                 "project's geometry code. Coordinates are angstroms."),
    parameters={
        "measure": "distance (2 points), angle (3, vertex second), dihedral (4) or centroid",
        "points": "'x, y, z' triples separated by ';', for example '0,0,0; 3,4,0'",
    },
    required=("measure", "points"),
    run=_geometry,
    wraps="eagent.science.geometry.distance, angle, dihedral, centroid",
))


# ---------------------------------------------------------------------------
# variant_check
# ---------------------------------------------------------------------------

def _variant_check(args: dict[str, str]) -> dict[str, Any]:
    sequence = sequence_arg(args, "sequence", maximum=MAX_SEQUENCE)
    first = integer_arg(args, "first_position", 1, minimum=0, maximum=100000)
    tokens = [token for token in re.split(r"[\s,;/]+", text_arg(args, "variants").strip()) if token]
    if not tokens:
        raise AgentRefusal("variants is empty; give substitutions such as 'A123G, L45V'")
    if len(tokens) > MAX_VARIANTS:
        raise AgentRefusal(f"{len(tokens)} variants given; this agent checks at most {MAX_VARIANTS}")
    checked: list[dict[str, Any]] = []
    n_valid = n_in_range = n_match = 0
    for token in tokens:
        found = _VARIANT.fullmatch(token)
        if not found:
            checked.append({"label": _echo(token, 16), "valid": False,
                            "error": "not of the form <residue><position><residue>"})
            continue
        wild_type, position, mutant = found.group(1).upper(), int(found.group(2)), found.group(3).upper()
        index = position - first
        try:
            Mutation(wild_type=wild_type, position_author=position,
                     position_index=index, mutant=mutant)
        except ValidationError as exc:
            message = exc.errors()[0]["msg"].removeprefix("Value error, ")
            checked.append({"label": f"{wild_type}{position}{mutant}", "valid": False,
                            "error": message})
            continue
        n_valid += 1
        in_range = 0 <= index < len(sequence)
        observed = sequence[index] if in_range else None
        matches = in_range and observed == wild_type
        n_in_range += in_range
        n_match += matches
        checked.append({"label": f"{wild_type}{position}{mutant}", "valid": True,
                        "sequence_residue": observed, "in_range": in_range,
                        "matches_wild_type": matches})
    return {
        "sequence_length": len(sequence),
        "first_position": first,
        "last_position": first + len(sequence) - 1,
        "n_variants": len(tokens),
        "n_valid_substitutions": n_valid,
        "n_invalid": len(tokens) - n_valid,
        "n_in_range": n_in_range,
        "n_out_of_range": n_valid - n_in_range,
        "n_wild_type_matches": n_match,
        # Only a position that exists can mismatch; one past the end is out of range,
        # which is a different problem (the numbering, or the wrong construct).
        "n_wild_type_mismatches": n_in_range - n_match,
        "variants": checked,
        "note": ("Each substitution is checked on its own against the sequence you gave, "
                 "numbered from first_position. A mismatch usually means the numbering "
                 "differs from the sequence (a tag, a signal peptide, a different construct)."),
    }


register(AgentSpec(
    name="variant_check",
    description=("Check substitutions such as 'A123G' against a parent sequence: is the "
                 "wild-type residue really at that position? Catches mis-numbered or "
                 "invented mutations before they reach an order sheet."),
    parameters={
        "sequence": "the parent sequence, one-letter codes",
        "variants": "substitutions separated by commas, spaces, ';' or '/', e.g. 'Y190F, H150N'",
        "first_position": "optional: the number of the first residue in the sequence (default 1)",
    },
    required=("sequence", "variants"),
    max_chars={"sequence": MAX_SEQUENCE * 2},
    run=_variant_check,
    wraps="eagent.schemas.variant.Mutation",
))
