"""Whatever a model sends, the engine answers with JSON and never raises.

A deterministic fuzz of every agent's call surface. It does not check that an
answer is right (the agents' own tests do); it checks the envelope's promises
hold for inputs nobody planned for: the call returns, the reply parses, a success
carries a citation that matches its value, a refusal carries none, nothing is
non-finite, and nothing takes long.
"""

from __future__ import annotations

import json
import math
import random
import string
import time
import unittest

import eagent.engine as engine
from eagent.engine import core

SEED = 20261011
ALPHABETS = (
    string.ascii_uppercase,
    string.ascii_lowercase + " \t\n",
    "ACDEFGHIKLMNPQRSTVWY",
    "0123456789.,;:-+eE_ ",
    "٣٤éıſK ​‮\u0000\U0001f9ec",
    "{}[]()\"'\\/<>&%$#@!?*^~`|",
    "K2AY9FT3S",
    "-0123456789.,; ",
)
SPECIALS = ("", " ", "0", "-0", "1e999", "-1e999", "nan", "inf", "NaN", "1_0", "0x10", "1e-999", "9" * 400,
            "0,0,0", "0,0,0;0,0,0", "1,2", "a,b,c", "task_id: T1", "task_id: T1\nx: &a 1\ny: *a",
            "KRED_PILOT_001", "../KRED_PILOT_001", "A" * 5000, "M" * 1201, "M" * 1200)


def noise(rng: random.Random) -> str:
    kind = rng.random()
    if kind < 0.25:
        return rng.choice(SPECIALS)
    alphabet = rng.choice(ALPHABETS)
    return "".join(rng.choice(alphabet) for _ in range(rng.choice((0, 1, 2, 5, 20, 80, 400, 1500, 3000))))


#: Calls that are valid as written. Random strings almost never are, so half the
#: fuzz starts here and damages one argument, which is how it reaches the code
#: behind the argument parsers.
SEEDS = {
    "align_sequences": [{"sequence_a": "MKTAYIAKQR", "sequence_b": "MKTAYLAKQR"},
                        {"sequence_a": "MKTAYIAKQRQISF", "sequence_b": "MKTAYIAKQR", "match": "3", "mismatch": "-2",
                         "gap_open": "8", "gap_extend": "1"}],
    "sequence_distance": [{"sequence_a": "ACDEFGHIK", "sequence_b": "ACDEFGHIR", "method": "hamming"},
                          {"sequence_a": "MKTAYIAKQR", "sequence_b": "MKTAYLAKQR", "method": "kmer_jaccard", "k": "3"}],
    "wilson_interval": [{"successes": "7", "trials": "20"}, {"successes": "0", "trials": "5", "confidence": "0.99"}],
    "geometry_measure": [{"measure": "distance", "points": "0,0,0; 3,4,0"},
                         {"measure": "angle", "points": "1,0,0; 0,0,0; 0,1,0"},
                         {"measure": "dihedral", "points": "1,0,0; 0,0,0; 0,1,0; 0,1,1"},
                         {"measure": "centroid", "points": "0,0,0; 2,0,0; 1,3,0"}],
    "variant_check": [{"sequence": "MKTAYIAKQR", "variants": "K2A, T3S, Y9F"},
                      {"sequence": "MKTAYIAKQR", "variants": "K12A; A1G", "first_position": "10"}],
    "pipeline_catalog": [{}],
    "task_gate_report": [{"task": "KRED_PILOT_001"}, {"task_yaml": "task_id: T1\n"}],
    "run_pipeline": [{"task": "KRED_PILOT_001"}, {"task_yaml": "task_id: T1\n"}],
}


def damage(rng: random.Random, text: str) -> str:
    choice = rng.random()
    if not text or choice < 0.1:
        return noise(rng)
    index = rng.randrange(len(text))
    if choice < 0.35:
        return text[:index] + rng.choice(rng.choice(ALPHABETS)) + text[index + 1:]
    if choice < 0.5:
        return text[:index] + text[index + 1:]
    if choice < 0.7:
        return text[:index] + rng.choice(rng.choice(ALPHABETS)) + text[index:]
    if choice < 0.85:
        return text * rng.choice((2, 5, 40))
    return text[:index]


def arguments_for(rng: random.Random, spec: core.AgentSpec) -> str:
    """A JSON string that is usually an object over the agent's parameters, sometimes not."""
    roll = rng.random()
    if roll < 0.03:
        return rng.choice(("", "null", "[]", "5", '"x"', "{", "{}}", "[1,2", "\u0000", "{\"a\":"))
    if roll < 0.55:
        chosen = dict(rng.choice(SEEDS[spec.name]))
        for _ in range(rng.choice((0, 1, 1, 2))):
            if chosen and rng.random() < 0.85:
                key = rng.choice(list(chosen))
                chosen[key] = damage(rng, chosen[key])
            elif chosen and rng.random() < 0.5:
                chosen.pop(rng.choice(list(chosen)), None)
            else:
                chosen[rng.choice(("extra", "__proto__", "points", "task"))] = noise(rng)
        return json.dumps(chosen, ensure_ascii=rng.random() < 0.5)
    keys = list(spec.parameters)
    if roll < 0.60:
        keys.append(rng.choice(("__proto__", "constructor", "extra", "task", "points", "")))
    chosen = {key: noise(rng) for key in keys if rng.random() < 0.8}
    if rng.random() < 0.05 and chosen:
        chosen[rng.choice(list(chosen))] = rng.choice((1, 2.5, None, True, ["a"], {"b": 1}))
    return json.dumps(chosen, ensure_ascii=rng.random() < 0.5)


def finite(value) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, list):
        return all(finite(item) for item in value)
    return True


class TheEnvelopeHoldsForAnyInput(unittest.TestCase):
    def check(self, name: str, arguments: str) -> dict:
        started = time.perf_counter()
        text = engine.call(name, arguments)
        elapsed = time.perf_counter() - started
        self.assertLess(elapsed, 5.0, f"{name} took {elapsed:.1f}s on {arguments[:80]!r}")
        reply = json.loads(text)
        self.assertIsInstance(reply, dict)
        self.assertEqual(reply["agent"], name)
        self.assertIs(type(reply["ok"]), bool)
        if reply["ok"]:
            self.assertEqual(sorted(reply), ["agent", "cite", "elapsed_ms", "ok", "value", "wraps"])
            value = reply["value"]
            self.assertIsInstance(value, dict)
            self.assertTrue(finite(value), f"{name}: a non-finite number in the value")
            self.assertEqual(reply["cite"]["sha256"], core.digest(value))
            self.assertEqual(reply["cite"]["row"], f"{name}-{reply['cite']['sha256'][:16]}")
            self.assertEqual(reply["cite"]["artifact"], core.ARTIFACT)
            self.assertLessEqual(len(core.canonical(value).encode("utf-8")), core.MAX_RESULT_BYTES)
        else:
            self.assertEqual(sorted(reply), ["agent", "kind", "ok", "refusal"])
            self.assertIn(reply["kind"], ("invalid", "refused", "internal", "unknown_agent"))
            self.assertIsInstance(reply["refusal"], str)
            # "internal" means a bug in an agent. Fuzzing is how that gets found.
            self.assertNotEqual(reply["kind"], "internal", f"{name} failed internally on {arguments[:120]!r}: {reply['refusal']}")
        return reply

    def test_every_agent_survives_random_arguments(self) -> None:
        rng = random.Random(SEED)
        for name in sorted(engine.AGENTS):
            spec = engine.AGENTS[name]
            oks = 0
            # run_pipeline runs a real controller each time (a quarter of a second).
            rounds = 40 if name == "run_pipeline" else 400
            for _ in range(rounds):
                oks += self.check(name, arguments_for(rng, spec))["ok"]
            with self.subTest(agent=name):
                self.assertGreater(oks, rounds // 20, f"{name}: the fuzz rarely produced a valid call, so it proved little")

    def test_unknown_names_and_bad_types_are_answered_not_raised(self) -> None:
        rng = random.Random(SEED + 1)
        for _ in range(300):
            name = rng.choice(("", "x", "__class__", "AGENTS", "wilson_interval ", "WILSON_INTERVAL", noise(rng)[:200]))
            reply = json.loads(engine.call(name, arguments_for(rng, engine.AGENTS["wilson_interval"])))
            self.assertFalse(reply["ok"])
            self.assertEqual(reply["kind"], "unknown_agent")
        for bad in (None, 5, [], {}, b"x"):
            reply = json.loads(engine.call(bad, "{}"))
            self.assertEqual((reply["ok"], reply["kind"]), (False, "unknown_agent"))
        for bad in (None, 5, [], {}, b"{}"):
            reply = json.loads(engine.call("wilson_interval", bad))
            self.assertFalse(reply["ok"])

    def test_a_valid_looking_input_is_never_a_non_finite_number(self) -> None:
        # Large and tiny magnitudes are where a float quietly becomes inf or nan.
        for coordinates in ("1e300,1e300,1e300; -1e300,1e300,1e300", "1e6,1e6,1e6; -1e6,-1e6,-1e6", "0,0,0; 0,0,0",
                            "1e-300,0,0; 2e-300,0,0"):
            reply = json.loads(engine.call("geometry_measure", json.dumps({"measure": "distance", "points": coordinates})))
            if reply["ok"]:
                self.assertTrue(math.isfinite(reply["value"]["distance_angstrom"]))
        for successes, trials in (("0", "1"), ("1", "1"), ("1000000000", "1000000000"), ("0", "1000000000"), ("1", "1000000000")):
            reply = json.loads(engine.call("wilson_interval", json.dumps({"successes": successes, "trials": trials})))
            self.assertTrue(reply["ok"], reply)
            self.assertTrue(finite(reply["value"]))
            self.assertLessEqual(reply["value"]["lower"], reply["value"]["proportion"])
            self.assertGreaterEqual(reply["value"]["upper"], reply["value"]["proportion"])


if __name__ == "__main__":
    unittest.main()
