# The compute engine: agents that run on your device

The chat page has two halves with one job each.

* **The controller is your model API.** It reads the question, decides which
  agents to call and with what, and reads their results. It computes nothing,
  approves nothing and cannot add an agent.
* **The agents run in your browser.** Evidence lookups run as JavaScript.
  Scientific computation runs as the project's own Python, compiled to
  WebAssembly, on your device's CPU, in a worker thread.

Nothing about a computation is delegated to the API. The model names an agent
and passes strings; the page checks them, runs one fixed function, and hands the
result back with a citation the numeric guard can verify.

```
 you ──question──▶ page ──▶ your model API (controller) ──"call align_sequences(…)"──┐
                    ▲                                                                │
                    │                                                                ▼
              answer with             result + citation  ◀── agents, on this device:
              checked numbers                                  JavaScript  evidence lookups, calculator
                                                               WebAssembly the project's Python, on the CPU
```

## What runs, and what does not

Eight compute agents, each a thin bounded wrapper over code the project already
has and already tests. They are generated into
[`web/agent-catalog.json`](../web/agent-catalog.json) from the code, and CI
refuses a stale copy.

| Agent | Runs | Does |
| --- | --- | --- |
| `align_sequences` | `eagent.science.numbering.needleman_wunsch`, `alignment_score` | Global alignment with affine gaps: identity over aligned columns, coverage, gaps, score |
| `sequence_distance` | `eagent.science.diversity.sequence_distance` | k-mer Jaccard or Hamming distance and similarity |
| `wilson_interval` | `eagent.science.robustness.wilson_interval` | Wilson score interval for a count out of trials |
| `geometry_measure` | `eagent.science.geometry` | Distance, angle, torsion or centroid from explicit coordinates |
| `variant_check` | `eagent.schemas.variant.Mutation` | Is the wild-type residue really at that position of the sequence you gave |
| `task_gate_report` | `TaskSpec.unresolved_for`, the interface registry | Which fields of a task each human decision point still lacks. Read-only |
| `pipeline_catalog` | the interface registry and the roadmap's status table | Which of the ten pipeline interfaces can run in a browser |
| `run_pipeline` | `ResearchController` and the interfaces it runs | Runs the project's own deterministic controller on a task, here, with the network off and no approvals: the reaction step runs for real, the run stops at the first human decision point, and the report says what each step found and what it waits on |

Where a result holds a fraction it also holds the same value as a percentage
(`identity_percent_over_aligned_columns`, `lower_percent`, …), because that is
how it is written in prose and a quoted "95%" must be citable against a field
that says 95.

**The pipeline's ten interfaces are not all runnable here, and the page says so
from the project's own table rather than from a claim.**

| Interface | In a browser | Needs |
| --- | --- | --- |
| `normalize_reaction` | runs | |
| `retrieve_evidence` | partly | network access to public databases (reads cached evidence only) |
| `mine_sequences` | needs a program | `blastp`, `mmseqs2` or `hmmsearch` |
| `annotate_family` | runs | seed accessions that sequence mining supplies |
| `prepare_structures` | needs a program | a structure predictor and a local structure index |
| `model_complexes` | needs a program | a docking program or a joint complex predictor |
| `evaluate_catalysis` | runs | poses it is given |
| `select_batch` | runs | |
| `ingest_results` | runs | |
| `propose_mutations` | partly | LigandMPNN for the inverse-folding path |

An interface that needs a program reports `tool_unavailable` and produces
nothing. It never substitutes a heuristic that imitates the program, in a
browser or anywhere else. That is the project's fifth rule.

**`run_pipeline` is the controller, not a model of it.** It builds the same
`ResearchController` the command line builds, in a throwaway directory, with the
network off, the template library the project ships, and an approval queue that
starts empty, and lets it run. On the pilot task that means `normalize_reaction`
executes for real (four blocking problems, four open questions) and the run
stops, `awaiting_human`, at `reaction_spec_confirmed`. A task whose reaction is
complete gets a normalised reaction spec and the same stop: the request for a
person to confirm it. In WebAssembly the report is byte-identical to the native
one (the parity test compares their hashes). If a model writes the task file
from your description, the structures in it are the model's reading of what you
said, not yours; the confirmation gate is where a person checks them, and the
controller's instructions to the model say so. Steps behind a closed gate do not
run, here or on the command line, until a person clears it with
`eagent approve`; that is why the five interfaces the table marks "runs" are
shown as able to run rather than as having run.

**Approvals are not an agent and cannot become one.** The three decision points
(`reaction_spec_confirmed`, `synthesis_authorized`,
`functional_criteria_confirmed`) are cleared only by a named person through
`eagent approve`. The controller *requests* an approval and waits;
`task_gate_report` reads what is still open. A test over the engine's syntax
trees pins that no engine module calls what grants or denies a request, or
resolves a task field, or records a decision, and a flag in a task file is not
an approval (a test runs a task whose file says every gate is confirmed and
watches the controller stop at the first anyway).

### CPU, GPU and NPU

The agents run on the **CPU**. That is a statement about the code, not a
limitation of the page: none of the agents has a GPU or NPU kernel, and the
largest input any agent accepts (two 1,200-residue sequences) aligns in about
1.3 s in the headless Chromium this was tested in (a 4-thread cloud sandbox). The **This device** panel lists what the browser
reports (CPU threads, memory, WebAssembly, WebGPU, WebNN) and can ask for a GPU
adapter, so you can see what the device offers; it does not claim that anything
uses it. If a GPU kernel is added, it belongs in the same registry behind the
same contract, with the same parity test against the native answer.

## How a computed number becomes citable

The existing numeric guard already requires every number with a unit in a
model's answer to carry `[cite artifact=… sha256=… row=… field=… method=read]`
and checks the number against the field it cites. Compute agents plug into the
same mechanism.

1. The engine returns `{"ok": true, "value": …, "cite": {"artifact":
   "engine_compute", "sha256": …, "row": …}}`. The digest is over the canonical
   JSON of the value; the row id is `<agent>-<first 16 hex of the digest>`.
2. The page checks that envelope, then registers the row with the guard under
   `engine_compute`. A row is registered **only after the engine produced it**,
   it is immutable, and it belongs to the run that computed it.
3. The model quotes a number with a citation to the field it came from
   (`identity_percent_over_aligned_columns`, `gates.0.n_unresolved`; nested
   fields use dotted paths because the citation grammar ends at `]`).
4. The guard compares the quoted number with that field, rounded to the digits
   quoted. A wrong number, a wrong field, a field that is null, a fraction cited
   for a percentage, a row that was never computed, or a real row with a forged
   digest all fail, and the run stops with the guard's refusal.

A refusal from an agent ("the sequences are different lengths, so a Hamming
distance is undefined") carries no value and no citation, so there is nothing a
model could quote.

## Security model

The model can only **name an agent and pass strings**.

* The agent must exist in a fixed table. An unknown name is refused, never
  guessed. Arguments must be a JSON object of strings, every key declared, every
  value within the agent's own length bound, checked in JavaScript before the
  call and again in Python.
* Numbers are strings here on purpose; each agent parses its own with a strict
  ASCII pattern, so `float("1_0")`, `nan` and the Arabic-Indic digit `٣` are
  refused rather than accepted.
* Results must be finite and at most 7,000 bytes; an oversized result is refused
  rather than clipped, because a clipped result is a number the model never saw.
* The engine package imports nothing that opens a process or a socket, nothing
  that evaluates code, and not the bridge back into the page (`js`, `pyodide`).
  A test over the modules' syntax trees keeps it so. The only file it reads is a
  shipped task, by id, from a fixed directory; the only directory it creates is
  the throwaway one `run_pipeline` works in, removed when the run ends (a test
  checks that, including when the controller fails).
* YAML in a task is parsed with the safe loader after anchors and aliases are
  refused, so a small document cannot expand into a large one.
* The worker has no DOM. In WebAssembly there is no process to start and no raw
  socket; the one thing the worker fetches is this site's own engine files.
* Your API key is never sent to the engine. The model API receives your
  question and the agents' results, as it must to answer; the engine itself
  sends nothing anywhere.

## Integrity: what you run is what was built

Nothing binary is in git. [`scripts/build_engine.py`](../scripts/build_engine.py)
assembles `web/engine/` at build time (CI does it on every deploy) from pinned
downloads, and refuses any whose SHA-256 differs from
[`scripts/engine.lock.json`](../scripts/engine.lock.json):

| Part | Source | Pin |
| --- | --- | --- |
| Pyodide 314.0.7 (CPython 3.14.2 as WebAssembly) | npm | version and tarball SHA-256 |
| `pydantic`, `pydantic-core` (its WebAssembly wheel), `typing-extensions`, `annotated-types`, `typing-inspection` | PyPI | exact file and SHA-256 |
| PyYAML 6.0.3 (pure-Python package only) | PyPI sdist | SHA-256 |
| the E-Agent package and shipped task files | this repository | hashed into the manifest |

The result is `web/engine/manifest.json`: a SHA-256 for every file, how to
install them, and a **self-test**: four agent calls whose answers the build
computed natively. In the visitor's browser:

* the browser checks the runtime's own downloads (the WebAssembly binary, the
  standard library) against the manifest with subresource integrity, and the
  installer checks every package and the project against it before installing;
* the engine is not "ready" until the four self-test calls return exactly what
  the build computed natively, so a runtime that does the sums differently is
  reported, not used;
* the page then checks that the engine holds exactly the agents this page was
  published with (`web/agent-catalog.json`), and offers nothing if it does not.

All zips are stored, with sorted entries and fixed timestamps, so the same
inputs give the same bytes on any machine and `--check` means something.

The two small `.mjs` loader files are loaded as module scripts, which a browser
cannot integrity-check from `import()`; they come from the same origin as the
page's own scripts.

Upgrading anything is a change to `PINS` in the build script plus
`python scripts/build_engine.py --update-lock`, which is a diff a reviewer can
read; then build, and run the tests below.

## Using it

The **Compute engine** card in the sidebar finds the engine your site was
published with, and shows what it will cost (about 20 MB once; your browser
keeps the files and re-checks them). **Start on this device** loads it. Nothing
is downloaded until you do, unless you tick *Start automatically on this
browser*. Once it reports **RUNNING**:

* **Send** offers the compute agents to your model alongside the evidence
  tools (untick *Offer compute agents to the model* to withhold them).
* **Run in browser** routes your question to agents by keyword and shows their
  results with no model and no network. It understands two sequences in capital
  one-letter codes with *align* or *identity*; *hamming* or *k-mer distance*;
  substitutions such as `K2A` with a sequence; `7 of 20` with *interval* or
  *poses*; coordinates `x,y,z` with *distance*, *angle*, *dihedral* or
  *centroid*; a shipped task id with *gates* or *unresolved*; *run* with
  *pipeline*, *controller* or a task id (“run KRED_PILOT_001 through the
  pipeline”); and *pipeline* for the capability map.
* The research trace marks every result with where it ran (`WASM · CPU` or
  `JS`) and which project function produced it.

If the site was published without the engine, if the browser has no
WebAssembly or workers, or if the engine does not match the page, the card says
so and everything else keeps working. A call that runs longer than 60 s stops
the worker (agents are stateless, so nothing is lost) and the next use starts a
fresh one.

Measured here, in headless Chromium on a 4-thread sandbox: the engine starts in
about 6.5 s (runtime 3.8 s, packages 0.6 s, project 2.4 s) and holds about
160 MB; a 1,200 × 1,200 alignment takes about 1.3 s, against 0.75 s for native
CPython; the first `run_pipeline` takes about 1.1 s (it loads the template
library) and later ones about 0.2 s. Phones will be slower. Verified in Chromium; Firefox and Safari have not been
tested here.

## Limits

* Sequences up to 1,200 residues, up to 40 variants per check, text arguments up
  to 4,000 characters (a task file up to 6,000).
* Reference data and the pipeline's external programs are not in the engine, and
  `run_pipeline` cannot get past a closed decision point: approving is a person's
  act, done on the command line.
* The engine needs a browser with WebAssembly and module Web Workers, and a
  host that serves `.mjs` as JavaScript and `.wasm` as `application/wasm`. The
  card names the problem if the host does not. nginx:
  `types { text/javascript mjs; application/wasm wasm; }`; Apache:
  `AddType text/javascript .mjs` and `AddType application/wasm .wasm`;
  Netlify and Cloudflare Pages read the `_headers` file the exporter writes.

## Adding an agent

1. Write it in `src/eagent/engine/agents.py` (or `pipeline.py`) as a function of
   the string arguments returning a dict, wrapping project code that already has
   its own tests, and `register` an `AgentSpec`. Parse every argument with the
   strict parsers in `core.py`. Put numbers a model might quote at the top level
   (or under dotted paths), with a `_percent` twin for any fraction.
2. Pin it in `tests/test_engine.py`: the set of agents, its results against the
   project function, every refusal, and that it passes the reach test.
3. Document it here; a test fails if an agent is not named in this file.
4. `python scripts/build_browser_bundle.py` (regenerates the catalog) and
   `python scripts/build_engine.py` (rebuilds the project zip).
5. `node --test tests/engine-*.test.cjs` compares your agent's answers in
   WebAssembly with native Python.

## Tests

```sh
python -m pytest tests/test_engine.py tests/test_build_engine.py tests/test_export_site.py -q
python scripts/build_browser_bundle.py --check
node --test tests/browser-*.test.cjs                  # registry, guard, panels, page wiring
python scripts/build_engine.py && python scripts/build_engine.py --check
EAGENT_REQUIRE_ENGINE=1 node --test tests/engine-*.test.cjs   # the real WebAssembly runtime
```

`tests/engine-wasm.test.cjs` runs forty calls natively and in WebAssembly
and compares them; `tests/engine-agents.test.cjs` drives the whole page stack
(registry, citations, numeric guard) against the real engine with a scripted
model. Without a build, the tests that need one are skipped; CI sets
`EAGENT_REQUIRE_ENGINE=1` so a missing build fails instead.
