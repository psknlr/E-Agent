# Chat with E-Agent

Open **[E-Agent chat](https://psknlr.github.io/E-Agent/)**.

The default **Browser + API** mode runs a bounded agent loop in your browser.
Your selected model chooses local tools, the browser queries the curated enzyme
reference data or performs arithmetic, and the chat displays the answer,
citations, guard results and tool transcript. Follow-up questions include a
bounded conversation history. A Python backend is optional.

## Run directly in your browser

1. Open the chat page and leave **Browser + API** selected. The page loads its
   local reference tools without contacting a model or requiring a backend.
2. Under **Model settings**, choose MiniMax, OpenAI / GPT, Anthropic / Claude,
   or Custom endpoint.
3. Enter your **API key**, exact **Model name**, and full **API URL**. Preset URLs
   and model names remain editable. Custom endpoints support OpenAI Chat
   Completions or Anthropic Messages.
4. Send a question, for example “Inspect Ssal-KRED activity on 2a” or “Use the
   calculator to evaluate `(2.5 + 3.5) * 4` and show the calculation.”
5. Review the research trace to see the local tools the model actually called.
   Chat readiness is verified only after a successful model response.

The browser sends the key and model requests directly to your chosen API URL.
Keys stay in tab memory and are excluded from saved/exported templates. The
page does not send browser-mode requests to an E-Agent server. Use your own key
and an API endpoint you trust.

The API must allow cross-origin browser requests (CORS) from
`https://psknlr.github.io`. OPTIONS and deliberately-invalid-key POST checks of
the MiniMax preset endpoint allowed that origin, POST, Authorization and
Content-Type during implementation. The POST returned HTTP 401 with the CORS
headers intact; these are transport checks, not authenticated model completions. The app also
implements GPT and Claude request formats, including Claude's browser-access
header. Endpoint, account and network policies can still reject direct access.
The page reports network/CORS failures and never uses an opaque `no-cors`
request or a public proxy. Select an API that allows browser access, or use the
optional backend mode below. Official
[OpenAI browser support documentation](https://developers.openai.com/api/reference/typescript)
and [Anthropic SDK documentation](https://platform.claude.com/docs/en/cli-sdks-libraries/typescript)
describe their browser clients.

Browser tools use a deterministic data bundle exported by the repository's
actual Python reference loaders. Their refused quantities, missing values,
detection limits, source citations and independence groups are preserved.
Arithmetic is performed by a bounded expression parser, not by executing model
code. Model numbers are checked against tool citations. This mode supports
reference queries and arithmetic; structure prediction, docking, pipeline
execution and experimental approvals remain in the CLI.

## Optional: connect a Python backend

GitHub Pages hosts the HTML, CSS and JavaScript. It does not run Python servers
([GitHub Pages documentation](https://docs.github.com/en/pages/getting-started-with-github-pages/what-is-github-pages)).
The repository includes a [Render Blueprint](../render.yaml) to host the Python
agent. Deployment does not require a provider API key: you can enter one in the
chat's **Model settings** for each session.

1. Open **[Deploy the E-Agent backend on Render](https://render.com/deploy?repo=https://github.com/psknlr/E-Agent)**
   and connect this repository. The Blueprint selects the free web-service plan;
   review the displayed plan when creating it.
2. Wait for deployment, then copy the service's public HTTPS URL. In Render's
   environment settings, copy the generated `EAGENT_CHAT_TOKEN`.
3. On the GitHub Pages site, select **Python backend**. Enter that **Backend URL**
   and **Backend access token** under Agent connection, then connect.
4. Under **Model settings**, select MiniMax, OpenAI / GPT, Anthropic / Claude, or
   Custom endpoint. Enter your **API key**, exact **Model name**, and full
   **API URL**. Every model name and URL is editable. Custom endpoint also lets
   you select OpenAI Chat Completions or Anthropic Messages as the API format.
5. Send a question. A connected backend has not yet verified these credentials
   or this model. The status changes after your selected model actually answers.

The built-in endpoint presets are:

| Provider | API URL | API format |
| --- | --- | --- |
| MiniMax | `https://api.minimax.io/v1/chat/completions` | OpenAI Chat Completions |
| OpenAI / GPT | `https://api.openai.com/v1/chat/completions` | OpenAI Chat Completions |
| Anthropic / Claude | `https://api.anthropic.com/v1/messages` | Anthropic Messages |
| Custom endpoint | Your full endpoint URL | Either supported format |

Use a model name available to your account and endpoint. See the official
[MiniMax](https://platform.minimax.io/docs/api-reference/text-chat-openai),
[OpenAI](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create),
and [Anthropic](https://platform.claude.com/docs/en/api/messages/create) references
for the corresponding formats. Other providers work when their endpoint accepts
one of these formats. Hosted backends require public HTTPS API endpoints; a
backend running on your own computer can also call a loopback local API.

**Reusable templates** can be saved in your browser, exported as JSON, and
imported later. They contain the execution mode, provider, API format, model name
and API URL. Version-one templates remain importable in Python backend mode.
API keys and backend tokens are excluded; enter the API key again after loading
a template or reloading the page. Imported files containing secret fields are
rejected.

The access token and provider API key are retained only in page memory and must
be re-entered after reloading. The backend URL and non-secret templates may be
saved in the browser. Each chat request sends your provider key to your chosen
E-Agent backend in Python backend mode, which uses it to call your API URL for
that run. Browser mode sends it directly to the API. Use a backend
and API endpoint you trust. The backend does not save the request's key or
conversation. Your question, history and tool results are sent to your selected
model provider. Download the run's
transcript from the chat if you want to keep the tool evidence.

To prefill the backend URL for all visitors, set the repository **Actions
variable** `EAGENT_BACKEND_URL` to the HTTPS service URL, then rerun the
`Deploy E-Agent chat to GitHub Pages` workflow. Only this public URL enters the
static build. Do not put secrets in a GitHub variable or `web/config.json`.

Render's [Blueprint reference](https://render.com/docs/blueprint-spec) explains
the generated access token and secret input. Free instances may sleep when idle;
if the first connection takes time, wait for the backend to wake and retry.

## Run on your own computer

From a checkout of this repository:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
export EAGENT_PROVIDER=minimax
export EAGENT_MODEL=MiniMax-M2.7
python -m eagent.web --host 127.0.0.1 --port 8787
```

Open **http://127.0.0.1:8787/** to use the same chat interface. A local loopback
server can run without an access token. Select Python backend mode and enter the
provider key, model name and API URL in Model settings. The same locally served
page also supports Browser + API mode. For hosted use, the server requires
`EAGENT_CHAT_TOKEN` and an HTTPS host or reverse proxy. The allowed browser
origin defaults to `https://psknlr.github.io`; override
`EAGENT_ALLOWED_ORIGINS` with a comma-separated list when hosting a different
frontend. Add only origins you operate.

Alternatively, select **Server default** to use credentials configured on the
backend. Set `EAGENT_PROVIDER`, `EAGENT_MODEL`, and the matching environment
variable (`MINIMAX_API_KEY`, `OPENAI_API_KEY`, or `ANTHROPIC_API_KEY`); provider
base URL overrides are described in [MODELS.md](MODELS.md). Request settings
never replace these server defaults or borrow their keys.

## Host it on your own domain

The `web/` directory is a self-contained static site, so it can be served from
any host and any domain, not only this project's GitHub Pages. To publish it at
a custom domain such as `enzyme.impf.ai`, build a portable copy with
`python scripts/export_site.py --out dist/site` (add `--cname <domain>` for
GitHub/Cloudflare Pages, or `--backend-url <https url>` to pin a Python
backend), upload the result, and add one DNS record. The exact per-host and DNS
steps are in [`DEPLOY_ENZYME_IMPF.md`](DEPLOY_ENZYME_IMPF.md). Browser + API
mode needs no backend; provider CORS still applies from the new origin.

## Status comes from the active runtime

- **Browser tools loaded:** local tools have loaded; complete your model settings
  before sending a question.
- **Not connected:** Python backend mode has no connected backend.
- **Backend unavailable:** a connection or request failed.
- **Needs configuration:** the backend responds, but the selected model settings
  or reference loaders are not ready.
- **Ready to verify:** the actual agent and tools are available and the selected
  settings are complete; send a question to verify the API credentials and model.
- **Model request failed:** the API or backend responded, but the model request failed;
  check Model settings and retry.
- **Chat ready:** the active configuration has completed a successful model request.

For Python backend mode, `GET /api/health` reports configuration, actual registered tools and
`runtime_ready` and `completion_verified`. The default completion flag concerns
only Server default; a DIY configuration is verified by its own chat response.
Health checks do not spend tokens to check a model. A provider
failure, citation refusal or loop limit is visible in the run transcript; a
partial answer is not silently presented as a completed run.

## Verify

```sh
python -m pip install -e '.[dev]'
python -m pytest tests/test_web.py tests/test_request_client.py tests/test_providers.py tests/test_toolloop.py tests/test_browser_bundle.py -q
for script in web/*.js; do node --check "$script"; done
python scripts/build_browser_bundle.py --check
node --test tests/browser-*.test.cjs
```

Automated provider tests simulate HTTP completions. A live model completion
requires a valid key and reachable API endpoint; passing these tests alone does
not prove that your account can use the selected model.
