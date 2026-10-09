# Chat with E-Agent

Open **[E-Agent chat](https://psknlr.github.io/E-Agent/)**.

The page connects to the actual Python `ToolLoop`, the same runtime used by
`eagent model ask`. Your selected model chooses tools, the repository's loaders execute
them over the curated enzyme reference set, and the chat displays the answer,
citations, guard results and tool transcript. Follow-up questions include a
bounded conversation history. This console reads evidence; pipeline execution,
structure prediction, docking and experimental approvals remain in the CLI.

## Connect a backend and choose your model

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
3. On the GitHub Pages site, enter that **Backend URL** and **Backend access
   token** under Agent connection, then connect.
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
imported later. They contain the provider, API format, model name and API URL.
API keys and backend tokens are excluded; enter the API key again after loading
a template or reloading the page. Imported files containing secret fields are
rejected.

The access token and provider API key are retained only in page memory and must
be re-entered after reloading. The backend URL and non-secret templates may be
saved in the browser. Each chat request sends your provider key to your chosen
E-Agent backend, which uses it to call your API URL for that run. Use a backend
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
server can run without an access token. Enter the provider key, model name and
API URL in Model settings. For hosted use, the server requires
`EAGENT_CHAT_TOKEN` and an HTTPS host or reverse proxy. The allowed browser
origin defaults to `https://psknlr.github.io`; override
`EAGENT_ALLOWED_ORIGINS` with a comma-separated list when hosting a different
frontend. Add only origins you operate.

Alternatively, select **Server default** to use credentials configured on the
backend. Set `EAGENT_PROVIDER`, `EAGENT_MODEL`, and the matching environment
variable (`MINIMAX_API_KEY`, `OPENAI_API_KEY`, or `ANTHROPIC_API_KEY`); provider
base URL overrides are described in [MODELS.md](MODELS.md). Request settings
never replace these server defaults or borrow their keys.

## Status comes from the running agent

- **Not connected:** no backend URL has been connected.
- **Backend unavailable:** a connection or request failed.
- **Needs configuration:** the backend responds, but the selected model settings
  or reference loaders are not ready.
- **Ready to verify:** the actual agent and tools are available and the selected
  settings are complete; send a question to verify the API credentials and model.
- **Model request failed:** the backend responded, but the model request failed;
  check Model settings and retry.
- **Chat ready:** the active configuration has completed a successful model request.

`GET /api/health` reports configuration, actual registered tools and
`runtime_ready` and `completion_verified`. The default completion flag concerns
only Server default; a DIY configuration is verified by its own chat response.
Health checks do not spend tokens to check a model. A provider
failure, citation refusal or loop limit is visible in the run transcript; a
partial answer is not silently presented as a completed run.

## Verify

```sh
python -m pip install -e '.[dev]'
python -m pytest tests/test_web.py tests/test_request_client.py tests/test_providers.py tests/test_toolloop.py -q
node --check web/app.js
```

Automated provider tests simulate HTTP completions. A live model completion
requires a running backend and valid key; passing these tests alone does
not prove that your account can use the selected model.
