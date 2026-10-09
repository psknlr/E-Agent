# Chat with E-Agent

Open **[E-Agent chat](https://psknlr.github.io/E-Agent/)**.

The page connects to the actual Python `ToolLoop`, the same runtime used by
`eagent model ask`. MiniMax chooses tools, the repository's loaders execute
them over the curated enzyme reference set, and the chat displays the answer,
citations, guard results and tool transcript. Follow-up questions include a
bounded conversation history. This console reads evidence; pipeline execution,
structure prediction, docking and experimental approvals remain in the CLI.

## Activate the MiniMax backend

GitHub Pages hosts the HTML, CSS and JavaScript. It does not run Python servers
([GitHub Pages documentation](https://docs.github.com/en/pages/getting-started-with-github-pages/what-is-github-pages)).
The repository includes a [Render Blueprint](../render.yaml) to host the Python
agent with MiniMax, with no provider key in the public page.

1. Open **[Deploy the E-Agent backend on Render](https://render.com/deploy?repo=https://github.com/psknlr/E-Agent)**
   and connect this repository. The Blueprint selects the free web-service plan;
   review the displayed plan when creating it.
2. Enter your `MINIMAX_API_KEY` in Render's secret environment-variable field.
   The Blueprint names `MiniMax-M2.7` explicitly. You can change `EAGENT_MODEL`
   to a model available to your MiniMax account. The default endpoint is
   `https://api.minimax.io/v1/chat/completions`, following
   [MiniMax's current API reference](https://platform.minimax.io/docs/api-reference/text-chat-openai).
   Set `EAGENT_MINIMAX_BASE_URL` if your account uses a different regional base URL.
3. Wait for deployment, then copy the service's public HTTPS URL. In Render's
   environment settings, copy the generated `EAGENT_CHAT_TOKEN`.
4. On the GitHub Pages site, open connection settings. Enter that backend URL
   and the **backend access token**. The MiniMax API key belongs only in Render.
5. Connect and send a question. A configured backend has not yet verified a
   model completion. The status changes after the model actually answers.

The access token is retained only in page memory and must be re-entered after
reloading. The backend URL may be saved in the browser. Chat history is kept in
page memory; the backend does not save conversations. The question, history and
tool results are sent to MiniMax when you send a message. Download the run's
transcript from the chat if you want to keep the tool evidence.

To prefill the backend URL for all visitors, set the repository **Actions
variable** `EAGENT_BACKEND_URL` to the HTTPS service URL, then rerun the
`Deploy E-Agent chat to GitHub Pages` workflow. Only this public URL enters the
static build. Do not put either secret in a GitHub variable or `web/config.json`.

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
read -r -s -p 'MiniMax API key: ' MINIMAX_API_KEY; printf '\n'
export MINIMAX_API_KEY
python -m eagent.web --host 127.0.0.1 --port 8787
```

The secret-input command above uses Bash. In Zsh, use
`read -rs 'MINIMAX_API_KEY?MiniMax API key: '` followed by
`export MINIMAX_API_KEY`. Both keep the value out of shell history.

Open **http://127.0.0.1:8787/** to use the same chat interface. A local loopback
server can run without an access token. For hosted use, the server requires
`EAGENT_CHAT_TOKEN` and an HTTPS host or reverse proxy. The allowed browser
origin defaults to `https://psknlr.github.io`; override
`EAGENT_ALLOWED_ORIGINS` with a comma-separated list when hosting a different
frontend. Add only origins you operate.

## Status comes from the running agent

- **Not connected:** no backend URL has been connected.
- **Backend unavailable:** a connection or request failed.
- **Needs configuration:** the backend responds, but its provider, model, key or
  reference loaders are not ready.
- **Backend configured:** the actual agent and tools are available; a live model
  completion has not yet been established by this backend process.
- **Chat ready:** this backend has completed a successful model request.

`GET /api/health` reports configuration, actual registered tools and
`completion_verified`. It does not spend tokens to check a model. A provider
failure, citation refusal or loop limit is visible in the run transcript; a
partial answer is not silently presented as a completed run.

## Verify

```sh
python -m pip install -e '.[dev]'
python -m pytest tests/test_web.py tests/test_providers.py tests/test_toolloop.py -q
node --check web/app.js
```

Automated provider tests simulate HTTP completions. A live MiniMax completion
requires your configured backend and valid key; passing these tests alone does
not prove that your account can use the selected model.
