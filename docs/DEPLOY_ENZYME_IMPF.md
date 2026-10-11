# Serve E-Agent at `enzyme.impf.ai`

The application is a static site: HTML, CSS, JavaScript, a pre-built evidence
bundle and, for the compute agents, a pinned build of Python as WebAssembly
(about 20 MB, see [`ENGINE.md`](ENGINE.md)). No Python process runs on the server.
**Browser + API mode needs no backend at all.**

There are two ways to make `enzyme.impf.ai` work. Pick one.

| | What `enzyme.impf.ai` does | What you host | Address bar |
| --- | --- | --- | --- |
| **1. Redirect** (simplest) | sends every visitor to the GitHub Pages site, which already carries the whole application | a few bytes of redirect config | changes to `psknlr.github.io/E-Agent/` |
| **2. Your own copy** | serves the application itself | the exported site (about 20 MB with the engine) | stays `enzyme.impf.ai` |

Two parts are outside this repository and only you (an owner of the `impf.ai`
domain and of the impf host) can do them; they are flagged **[you]** below:

- **[you]** add one DNS record on `impf.ai`, and
- **[you]** put the output on the impf host and enable HTTPS.

Everything else is a command in this repo.

---

## Option 1 — redirect to the GitHub Pages site

The Pages site (`https://psknlr.github.io/E-Agent/`) is built from `main` by
`.github/workflows/pages.yml`, which also builds and tests the compute engine, so
once a change is on `main` the Pages site carries the whole application. A
domain that only needs to lead there does not need a copy of it. (Until this
change is merged, the Pages site is still the earlier version.)

```bash
python scripts/export_site.py --redirect-to https://psknlr.github.io/E-Agent/ --out dist/enzyme-redirect
# add --cname enzyme.impf.ai if the redirect itself is served from GitHub Pages
```

This writes no application files, only what a host needs to send visitors on:

| File | Use it on |
| --- | --- |
| `index.html`, `404.html` | any host that only serves files (S3, GitHub Pages): a page that redirects at once and says where it went |
| `_redirects` | Netlify and Cloudflare Pages: a real HTTP 301 for every path |
| `vercel.json` | Vercel: the same, as a permanent redirect |
| `.htaccess` | Apache (`mod_rewrite`) |
| `nginx-redirect.conf.example` | nginx: one `return 301` line for the server block |

Use the one your host understands and ignore the rest. A host that can answer
with a real 301 should; the HTML page is the fallback for hosts that cannot.
The target must be a plain HTTPS URL (the exporter refuses anything else), and
because visitors end up on `psknlr.github.io`, your model provider must allow
browser requests from that origin (see "Provider CORS" below).

Then add the DNS record for the host you chose (step 3 below).

If you would rather keep `enzyme.impf.ai` in the address bar without hosting a
copy, the Pages site itself can take a custom domain (the repository's
**Settings → Pages → Custom domain**, plus a `CNAME` record pointing at
`psknlr.github.io`). That moves where the main site lives, so it is the
repository owner's call and is not done by anything here.

---

## Option 2 — host your own copy

### 1. Build the portable bundle

From a checkout of this repository:

```bash
python scripts/build_engine.py        # the compute engine; needs network once, then cached

# Browser + API mode (recommended; no backend needed)
python scripts/export_site.py --out dist/enzyme-impf

# …or, if the impf host is GitHub Pages or Cloudflare Pages and wants the
# custom domain written into the build:
python scripts/export_site.py --out dist/enzyme-impf --cname enzyme.impf.ai

# …or, to also pin an optional Python backend URL into the build:
python scripts/export_site.py --out dist/enzyme-impf --backend-url https://eagent-api.impf.ai

# …or without the 20 MB engine (the page then says the engine is not published
# and keeps working with the evidence tools and calculator):
python scripts/export_site.py --out dist/enzyme-impf --without-engine
```

The script copies the verified static files, writes `config.json`, optionally
writes `CNAME`, checks that every asset and JSON file resolves, and **checks the
copied engine against its build manifest** (every file's SHA-256, and that it was
built from this checkout). The result in `dist/enzyme-impf/` is the exact set of
files to publish. It also writes a `_headers` file (read by Netlify and
Cloudflare Pages; harmless elsewhere).

> Rebuilding the science bundle is a separate step and is only needed if the
> reference data or the agents changed: `python scripts/build_browser_bundle.py`
> (CI runs `--check`).

### 2. Publish to the impf host — pick one

#### A. Static host (nginx, S3 + CloudFront, Apache, any CDN) — **[you]**
Upload the **contents** of `dist/enzyme-impf/` to the site's web root so that
`index.html` is served at `/`. No server-side code runs. Serve `.json` as
`application/json` and `.js` as `text/javascript` (most hosts do by default),
**and the engine's two loader files, `.mjs`, as JavaScript and `.wasm` as
`application/wasm`.** Browsers refuse a module script served as anything else;
the page's Compute engine card says so if yours does. For nginx add
`types { text/javascript mjs; application/wasm wasm; }` (inside `http` or
`server`); for Apache `AddType text/javascript .mjs` and
`AddType application/wasm .wasm`.

#### B. GitHub Pages on an impf-owned repo — **[you]**
1. Create/choose a repo under the impf org (e.g. `impf/enzyme`), and copy the
   `dist/enzyme-impf/` files into it (the `CNAME` file is present when you
   exported with `--cname enzyme.impf.ai`).
2. Push, then in that repo: **Settings → Pages → Build and deployment → Deploy
   from a branch** (or a Pages Action), selecting the branch/folder that holds
   `index.html`.
3. **Settings → Pages → Custom domain** → enter `enzyme.impf.ai` → **Save**,
   then tick **Enforce HTTPS** once the certificate is issued.

#### C. Cloudflare Pages / Netlify / Vercel — **[you]**
Create a project from the impf repo (or drag-and-drop `dist/enzyme-impf/`).
Build command: none. Output/publish directory: the folder with `index.html`.
Add `enzyme.impf.ai` as a custom domain in the project's domain settings.
Cloudflare Pages and Netlify pick up the `_headers` file for the engine's types;
on Vercel set the same two types in `vercel.json`.

> This GitHub repository (`psknlr/E-Agent`) and its Pages deploy are **not**
> changed by this guide; no `CNAME` is added here.

---

## 3. Point the DNS at the host — **[you]**

`enzyme` is a subdomain, so a `CNAME` record is correct (no apex-domain
limitations apply). In the `impf.ai` DNS zone, add **one** record whose target
is given by the host you picked:

| Host | Record | Name | Target |
| --- | --- | --- | --- |
| GitHub Pages | CNAME | `enzyme` | `<impf-org>.github.io` |
| Cloudflare Pages | CNAME | `enzyme` | `<project>.pages.dev` |
| Netlify | CNAME | `enzyme` | `<site>.netlify.app` |
| Vercel | CNAME | `enzyme` | `cname.vercel-dns.com` |
| Your own nginx/CDN | CNAME (or A/AAAA) | `enzyme` | the host's name (or its IP) |

Use the exact target the host shows in its custom-domain screen. TLS
certificates are issued automatically by every host above once the record
resolves; this can take a few minutes to a few hours. Then enforce HTTPS.

---

## 4. If you also run the optional Python backend

Browser + API mode does not need this. Only if you want **Python backend**
mode reachable from `enzyme.impf.ai`:

- Deploy the backend (see [`WEB_CHAT.md`](WEB_CHAT.md) and
  [`../render.yaml`](../render.yaml)) at an HTTPS URL, e.g.
  `https://eagent-api.impf.ai`.
- Allow the new frontend origin on the backend:
  `EAGENT_ALLOWED_ORIGINS=https://enzyme.impf.ai` (comma-separate to add more).
  Add only origins you operate.
- Pin the backend URL into the build with
  `--backend-url https://eagent-api.impf.ai` (Option 2, step 1), or leave it
  blank and let users enter it under **Agent connection**.

---

## 5. Verify

Open `https://enzyme.impf.ai` (or wherever it leads) and confirm:

1. The status reads **Browser tools loaded** and the **Agents** panel lists the
   evidence tools, the calculator and the compute agents. This needs no key and
   proves the static bundle was served correctly.
2. The **Compute engine** card reads **OFF** with its download size (or **NOT
   PUBLISHED** if you exported with `--without-engine`). Press **Start on this
   device**: it should reach **RUNNING** in a few seconds, with *Integrity: every
   file checked against its SHA-256* and *Self-test: 4 calls match the native
   build*. If it says the host serves `.mjs` as something else, fix the type as
   in step 2A.
3. Under **Model**, enter your API key, exact model name and full API URL, then
   send “Align MKTAYIAKQRQISFVKSHFSRQ with MKTAYLAKQRQVSFVKSHFSRQ and report the
   identity.” A first successful reply flips the status to **Chat ready** and
   shows the research trace with a `WASM · CPU` result.

**Provider CORS still applies.** Browser + API mode calls your model endpoint
directly from the page, so the endpoint must allow cross-origin requests from the
origin the page is finally served from (`https://enzyme.impf.ai` for Option 2,
`https://psknlr.github.io` for Option 1). Anthropic's browser-access header is
sent automatically; some providers and gateways reject browser origins — if so,
use an endpoint that permits them, or run Python backend mode (step 4). This is a
provider transport policy, not an E-Agent setting.

---

## What this guide does not do

- It does not change `impf.ai` DNS or touch the impf host — those are the
  **[you]** steps above and require access this repository does not have.
- It does not move the primary GitHub Pages site; `psknlr.github.io/E-Agent/`
  keeps working unchanged.
