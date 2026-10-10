# Serve E-Agent at `enzyme.impf.ai`

This guide publishes the E-Agent chat on an **impf-owned host** (a separate
repository, static host, or server) and points `enzyme.impf.ai` at it. The
`web/` directory is a self-contained static site — HTML, CSS, JavaScript and a
pre-built evidence bundle — so it runs on any static host with no Python
process. **Browser + API mode needs no backend at all.**

Two parts are outside this repository and only you (an owner of the `impf.ai`
domain and of the impf host) can do them; they are flagged **[you]** below:

- **[you]** add one DNS record on `impf.ai`, and
- **[you]** upload the files to the impf host and enable HTTPS.

Everything else is a command in this repo.

---

## 1. Build the portable bundle

From a checkout of this repository:

```bash
# Browser + API mode (recommended; no backend needed)
python scripts/export_site.py --out dist/enzyme-impf

# …or, if the impf host is GitHub Pages or Cloudflare Pages and wants the
# custom domain written into the build:
python scripts/export_site.py --out dist/enzyme-impf --cname enzyme.impf.ai

# …or, to also pin an optional Python backend URL into the build:
python scripts/export_site.py --out dist/enzyme-impf --backend-url https://eagent-api.impf.ai
```

The script copies the verified static files, writes `config.json`, optionally
writes `CNAME`, and checks every asset and JSON file resolves. The result in
`dist/enzyme-impf/` is the exact set of files to publish.

> Rebuilding the science bundle is a separate step and is only needed if the
> reference data changed:
> `python scripts/build_browser_bundle.py` (CI runs `--check`).

---

## 2. Publish to the impf host — pick one

### A. Static host (nginx, S3 + CloudFront, Apache, any CDN) — **[you]**
Upload the **contents** of `dist/enzyme-impf/` to the site's web root so that
`index.html` is served at `/`. Serve `.json` as `application/json` and `.js` as
`text/javascript` (most hosts do by default). No server-side code runs.

### B. GitHub Pages on an impf-owned repo — **[you]**
1. Create/choose a repo under the impf org (e.g. `impf/enzyme`), and copy the
   `dist/enzyme-impf/` files into it (or copy `web/` and run the exporter with
   `--cname enzyme.impf.ai` so `CNAME` is present).
2. Push, then in that repo: **Settings → Pages → Build and deployment → Deploy
   from a branch** (or a Pages Action), selecting the branch/folder that holds
   `index.html`.
3. **Settings → Pages → Custom domain** → enter `enzyme.impf.ai` → **Save**,
   then tick **Enforce HTTPS** once the certificate is issued.

### C. Cloudflare Pages / Netlify / Vercel — **[you]**
Create a project from the impf repo (or drag-and-drop `dist/enzyme-impf/`).
Build command: none. Output/publish directory: the folder with `index.html`.
Add `enzyme.impf.ai` as a custom domain in the project's domain settings.

> This GitHub repository (`psknlr/E-Agent`) and its Pages deploy are **not**
> changed by this guide; no `CNAME` is added here. `enzyme.impf.ai` is served
> by the impf host you choose above.

---

## 3. Point the DNS at the host — **[you]**

`enzyme` is a subdomain, so a `CNAME` record is correct (no apex-domain
limitations apply). In the `impf.ai` DNS zone, add **one** record whose target
is given by the host you picked in step 2:

| Host (step 2) | Record | Name | Target |
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
  `--backend-url https://eagent-api.impf.ai` (step 1), or leave it blank and
  let users enter it under **Agent connection**.

---

## 5. Verify

Open `https://enzyme.impf.ai` and confirm:

1. The status reads **Browser tools loaded** with the tool inventory listed —
   this needs no key and proves the static bundle was served correctly.
2. Under **Model settings**, enter your API key, exact model name and full API
   URL, then send a question. A first successful reply flips the status to
   **Chat ready** and shows the research trace.

**Provider CORS still applies.** Browser + API mode calls your model endpoint
directly from the page, so the endpoint must allow cross-origin requests from
`https://enzyme.impf.ai`. Anthropic's browser-access header is sent
automatically; some providers and gateways reject browser origins — if so, use
an endpoint that permits them, or run Python backend mode (step 4). This is a
provider transport policy, not an E-Agent setting.

---

## What this guide does not do

- It does not change `impf.ai` DNS or touch the impf host — those are the
  **[you]** steps above and require access this repository does not have.
- It does not move the primary GitHub Pages site; `psknlr.github.io/E-Agent/`
  keeps working unchanged.
