# Model providers and the read-only analysis console

Two separate things live here, and the difference matters:

1. **The planner** (`harness/planner.py`) is where a model can influence what a
   run does. It has its own guards, its own approvals, and it is covered in
   `docs/ARCHITECTURE.md`.
2. **The analysis console** (`harness/toolloop.py`) is this document. A model
   reads the finished reference set through the project's own loaders and
   answers a question about it. It cannot change anything.

## Why a model reads through the loaders and not the CSVs

The reference set is a few thousand numbers with a rule attached to most of
them: this `Km` is a bound, that zero is a detection limit, these two entries
are one lineage, this `ND` is not a zero. The CSVs carry the numbers; the
loaders carry the rules. So the tools a model is given are
`eagent.eval.kred_reference` and `eagent.eval.kred_activity`, which refuse a
withheld quantity and say why, in the same words a person would get:

```
$ eagent model ask "What is the Km of PaHBDH H150N?" \
    --provider anthropic --model <id> --allow-network
```
```
  [refused] kinetic_record({"label_id": "PaHBDH_H150N_AAE_activity_only"})
      PaHBDH_H150N_AAE_activity_only: no km: the source gives only a bound
      (< 5400 mM); a bound is not a point estimate
```

A model given the CSV would have read `5400` and reported a Michaelis constant.

## The fence

| rule | how it is enforced |
|---|---|
| read-only | `ToolLoop.register` refuses any tool declaring `writes` or `reaches_network`. The shipped tools are closures over data loaded before the loop starts. |
| bounded | turns, calls per turn, calls in total, bytes per result, wall-clock. Each is a field of `LoopLimits` and each has a test that stops the loop. |
| the model decides nothing | nothing in the module writes to the repository. The output is a transcript. |
| quantities stay sourced | the same `NumericGuard` the planner uses. With `--strict-citations` an answer containing an uncited number is withheld and the transcript says so. Every tool result carries the artifact key, data digest and row, so a model *can* cite properly. |
| the question leaves the machine | `--allow-network` is required, and the refusal says what is sent. |

## Providers

| provider | wire format | key variable | base URL (override) |
|---|---|---|---|
| `anthropic` | Messages API (system prompt is its own field) | `ANTHROPIC_API_KEY` | `https://api.anthropic.com` (`EAGENT_ANTHROPIC_BASE_URL`) |
| `openai` | chat completions (system prompt is the first message) | `OPENAI_API_KEY` | `https://api.openai.com/v1` (`EAGENT_OPENAI_BASE_URL`) |
| `minimax` | chat completions, at `chat/completions` | `MINIMAX_API_KEY` | `https://api.minimax.io/v1` (`EAGENT_MINIMAX_BASE_URL`) |

`OpenAIChatClient` is one client with a configurable base URL, so any service
speaking the chat-completions format — a different vendor, a self-hosted model
— is a configuration change and not a new class:

```
export EAGENT_OPENAI_BASE_URL=http://127.0.0.1:8000/v1
export OPENAI_API_KEY=whatever-the-local-server-wants
eagent model ask "..." --provider openai --model local-model --allow-network
```

### Keys

Read from the environment at call time. Never stored on the object, never in an
exception, a URL, a transcript or an audit file; `redact()` is applied to
anything derived from a provider's own error body, and a test asserts the key
does not appear in any of those. There is no configuration file for keys and
nothing in this repository writes one.

A provider with no key **refuses**. It does not fall back to another provider,
because a run whose answer depends on which key happened to be set is not
reproducible.

### What has actually been established

`eagent model providers` prints this, and the distinction is the same one the
connector layer makes between a verified route and a verified request shape:

* **The route answers.** Each provider was sent one unauthenticated request on
  2026-10-09 and replied with **its own** structured authentication error —
  Anthropic's `authentication_error`, OpenAI's Bearer instruction, MiniMax's
  `authorized_error` naming the Authorization header on the current endpoint.
  A CDN page or a login wall produces none of
  those, so the host, the path and the request parsing are the real API.
  `eagent model providers --probe --allow-network` re-runs that check.
* **No request shape is verified.** No provider credential existed in the
  environment this was written in, so **not one real completion has been
  requested from any provider.** Every request body is written from the
  published reference and exercised against a fake opener.
  `shape_verified` is `False` for all three and is a field, not a footnote, so
  a run can record it.

Older MiniMax responses reported failures inside HTTP 200 using
`base_resp.status_code`; the client still checks that field. The current
MiniMax integration uses `max_completion_tokens` and `reasoning_split` so
thinking text is separate from the harness JSON answer, following
[MiniMax's API reference](https://platform.minimax.io/docs/api-reference/text-chat-openai).

## Web chat

[The GitHub Pages interface](https://psknlr.github.io/E-Agent/) connects to
`python -m eagent.web`, which executes this same console over the reference set.
The backend includes tool schemas in the system prompt so the model can
discover the actual readers. Read-only catalogs enumerate structure and activity
identifiers before individual records are requested. The web backend uses strict
citation checks and exposes the full tool transcript, including refusals.

[WEB_CHAT.md](WEB_CHAT.md) describes MiniMax hosting and connection settings.
The public page contains no model key. Runtime health reports configured status
separately from a successful provider completion; it supersedes static prose
as evidence of whether your backend is available.

## 中文摘要

这里有两个不同的东西。**planner** 是模型能影响运行流程的接缝，有自己的守卫与审批；本文讲的是
**只读分析控制台**：模型通过本项目自己的加载器读已经定稿的参考集，回答问题，改不了任何东西。

为什么让模型走加载器而不是直接给 CSV：参考集里几千个数字，大多数都附着一条规则——这个 `Km` 只是界限、
那个 0 是检出限、这两个条目是同一谱系、`ND` 不是 0。CSV 只有数字，加载器才有规则。所以模型拿到的工具会
**拒绝**被扣留的量并说明原因（例如 H150N 的 `Km`：原文只给 `<5400 mM`，界限不是点估计）。给它 CSV 的话，
它会读到 `5400` 然后当成米氏常数报出来。

围栏：注册时拒绝任何声明会写入或联网的工具；轮数、每轮调用数、总调用数、单条结果字节数、挂钟时间都有上限；
模块不向仓库写任何东西，输出只是一份完整记录；沿用 planner 的数值守卫，`--strict-citations` 下未标注来源的
数字会让答案被扣留；发送需要 `--allow-network`，拒绝信息会说明什么会离开本机。

支持三个提供方：`anthropic`（Messages API）、`openai` 与 `minimax`（都是 chat-completions 格式）。
`OpenAIChatClient` 用可配置的 base URL，所以任何兼容该格式的服务（含自建本地模型）只是改配置。
密钥只在调用时从环境变量读取，不存在对象上、不进异常、不进记录文件，仓库里没有也不会写任何密钥配置文件；
没有密钥的提供方**直接拒绝**，不会回退到别的提供方。

**已确认与未确认**：三个提供方的路由都真实存在——各自返回了自己的结构化认证错误（2026-10-09 实测），
所以主机、路径和请求解析是真的 API。但**没有任何一个请求体被验证过**：本环境里没有任何提供方密钥，
所以没有向任何提供方发出过一次真实补全，`shape_verified` 三个都是 `false`。实测唯一改变了代码的一点：
旧版 MiniMax 把认证失败放在 HTTP 200 里的 `base_resp.status_code`，客户端仍检查它。当前端点为
`https://api.minimax.io/v1/chat/completions`，实测返回 `authorized_error`。
网页聊天通过 Python 后端运行同一控制台；实时连接状态与实际补全验证分开显示，部署方法见 `WEB_CHAT.md`。
