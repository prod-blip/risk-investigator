# Risk Investigator

A reproducible prototype for testing whether a tool-using AI agent can investigate changes in payment risk more accurately and efficiently than a scripted diagnostic workflow.

**Current status: Steps 1–3 are implemented; Step 3 awaits user review.** You can generate experiments, run the scripted baseline, and use a checkpointed investigation orchestrator in clearly labeled demo or live OpenAI mode. All 64 offline tests pass. Live OpenAI smoke tests produced supported diagnoses for the three development fixtures and suspected fraud before outcomes were available. Earlier attempts were inconclusive, and the early retest required a rejected-report correction. See [the validation record](docs/step3-validation.md). The web interface and comparative agent benchmark are not implemented yet. No agent superiority claims are made.

## What you can run today

- Generate synthetic payments for three controlled conditions: emerging fraud, a broken device signal, and an overly restrictive rule.
- Run a transparent heuristic scorer and policy engine on every transaction.
- Inspect before/after decline rates and missing signals.
- Retrieve recorded inputs, feature contributions, triggered rules, versions, and deployment events from SQLite.
- Reproduce decisions from their recorded inputs.
- Run a scripted investigation with evidence references and a tested candidate response.
- Compare an early investigation with one where synthetic fraud outcomes have become available.
- Export JSON and Markdown reports for review.

The scorer is explicitly **a heuristic fixture, not a trained fraud model or foundation model**. The foundation model guides an investigation using controlled tools. Existing systems already automate and investigate risk decisions; this project tests incremental value rather than asserting that a particular company lacks those capabilities.

## Quick start

Requires Python 3.11 or newer. The project uses the Python standard library: no package installation is required. Baseline and demo mode need no API key. Live mode requires an OpenAI API key and network access.

From the repository root:

```bash
python3 -m risk_investigator generate --scenario all --seed 42 --output-dir data/demo
python3 -m risk_investigator inspect --database data/demo/observations.sqlite
python3 -m risk_investigator investigate --database data/demo/observations.sqlite --as-of 2026-01-27T00:00:00Z --output-dir data/reports-mature
python3 -m unittest discover -s tests -v
```

On Windows, use `python` instead of `python3` if needed.

Generation refuses to overwrite existing experiment files. To try a different seed, choose a different output directory:

```bash
python3 -m risk_investigator generate --scenario broken_signal --seed 43 --output-dir data/seed-43
```

Each scenario has 500 transactions before and 500 after the intervention by default. Set `--transactions-per-period` to change this (50–100,000). `--scenario all` creates three incidents, totaling 3,000 transactions with default settings.

## Controlled conditions

| Condition | Intervention | What stays controlled |
|---|---|---|
| Emerging fraud | More fraudulent accounts share payment instruments while mostly keeping individual transaction velocity low. | Scorer, policy, and feature integration versions. |
| Broken signal | A device parser deployment drops device information on mobile web transactions. The existing missing-device rule declines those payments. | Scorer, policy, and underlying traffic/labels compared with the restrictive-rule scenario. |
| Restrictive rule | The decline score threshold changes from 70 to 25. | Scorer and feature integration; underlying traffic/labels compared with the broken-signal scenario. |

These are deliberately simplified experimental conditions. Additional held-out variants, healthy periods, customer-mix changes, and ambiguous incidents will be needed before making claims about investigative reliability.

## Evidence versus grading truth

Generation creates three local artifacts:

| Artifact | Contents | Intended consumer |
|---|---|---|
| `observations.sqlite` | Transactions, decisions, input snapshots, signal timestamps, versions, changes, and incident boundaries. | Investigator tools. |
| `outcomes.sqlite` | Synthetic fraud/legitimacy outcome events with explicit availability timestamps; no scenario causes. | Time-filtered outcome tools. |
| `evaluation-private.json` | Expected scenario causes and synthetic fraud labels with their availability times. | Benchmark grader only. |

The evidence database does **not** contain scenario names, expected causes, or fraud labels. Investigator-facing incident identifiers are opaque. Synthetic labels become available 14 days after each transaction; this is an experimental convention, not a claim about real fraud reporting times. Real-time investigations must not use those future outcomes. Any future outcome tool must enforce an explicit investigation timestamp.

Source device values represent a retained upstream payload. Delivered device values represent the features actually supplied to the scorer. That distinction lets the broken-signal scenario be investigated from source evidence without inventing missing data. The input snapshot records both, but scoring uses only the delivered value.

File separation is a design boundary, not an OS security boundary. The future agent will receive controlled tools rather than arbitrary filesystem/SQL access. The scripted investigator never reads `evaluation-private.json`. The grader must keep that private artifact out of prompts and tool responses. The outcome adapter exposes only transaction outcomes already available at the requested investigation time.

For experiments created before Step 2, regenerate into a new directory to create the outcome store. Existing files will not be overwritten:

```bash
python3 -m risk_investigator generate --output-dir data/step2
python3 -m risk_investigator investigate --database data/step2/observations.sqlite --as-of 2026-01-27T00:00:00Z --output-dir data/step2-reports
```

Generated artifacts are ignored by Git. No real customer, merchant, card, or payment data is used. Amounts are integer cents in a single synthetic currency; country values describe traffic cohorts, not currency conversion.

## What the evidence contains

The SQLite `decision_evidence` view joins transactions with their recorded decisions. `changes` contains deployment/configuration events. The `incidents` table supplies analysis windows.

A recorded decision includes:

- The full timestamped input snapshot.
- Heuristic score contributions.
- Effective decline threshold and model/policy versions.
- All triggered decline rules.
- The final approve/decline action.

The first policy declines a payment if its score is at least 70 or its delivered device signal is missing. The restrictive-rule experiment lowers that threshold to 25. These are synthetic fixture policies, not recommended production thresholds.

## Architecture

```text
Synthetic payments → scorer + rules → recorded decisions
                                          ↓
                              alert or analyst question
                                          ↓
                              investigation orchestrator
                                          ↓
                              LLM ↔ controlled tools
                                          ↓
                         evidence report + sandbox replay
                                          ↓
                                    human review
```

The LLM selects checks and interprets evidence. Tools calculate metrics and run replay. Proposed changes are evaluated separately from the live decision path.

## Scripted investigation

The runbook calls six checks in a fixed order, then selects one of three bounded response tests. It is a deterministic baseline, not an LLM agent.

| Tool | Purpose |
|---|---|
| `incident_context` | Identify the configured window and visible observations. |
| `compare_cohorts` | Compare channel, country, or merchant groups and overall decision rates. |
| `feature_health` | Inspect missing/stale signals, upstream/delivered mismatches, and missing-signal declines. |
| `deployment_history` | Inspect timestamp-eligible deployment events and actual decision versions/thresholds. |
| `linked_activity` | Inspect shared instruments across distinct accounts; linkage alone is not fraud proof. |
| `outcome_metrics` | Calculate trade-offs only for labels available at the investigation timestamp. |
| `decision_trace` | Resolve a decision reference to its recorded input, score contributions, and triggered rules. |
| `replay` | Reproduce original decisions, then test a signal repair, previous threshold, or sequential linkage control. |

All tools are read-only and scoped to one incident and investigation timestamp. There is no arbitrary SQL or filesystem tool. Each session defaults to 12 calls; failed tool executions consume budget. Inputs are bounded to 20,000 visible transactions and 100 change events. Observation fingerprints and eligible-outcome fingerprints identify the evidence snapshots used.

Run the same experiment before labels are available:

```bash
python3 -m risk_investigator investigate --database data/step2/observations.sqlite --as-of 2026-01-13T00:00:00Z --output-dir data/reports-early
```

The linked attack remains **suspected** at this timestamp. With eligible outcomes on January 27, the supplied attack fixture receives a supported diagnosis. The configured 14-day delay is synthetic. Missing outcomes never imply legitimate payments.

Reports distinguish `supported`, `suspected`, `inconclusive`, and `no_material_change`. Multiple supported mechanisms, incomplete windows, unsupported models, or unreproducible decision traces prevent endorsing a single response. A no-material-change result covers only the runbook's indicators and thresholds.

Use `--incident-id` to investigate one incident, `--format markdown` to print a readable report, or `--outcomes-database` to select an explicit outcome store. Otherwise, the CLI discovers adjacent `outcomes.sqlite` when present. JSON/Markdown export includes tool results and evidence references. Report files are never silently overwritten.

Replay is a **what-if experiment**, not a deployment or an estimate of actual future loss savings. Linkage counts grow chronologically and never use later accounts to classify earlier transactions. Behavioral responses to a new policy are not simulated. Synthetic labels exist for declined payments too; real rejected-payment counterfactuals may be unknown. Approved-fraud amounts measure synthetic exposure, not realized accounting loss. Partial outcome metrics must not be generalized to the unlabelled population.

No real policy is changed. Human review is required for every candidate response.

## Agent and orchestrator

The real agent uses the Responses API to choose tools, inspect their outputs, revise short hypotheses, and return a structured report. The model may choose a different sequence from the baseline. Investigation evidence and private grading answers remain separate.

The orchestrator validates tool names and arguments in application code, applies budgets, records usage, saves checkpoints, and validates the final report. No deployment, shell, filesystem, or arbitrary SQL tool is exposed to the model. Evidence strings are treated as data, not instructions.

### No-key demonstration

```bash
python3 -m risk_investigator agent --provider demo --database data/step2/observations.sqlite --all --as-of 2026-01-27T00:00:00Z --output-dir data/agent-demo
```

Demo mode exercises the same orchestration/checkpoint/validation path with a scripted provider. Reports state `provider: demo`, `model: scripted-demo-not-an-llm`, and zero model calls. It does not demonstrate an LLM's performance and must not be included as live-agent benchmark evidence.

### Live-model setup

Copy `.env.example` to `.env` and edit the key locally. The file is ignored by Git. Existing shell environment values take precedence. The loader does not evaluate shell commands or expand variables.

The template uses `gpt-4.1-mini` for initial smoke tests. It supports function calling and structured output; the template's token rates are $0.40 input and $1.60 output per million tokens, verified on October 12, 2026. Verify rates again before testing or changing models. [Official model documentation](https://developers.openai.com/api/docs/models/gpt-4.1-mini)

The API key belongs in `OPENAI_API_KEY`, not in a prompt, CLI argument, or checked-in file. Configure `OPENAI_MODEL`, `OPENAI_PRICING_MODEL`, and both pricing rates together. Selecting a different model with mismatched pricing is rejected. Both rates can alternatively be supplied explicitly through CLI flags.

For a single smoke-test incident from the default seed:

```bash
python3 -m risk_investigator agent --provider openai --database data/step2/observations.sqlite --incident-id inc_02ac821c17685ea5 --as-of 2026-01-27T00:00:00Z --max-cost-usd 0.25 --output-dir data/agent-live
```

Use `inspect` to obtain incident identifiers for other seeds. Live requests send synthetic evidence to OpenAI. They do not send the grader file or credentials as model input. The adapter uses `store: false` and carries tool calls, matching call outputs, and returned reasoning items in a local stateless transcript. This setting is not a general promise of zero data retention. [Function calling](https://developers.openai.com/api/docs/guides/function-calling), [Responses state guidance](https://developers.openai.com/api/docs/guides/migrate-to-responses)

### Checkpoints and limits

Run IDs are printed when investigations start. SQLite checkpoints are stored at `data/runs.sqlite` by default.

```bash
python3 -m risk_investigator run-info RUN_ID
python3 -m risk_investigator agent --resume RUN_ID
```

`run-info` does not need a key. Resume preserves the recorded scope, time, model, evidence fingerprints, and consumed budgets. Completed runs return the saved report without another model call. Changed evidence prevents mixing snapshots. A lease permits one active worker per run; after a hard crash it expires within 120 seconds. Read-only tool work may be recomputed if a process dies between execution and its checkpoint.

Defaults per incident are 12 provider attempts (including retries), 12 tool attempts (including rejected calls), 2,500 output tokens per response, 160,000 request bytes, and 180 seconds of active loop time. HTTP timeouts are bounded by the remaining time. Tool work and deadlines are checked cooperatively; they are not a hard OS CPU deadline. Startup evidence loading is outside the active-loop timer.

Live mode defaults to a $0.25 **estimated cost guard per incident**. Before another request, the guard reserves a conservative byte-based input estimate plus the maximum output allowance. After responses it accounts for reported tokens at the configured rates, ignoring cache discounts. This is not exact tokenization or an invoice-level cap. Prices must match the actual model. Ambiguous transport failures retain a cost reserve and mark billing uncertain; only explicit rate-limit refusals receive automatic bounded retries. An explicit resume after an ambiguous failed request can incur another charge. `--all` applies these limits separately to each incident.

### Report verification

Every final finding must cite an existing successful tool result and a valid JSON pointer. The application attaches the observed value itself. It rejects invalid references, unsupported diagnosis prerequisites, fraud confirmation without sufficient eligible outcomes, and a response that lacks successful replay. It never accepts deployment authority from the model.

These checks do **not** establish that all natural-language interpretations or hypothesis verdicts are faithful. The report labels that limit and still requires human review. Quantitative replay values come directly from tools; qualitative interpretation remains model-generated. Token/cost metadata is a run measurement, not a claim of investigation correctness or agent superiority.

### Verification status

All 64 offline tests pass, covering the no-key workflow, mocked Responses payload/usage, report rejections, schema validation, delayed labels, checkpoint resume, worker leases, budget stops, and error handling. Actual `gpt-4.1-mini` Responses API calls also exercised the provider, tool loop, report validation, and replay. A temporary curl relay was needed because this Codex sandbox blocked Python DNS; the normal Python HTTP transport has mocked coverage but was not verified end to end here. [Full smoke-test results and limitations](docs/step3-validation.md).

## Evaluation plan

Compare the agent and a scripted diagnostic workflow with identical evidence, tools, and budgets on multiple held-out instances. Measure diagnosis correctness, time to a correct diagnosis, cost per correct diagnosis, unsupported conclusions, evidence coverage, and replay trade-offs between fraud captured and legitimate payments blocked.

There are no agent-versus-baseline results yet. Simulator statistics and passing unit tests are not proof of agent superiority or production impact. Synthetic results must remain labeled as synthetic, including in the deck.

See [the build plan](docs/build-plan.md) for the gated implementation steps.
