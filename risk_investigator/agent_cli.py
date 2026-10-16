from contextlib import closing
import json
import os
from pathlib import Path
import sqlite3

from .agent import AgentLimits, new_run, run_agent
from .baseline import render_markdown
from .config import load_local_env
from .demo_provider import DemoProvider
from .provider import OpenAIProvider
from .run_store import RunStore
from .tools import InvestigationTools


def add_agent_parser(commands) -> None:
    parser = commands.add_parser("agent", help="Run the orchestrator in explicit demo or live OpenAI mode")
    parser.add_argument("--provider", choices=("demo", "openai"), help="Defaults to demo for new runs")
    parser.add_argument("--database", type=Path, default=Path("data/demo/observations.sqlite"))
    parser.add_argument("--outcomes-database", type=Path)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--incident-id")
    selection.add_argument("--all", action="store_true", help="Run all incidents; live limits apply per incident")
    selection.add_argument("--resume", metavar="RUN_ID", help="Continue from saved scope, provider, and budgets")
    parser.add_argument("--as-of", help="Required for new runs; timezone-aware timestamp")
    parser.add_argument("--question", default="Investigate why payment declines or fraud outcomes changed.")
    parser.add_argument("--model", help="OpenAI model; otherwise read OPENAI_MODEL")
    parser.add_argument("--state-db", type=Path, default=Path("data/runs.sqlite"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--format", choices=("json", "markdown"), default="json")
    for flag in ("max-model-calls", "max-tool-calls", "max-output-tokens", "max-input-bytes", "max-retries"):
        parser.add_argument(f"--{flag}", type=int)
    for flag in ("time-limit-seconds", "request-timeout-seconds", "max-cost-usd",
                 "input-usd-per-million", "output-usd-per-million"):
        parser.add_argument(f"--{flag}", type=float)
    info = commands.add_parser("run-info", help="Inspect saved progress or a report without an API key")
    info.add_argument("run_id")
    info.add_argument("--state-db", type=Path, default=Path("data/runs.sqlite"))


def _provider(kind: str, model: str | None):
    if kind == "demo":
        return DemoProvider()
    load_local_env()
    return OpenAIProvider(model or os.environ.get("OPENAI_MODEL", ""))


def _limits(args, provider) -> AgentLimits:
    values = {key: getattr(args, key) for key in (
        "max_model_calls", "max_tool_calls", "max_output_tokens", "max_input_bytes",
        "max_retries", "time_limit_seconds", "request_timeout_seconds",
    ) if getattr(args, key) is not None}
    if provider.kind == "openai":
        explicit = (args.input_usd_per_million, args.output_usd_per_million)
        if any(v is not None for v in explicit):
            if any(v is None for v in explicit):
                raise ValueError("Pass both input/output pricing rates for the selected model")
            input_rate, output_rate = explicit
        else:
            pricing_model = os.environ.get("OPENAI_PRICING_MODEL", "gpt-4.1-mini")
            if pricing_model != provider.model:
                raise ValueError("Pricing model differs from the requested model; configure matching rates/model")
            try:
                input_rate = float(os.environ["OPENAI_INPUT_USD_PER_MILLION"])
                output_rate = float(os.environ["OPENAI_OUTPUT_USD_PER_MILLION"])
            except (ValueError, KeyError):
                raise ValueError("Configure numeric input/output pricing rates or pass both rate flags") from None
        values.update(input_usd_per_million=input_rate, output_usd_per_million=output_rate,
                      max_cost_usd=args.max_cost_usd if args.max_cost_usd is not None else 0.25)
    elif args.max_cost_usd is not None:
        raise ValueError("API cost guard is for live mode; demo performs no API calls")
    return AgentLimits(**values)


def execute_agent(args) -> dict | str:
    store = RunStore(args.state_db)
    if args.command == "run-info":
        state = store.load(args.run_id)
        return {"run_id": state["run_id"], "run_status": state["run_status"],
                "provider": state["provider"], "model": state["model"],
                "provider_attempts": state["provider_attempts"], "tool_attempts": state["tool_attempts"],
                "active_seconds": state["active_seconds"], "errors": state["errors"],
                "report": state["report"]}
    if args.resume:
        saved = store.load(args.resume)
        if args.provider is not None and args.provider != saved["provider"]:
            raise ValueError("Resume must preserve the saved provider")
        if args.model is not None and args.model != saved["model"]:
            raise ValueError("Resume must preserve the saved model")
        if args.as_of is not None or any(getattr(args, key) is not None for key in (
            "max_model_calls", "max_tool_calls", "max_output_tokens", "max_input_bytes", "max_retries",
            "time_limit_seconds", "request_timeout_seconds", "max_cost_usd", "input_usd_per_million",
            "output_usd_per_million",
        )):
            raise ValueError("Resume preserves saved timestamps and limits; start a new run to change them")
        if saved["run_status"] in ("completed", "budget_exhausted"):
            reports = [saved["report"]]
        else:
            provider = _provider(saved["provider"], saved["model"])
            reports = [run_agent(store, args.resume, provider)]
    else:
        if not args.as_of or not (args.incident_id or args.all):
            raise ValueError("New runs require --as-of and either --incident-id or --all")
        provider = _provider(args.provider or "demo", args.model)
        limits = _limits(args, provider)
        with closing(sqlite3.connect(args.database.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
            ids = [r[0] for r in connection.execute("SELECT incident_id FROM incidents ORDER BY incident_id")]
        if args.incident_id:
            if args.incident_id not in ids:
                raise ValueError("Unknown incident")
            ids = [args.incident_id]
        outcomes_database = args.outcomes_database
        if outcomes_database is None:
            adjacent = args.database.with_name("outcomes.sqlite")
            outcomes_database = adjacent if adjacent.exists() else None
        reports = []
        for incident_id in ids:
            tools = InvestigationTools(args.database, incident_id, as_of=args.as_of,
                                       outcomes_database=outcomes_database, max_calls=limits.max_tool_calls)
            run_id = new_run(store, tools, provider, limits, database=args.database,
                             outcomes_database=outcomes_database, question=args.question)
            print(f"Started {provider.kind} investigation {run_id}", file=__import__("sys").stderr, flush=True)
            reports.append(run_agent(store, run_id, provider))
    args.agent_exit_code = 0 if all(r["run_status"] == "completed" for r in reports) else 1
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        destinations = [(report, args.output_dir / f"{report['run_id']}.{suffix}", suffix)
                        for report in reports for suffix in ("json", "md")]
        if any(path.exists() for _, path, _ in destinations):
            raise FileExistsError("A report already exists; choose a new --output-dir")
        for report, path, suffix in destinations:
            with path.open("x") as handle:
                handle.write(render_markdown(report) if suffix == "md" else json.dumps(report, indent=2) + "\n")
        return {"synthetic_data": True, "reports_directory": str(args.output_dir.resolve()),
                "reports": [{key: report[key] for key in (
                    "run_id", "run_status", "provider", "model", "incident_id", "status", "diagnosis",
                    "tool_calls", "model_calls", "usage", "mode_notice",
                )} for report in reports]}
    if args.format == "markdown":
        return "\n\n---\n\n".join(render_markdown(report) for report in reports)
    return {"synthetic_data": True, "reports": reports}
