import argparse
import json
from pathlib import Path
import sqlite3

from .baseline import investigate, render_markdown
from .agent_cli import add_agent_parser, execute_agent
from .outcomes import write_outcomes
from .simulation import SCENARIOS, simulate
from .storage import summarize, write_experiments
from .tools import InvestigationTools


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate and inspect reproducible synthetic payment risk experiments."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate", help="Create evidence and separate private grading truth")
    generate.add_argument("--scenario", choices=["all", *SCENARIOS], default="all")
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument("--transactions-per-period", type=int, default=500)
    generate.add_argument("--output-dir", type=Path, default=Path("data/demo"))
    inspect = commands.add_parser("inspect", help="Show observed decline rates without fraud labels")
    inspect.add_argument("--database", type=Path, default=Path("data/demo/observations.sqlite"))
    investigation = commands.add_parser("investigate", help="Run the scripted baseline (no LLM)")
    investigation.add_argument("--database", type=Path, default=Path("data/demo/observations.sqlite"))
    investigation.add_argument("--outcomes-database", type=Path,
                               help="Optional outcome store; otherwise use adjacent outcomes.sqlite if present")
    investigation.add_argument("--incident-id", help="One incident; omit to investigate all incidents")
    investigation.add_argument("--as-of", required=True, help="Timezone-aware investigation timestamp")
    investigation.add_argument("--format", choices=("json", "markdown"), default="json")
    investigation.add_argument("--output-dir", type=Path, help="Save full JSON/Markdown reports and print a summary")
    add_agent_parser(commands)
    args = parser.parse_args()
    try:
        if args.command == "generate":
            scenarios = SCENARIOS if args.scenario == "all" else (args.scenario,)
            experiments = [simulate(
                scenario, seed=args.seed, transactions_per_period=args.transactions_per_period
            ) for scenario in scenarios]
            outcome_path = args.output_dir / "outcomes.sqlite"
            if outcome_path.exists():
                raise FileExistsError(f"Output already exists in {args.output_dir}; choose a new --output-dir")
            database, evaluation = write_experiments(experiments, args.output_dir)
            outcomes = write_outcomes(experiments, outcome_path)
            result = {
                "synthetic_data": True,
                "observations_database": str(database),
                "private_evaluation_file": str(evaluation),
                "timestamped_outcomes_database": str(outcomes),
                "summary": summarize(database),
            }
        elif args.command == "inspect":
            result = {"synthetic_data": True, "summary": summarize(args.database)}
        elif args.command in ("agent", "run-info"):
            result = execute_agent(args)
            if isinstance(result, str):
                print(result)
                parser.exit(getattr(args, "agent_exit_code", 0))
        else:
            from contextlib import closing
            with closing(sqlite3.connect(args.database.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
                incident_ids = [r[0] for r in connection.execute("SELECT incident_id FROM incidents ORDER BY incident_id")]
            if args.incident_id:
                if args.incident_id not in incident_ids:
                    raise ValueError(f"Unknown incident: {args.incident_id}")
                incident_ids = [args.incident_id]
            outcomes_database = args.outcomes_database
            if outcomes_database is None:
                adjacent = args.database.with_name("outcomes.sqlite")
                outcomes_database = adjacent if adjacent.exists() else None
            reports = [investigate(InvestigationTools(
                args.database, incident_id, as_of=args.as_of, outcomes_database=outcomes_database
            )) for incident_id in incident_ids]
            if args.output_dir:
                args.output_dir.mkdir(parents=True, exist_ok=True)
                destinations = [(report, args.output_dir / f"{report['incident_id']}.{suffix}", suffix)
                                for report in reports for suffix in ("json", "md")]
                if any(path.exists() for _, path, _ in destinations):
                    raise FileExistsError("A report already exists; choose a new --output-dir")
                for report, path, suffix in destinations:
                    content = render_markdown(report) if suffix == "md" else json.dumps(report, indent=2) + "\n"
                    with path.open("x") as handle:
                        handle.write(content)
                result = {
                    "mode": "scripted-baseline-v1", "synthetic_data": True,
                    "reports_directory": str(args.output_dir.resolve()),
                    "reports": [{key: report[key] for key in
                                 ("incident_id", "as_of", "status", "diagnosis", "tool_calls")}
                                for report in reports],
                }
            elif args.format == "markdown":
                print("\n\n---\n\n".join(render_markdown(report) for report in reports))
                return
            else:
                result = {"mode": "scripted-baseline-v1", "synthetic_data": True, "reports": reports}
        print(json.dumps(result, indent=2))
        if args.command == "agent":
            parser.exit(args.agent_exit_code)
    except (ValueError, OSError, sqlite3.Error) as error:
        parser.exit(2, f"Error: {error}\n")
