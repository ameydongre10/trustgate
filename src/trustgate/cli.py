"""CLI entrypoint for TrustGate."""

import click
from rich.console import Console

console = Console()


@click.group()
@click.version_option(version="0.1.0")
def main() -> None:
    """TrustGate: Impact-Aware Data Trust and Verification Agent."""
    pass


@main.command()
def info() -> None:
    """Display system status, configuration, and registered components."""
    console.print("[bold green]TrustGate v0.1.0[/bold green] - Impact-Aware Data Trust Agent")
    console.print("Operational Mode: Deterministic Policy Engine")


@main.command()
def demo() -> None:
    """Run an end-to-end interactive demo across sample corrupted batches."""
    console.print("[bold cyan]Starting TrustGate Demonstration...[/bold cyan]")


if __name__ == "__main__":
    main()
