import time
from datetime import datetime

import click
from lib_defs.type_defs.base_events import Event, EventTopic, EventType
from lib_events.client.event_bus_client import EventBusClient
from rich.console import Console
from rich.panel import Panel

console = Console()


@click.command("monitor-events")
@click.option(
    "--host",
    default="localhost",
    help="Event bus host to connect to",
    show_default=True,
)
@click.option(
    "--port",
    default=5580,
    type=int,
    help="Event bus port (ROUTER socket)",
    show_default=True,
)
@click.option(
    "--topic",
    default=None,
    help="Filter by specific topic (omit to subscribe to all topics)",
    show_default=True,
)
@click.pass_context
def monitor_events(ctx: click.Context, host: str, port: int, topic: str | None) -> None:
    """
    Monitor event bus traffic in real-time.

    This development tool connects to the coredinator event bus and displays
    all events flowing through the system. Useful for debugging event-driven
    interactions between services.
    """
    endpoint = f"tcp://{host}:{port}"
    subscription = "ALL topics" if topic is None else f"topic '{topic}'"

    console.print(Panel.fit(
        f"[bold cyan]EVENT BUS MONITOR[/bold cyan]\n\n"
        f"[dim]Endpoint:[/dim] {endpoint}\n"
        f"[dim]Subscription:[/dim] {subscription}\n\n"
        f"[yellow]Press Ctrl+C to stop[/yellow]",
        border_style="cyan",
    ))

    received_events = []

    def display_event(event: Event):
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        event_type = event.type.name
        event_id = str(event.id)[:8]

        console.print(f"[dim]{timestamp}[/dim] [bold blue]{event_type}[/bold blue] [dim]({event_id})[/dim]")
        console.print(f"  [dim]→[/dim] {event.model_dump_json()}\n")
        received_events.append(event)

    client = EventBusClient(
        host=host,
        port=port,
        service_id="monitor-cli",
    )

    try:
        client.connect()

        if topic is None:
            for event_topic in EventTopic:
                client.subscribe(event_topic)
        else:
            try:
                event_topic = EventTopic[topic]
                client.subscribe(event_topic)
            except KeyError as e:
                console.print(f"[red]✗[/red] Invalid topic: {topic}", style="bold red")
                console.print(f"[dim]Valid topics:[/dim] {', '.join(t.name for t in EventTopic)}")
                raise click.Abort() from e

        for event_type in EventType:
            client.attach_callback(event_type, display_event)

        client.start_consumer()
        console.print(f"[green]✓[/green] Connected to {endpoint}\n")

        while True:
            time.sleep(0.1)

    except KeyboardInterrupt:
        console.print("\n[yellow]Monitor stopped[/yellow]")
    except Exception as e:
        console.print(f"[red]✗[/red] Error: {e}", style="bold red")
        raise click.Abort() from e
    finally:
        client.close()
