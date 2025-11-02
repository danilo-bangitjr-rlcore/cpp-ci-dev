from datetime import datetime

import click
import zmq
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
    default=5571,
    type=int,
    help="Event bus subscriber port (XPUB socket)",
    show_default=True,
)
@click.option(
    "--topic",
    default="",
    help="Filter by specific topic (empty string subscribes to all)",
    show_default=True,
)
@click.pass_context
def monitor_events(ctx: click.Context, host: str, port: int, topic: str) -> None:
    """
    Monitor event bus traffic in real-time.

    This development tool connects to the coredinator event bus and displays
    all events flowing through the system. Useful for debugging event-driven
    interactions between services.
    """
    endpoint = f"tcp://{host}:{port}"
    subscription = "ALL topics" if topic == "" else f"topic '{topic}'"

    console.print(Panel.fit(
        f"[bold cyan]EVENT BUS MONITOR[/bold cyan]\n\n"
        f"[dim]Endpoint:[/dim] {endpoint}\n"
        f"[dim]Subscription:[/dim] {subscription}\n\n"
        f"[yellow]Press Ctrl+C to stop[/yellow]",
        border_style="cyan",
    ))

    context = zmq.Context()
    sub_socket = context.socket(zmq.SUB)

    try:
        sub_socket.connect(endpoint)
        sub_socket.setsockopt_string(zmq.SUBSCRIBE, topic)

        console.print(f"[green]✓[/green] Connected to {endpoint}\n")

        while True:
            if sub_socket.poll(timeout=100):
                message_parts = sub_socket.recv_multipart()

                if len(message_parts) >= 2:
                    msg_topic = message_parts[0].decode("utf-8", errors="ignore")
                    payload = message_parts[1].decode("utf-8", errors="ignore")
                    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]

                    console.print(f"[dim]{timestamp}[/dim] [bold blue]{msg_topic}[/bold blue]")
                    console.print(f"  [dim]→[/dim] {payload}\n")
                else:
                    console.print(
                        f"[yellow]⚠[/yellow] Malformed message ({len(message_parts)} parts)\n",
                    )

    except KeyboardInterrupt:
        console.print("\n[yellow]Monitor stopped[/yellow]")
    except zmq.ZMQError as e:
        console.print(f"[red]✗[/red] ZMQ error: {e}", style="bold red")
        raise click.Abort() from e
    finally:
        sub_socket.close()
        context.term()
