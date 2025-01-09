#!/usr/bin/env python3

from typing import Any, List


from argparse import ArgumentParser
from textual.app import App, ComposeResult
from textual.containers import HorizontalGroup, VerticalScroll
from textual.widgets import Button, Checkbox, Footer, Header, Input, Label

from corerl.bin_modules.opc_explorer_utils import get_variables_from_dict, read_opc


class VariableRow(HorizontalGroup):
    def __init__(self, variable: dict, **kwargs: Any):
        super().__init__(**kwargs)
        self.variable = variable

    def compose(self) -> ComposeResult:
        yield Label(str(self.variable['key']))
        yield Label(str(self.variable['path']))
        yield Label(str(self.variable['val']))
        yield Label(str(self.variable['DataType']))
        yield Label(str(self.variable['nodeid']))
        yield Checkbox()
        yield Checkbox()


class ButtonsRow(HorizontalGroup):
    def __init__(self, url: str, **kwargs: Any):
        super().__init__(**kwargs)
        self.url = url

    def compose(self) -> ComposeResult:
        yield Label(f"Connected to {self.url}")
        yield Button("Import Config", variant="warning")
        yield Button("Export Config", variant="warning")


class HeadersRow(HorizontalGroup):
    def __init__(self, headers: List[str], **kwargs: Any):
        super().__init__(**kwargs)
        self.headers = headers

    def compose(self) -> ComposeResult:
        for header in self.headers:
            yield Label(header)


class OpcApp(App):
    CSS_PATH = "../bin_modules/opc_explorer.tcss"

    def __init__(self, url: str, **kwargs: Any):
        super().__init__(**kwargs)
        self.url = url

    def compose(self) -> ComposeResult:
        self.vertical_scroll = VerticalScroll(id="results-container")
        yield Header()
        yield ButtonsRow(self.url, id="buttons")
        yield Input(placeholder="Search for a variable or path", id="opc-search")
        yield HeadersRow(["Variable Name",
                          "Variable Path",
                          "Variable Value",
                          "Variable Type",
                          "NodeId",
                          "Is Action",
                          "Is Observation"],
                         id="headers")
        yield self.vertical_scroll
        yield Footer()

    def on_mount(self) -> None:
        # Reads from OPC when the app starts
        dict_data = read_opc(self.url)
        self.variables_list = get_variables_from_dict(dict_data)

        for variable in self.variables_list:
            newRow = VariableRow(variable)
            self.query_one("#results-container").mount(newRow)

    def on_input_changed(self, message: Input.Changed) -> None:
        if message.value:
            for variable, child in zip(self.variables_list, self.vertical_scroll.children, strict=True):
                if message.value in variable["key"] or message.value in variable["path"]:
                    child.remove_class("invisible")
                else:
                    child.add_class("invisible")
        else:
            for child in self.vertical_scroll.children:
                if child.has_class("invisible"):
                    child.remove_class("invisible")


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "-u",
        "--url",
        help="URL/endpoint of OPC UA server",
        default="opc.tcp://0.0.0.0:4840/",
        metavar="URL",
    )
    args = parser.parse_args()

    app = OpcApp(args.url)
    app.run()


if __name__ == "__main__":
    main()
