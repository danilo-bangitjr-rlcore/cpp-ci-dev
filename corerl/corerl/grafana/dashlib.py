import json
from typing import Any

from pydantic import BaseModel, Field


class GridPos(BaseModel):
    h: int
    w: int
    x: int
    y: int

class Datasource(BaseModel):
    type: str = "grafana-postgresql-datasource"
    uid: str

class SqlTarget(BaseModel):
    datasource: Datasource
    rawSql: str
    refId: str
    format: str = "table"
    rawQuery: bool = True
    editorMode: str = "code"

class ReduceOptions(BaseModel):
    calcs: list[str] = Field(default_factory=lambda: ["lastNotNull"])
    fields: str = ""
    values: bool = False

class StatOptions(BaseModel):
    reduceOptions: ReduceOptions = Field(default_factory=ReduceOptions)
    wideLayout: bool = True

class Stat(BaseModel):
    title: str
    type: str = "stat"
    id: int | None = None
    datasource: Datasource
    targets: list[SqlTarget]
    gridPos: GridPos
    options: StatOptions = Field(default_factory=StatOptions)

class TimeSeries(BaseModel):
    title: str
    type: str = "timeseries"
    id: int | None = None
    datasource: Datasource
    targets: list[SqlTarget]
    gridPos: GridPos
    transformations: list[dict[str, Any]] | None = None

class Time(BaseModel):
    from_: str = Field(default_factory=lambda:"now-30m", alias="from")
    to: str = "now"

class Annotations(BaseModel):
    list_: list[dict[str, Any]] = Field(default_factory=list, alias="list")

class Templating(BaseModel):
    list_: list[dict[str, Any]] = Field(default_factory=list, alias="list")

class Dashboard(BaseModel):
    title: str
    uid: str
    version: int = 1
    refresh: str = "5m"
    timezone: str = "browser"
    time: Time = Field(default_factory=Time)
    panels: list[Stat | TimeSeries]
    templating: Templating = Field(default_factory=Templating)
    annotations: Annotations = Field(default_factory=Annotations)

    def auto_panel_ids(self):
        """Automatically assign panel IDs starting from 1"""
        for i, panel in enumerate(self.panels, 1):
            panel.id = i
        return self

    def to_json(self) -> str:
        """Convert dashboard to JSON string"""
        return json.dumps(self.model_dump(by_alias=True), indent=2)

    def write_to_file(self, filename: str) -> None:
        """Write dashboard JSON to file"""
        with open(filename, 'w') as f:
            f.write(self.to_json())

