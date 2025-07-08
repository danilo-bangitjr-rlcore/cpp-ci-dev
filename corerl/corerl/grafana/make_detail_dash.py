from corerl.grafana.dashlib import (
    Dashboard,
    Datasource,
    GridPos,
    ReduceOptions,
    SqlTarget,
    Stat,
    StatOptions,
    TimeSeries,
)


def make_buffer_size_panel(datasource: Datasource):
    return Stat(
        title="Buffer size",
        datasource=datasource,
        targets=[
            SqlTarget(
                datasource=datasource,
                rawSql="SELECT value, time, metric\n"\
                       "FROM metrics\n"\
                       "WHERE $__timeFilter(\"time\")\n"\
                       "AND metric='buffer_critic[0]_size'\n"\
                       "ORDER BY \"time\"",
                refId="A",
            ),
        ],
        gridPos=GridPos(h=3, w=2, x=0, y=0),
    )


def make_agent_step_panel(datasource: Datasource):
    return Stat(
        title="Agent step",
        datasource=datasource,
        targets=[
            SqlTarget(
                datasource=datasource,
                rawSql="SELECT time, agent_step\n"\
                       "FROM metrics\n"\
                       "WHERE $__timeFilter(\"time\")\n"\
                       "ORDER BY \"time\" DESC\n"\
                       "LIMIT 1",
                refId="A",
            ),
        ],
        gridPos=GridPos(h=3, w=2, x=2, y=0),
    )


def make_goal_satisfied_panel(datasource: Datasource):
    return Stat(
        title="Goal is satisfied",
        datasource=datasource,
        targets=[
            SqlTarget(
                datasource=datasource,
                rawSql="SELECT time, value > -0.5 as goals_satisfied\n"\
                       "FROM metrics\n"\
                       "WHERE $__timeFilter(\"time\")\n"\
                       "AND metric LIKE 'reward%'\n"\
                       "ORDER BY \"time\" DESC\n"\
                       "LIMIT 1",
                refId="A",
            ),
        ],
        gridPos=GridPos(h=3, w=3, x=4, y=0),
        options=StatOptions(
            reduceOptions=ReduceOptions(
                calcs=["lastNotNull"],
                fields="goals_satisfied",
            ),
        ),
    )

def make_reward_panel(datasource: Datasource):
    return TimeSeries(
        title="Reward",
        datasource=datasource,
        targets=[
            SqlTarget(
                datasource=datasource,
                rawSql="SELECT value, time, metric\n"\
                       "FROM metrics\n"\
                       "WHERE $__timeFilter(\"time\")\n"\
                       "AND metric LIKE 'reward%'\n"\
                       "ORDER BY \"time\"",
                refId="A",
                format="table",
                rawQuery=True,
            ),
        ],
        gridPos=GridPos(h=9, w=14, x=10, y=3),
    )

def make_action_panel(datasource: Datasource, action_name: str):
    return TimeSeries(
        title="Action Bounds",
        datasource=datasource,
        targets=[
            SqlTarget(
                datasource=datasource,
                rawSql="SELECT value, time, metric\n"\
                       "FROM metrics\n"\
                       "WHERE $__timeFilter(\"time\")\n"\
                       f"AND (metric LIKE '%{action_name}%-lo'\n"\
                       f"OR metric LIKE '%{action_name}%-hi'\n"\
                       f"OR metric LIKE 'ACTION-{action_name}%')\n"\
                       "ORDER BY \"time\"",
                refId="A",
                format="table",
                rawQuery=True,
            ),
        ],
        gridPos=GridPos(h=8, w=14, x=10, y=20),
        transformations=[{
            "id": "prepareTimeSeries",
            "options": {
                "format": "multi",
            }},
        ],
    )


def create_coag_dashboard():
    panels = []

    datasource = Datasource(
        type="grafana-postgresql-datasource",
        uid="fe9zkc0takwlcc",
    )

    panels.append(make_buffer_size_panel(datasource))
    panels.append(make_agent_step_panel(datasource))
    panels.append(make_goal_satisfied_panel(datasource))
    panels.append(make_reward_panel(datasource))
    panels.append(make_action_panel(datasource, 'action-0'))

    return Dashboard(
        title="Detail Dashboard",
        uid="fdsqy8vove874f",
        panels=panels,
    ).auto_panel_ids()

# Generate the dashboard JSON
dashboard = create_coag_dashboard()
dashboard.write_to_file('out.json')
