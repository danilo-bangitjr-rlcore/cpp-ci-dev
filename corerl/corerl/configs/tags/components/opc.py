from enum import StrEnum, auto

from lib_config.config import config


class Agg(StrEnum):
    avg = auto()
    last = auto()
    bool_or = auto()


@config()
class OPCTag:
    agg: Agg = Agg.avg
    """
    Kind: internal

    The temporal aggregation strategy used when querying timescale db. For most tags,
    this should be Agg.avg. For setpoints, this should be Agg.last.
    """

    connection_id: str | None = None
    """
    Kind: required external

    The UUID associated to the OPC-UA server connection, used within the CoreIO thin client.
    """

    node_identifier: str | None = None
    """
    Kind: required for ai_setpoint, external

    The full OPC-UA node identifier string (e.g. ns=#;i=?). This is used by coreio in
    communication with the OPC server.
    """

    dtype: str = 'float'
    """
    Kind: optional external

    The datatype of the OPC data. Typically this will just be a float. In rare cases, this
    may be a boolean, integer, or string.
    """
