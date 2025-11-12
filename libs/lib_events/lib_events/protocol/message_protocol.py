from dataclasses import dataclass
from enum import StrEnum

from lib_utils.maybe import Maybe


class MessageType(StrEnum):
    REQUEST = "REQUEST"
    REPLY = "REPLY"
    PUBLISH = "PUBLISH"
    SUBSCRIBE = "SUBSCRIBE"
    REGISTER = "REGISTER"


class ProtocolError(Exception):
    pass


@dataclass
class ParsedMessage:
    destination: str
    msg_type: MessageType
    correlation_id: str
    payload: bytes
    frames: list[bytes]


def build_message(
    destination: str,
    msg_type: MessageType,
    correlation_id: str,
    payload: bytes,
) -> list[bytes]:
    return [
        destination.encode("utf-8"),
        msg_type.value.encode("utf-8"),
        correlation_id.encode("utf-8"),
        payload,
    ]


def parse_message(frames: list[bytes]) -> Maybe[ParsedMessage]:
    if len(frames) != 4:
        return Maybe(None)

    return Maybe.from_try(
        lambda: ParsedMessage(
            destination=frames[0].decode("utf-8"),
            msg_type=MessageType(frames[1].decode("utf-8")),
            correlation_id=frames[2].decode("utf-8"),
            payload=frames[3],
            frames=frames,
        ),
        e=(UnicodeDecodeError, ValueError),
    )
