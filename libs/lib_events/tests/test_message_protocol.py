from lib_events.protocol.message_protocol import MessageType, build_message, parse_message


def test_build_message_all_frames():
    """
    Build a complete 4-frame message with all fields populated
    """
    frames = build_message(
        destination="corerl-reactor1",
        msg_type=MessageType.REQUEST,
        correlation_id="abc123",
        payload=b'{"command": "pause"}',
    )

    assert len(frames) == 4
    assert frames[0] == b"corerl-reactor1"
    assert frames[1] == b"REQUEST"
    assert frames[2] == b"abc123"
    assert frames[3] == b'{"command": "pause"}'


def test_build_message_empty_correlation_id():
    """
    Build message with empty correlation ID (for PUBLISH/SUBSCRIBE)
    """
    frames = build_message(
        destination="corerl",
        msg_type=MessageType.PUBLISH,
        correlation_id="",
        payload=b'{"event": "started"}',
    )

    assert len(frames) == 4
    assert frames[2] == b""


def test_build_message_all_types():
    """
    Verify all MessageType values can be built
    """
    for msg_type in MessageType:
        frames = build_message(
            destination="test",
            msg_type=msg_type,
            correlation_id="",
            payload=b"test",
        )
        assert frames[1] == msg_type.value.encode("utf-8")


def test_parse_message_valid():
    """
    Parse a valid 4-frame message
    """
    frames = [
        b"corerl-reactor1",
        b"REQUEST",
        b"abc123",
        b'{"command": "pause"}',
    ]

    parsed = parse_message(frames)
    assert parsed.is_some()
    msg = parsed.expect()

    assert msg.destination == "corerl-reactor1"
    assert msg.msg_type == MessageType.REQUEST
    assert msg.correlation_id == "abc123"
    assert msg.payload == b'{"command": "pause"}'


def test_parse_message_empty_correlation_id():
    """
    Parse message with empty correlation ID
    """
    frames = [
        b"corerl",
        b"PUBLISH",
        b"",
        b'{"event": "started"}',
    ]

    parsed = parse_message(frames)
    assert parsed.is_some()
    msg = parsed.expect()

    assert msg.destination == "corerl"
    assert msg.msg_type == MessageType.PUBLISH
    assert msg.correlation_id == ""
    assert msg.payload == b'{"event": "started"}'


def test_parse_message_all_types():
    """
    Parse messages for all MessageType values
    """
    for msg_type in MessageType:
        frames = [
            b"test",
            msg_type.value.encode("utf-8"),
            b"",
            b"test",
        ]
        parsed = parse_message(frames)
        assert parsed.is_some()
        msg = parsed.expect()
        assert msg.msg_type == msg_type


def test_parse_message_too_few_frames():
    """
    Return None when fewer than 4 frames provided
    """
    frames = [b"destination", b"REQUEST", b"abc123"]

    parsed = parse_message(frames)
    assert parsed.is_none()


def test_parse_message_too_many_frames():
    """
    Return None when more than 4 frames provided
    """
    frames = [b"destination", b"REQUEST", b"abc123", b"payload", b"extra"]

    parsed = parse_message(frames)
    assert parsed.is_none()


def test_parse_message_invalid_message_type():
    """
    Return None for invalid message type
    """
    frames = [
        b"destination",
        b"INVALID_TYPE",
        b"abc123",
        b"payload",
    ]

    parsed = parse_message(frames)
    assert parsed.is_none()


def test_parse_message_invalid_utf8_destination():
    """
    Return None for non-UTF-8 destination
    """
    frames = [
        b"\xff\xfe",
        b"REQUEST",
        b"abc123",
        b"payload",
    ]

    parsed = parse_message(frames)
    assert parsed.is_none()


def test_parse_message_invalid_utf8_correlation_id():
    """
    Return None for non-UTF-8 correlation ID
    """
    frames = [
        b"destination",
        b"REQUEST",
        b"\xff\xfe",
        b"payload",
    ]

    parsed = parse_message(frames)
    assert parsed.is_none()


def test_roundtrip():
    """
    Build and parse a message, verifying roundtrip consistency
    """
    destination = "corerl-reactor1"
    msg_type = MessageType.REQUEST
    correlation_id = "abc123"
    payload = b'{"command": "pause"}'

    frames = build_message(destination, msg_type, correlation_id, payload)
    parsed = parse_message(frames)
    assert parsed.is_some()
    msg = parsed.expect()

    assert msg.destination == destination
    assert msg.msg_type == msg_type
    assert msg.correlation_id == correlation_id
    assert msg.payload == payload


def test_unicode_destination():
    """
    Handle Unicode characters in destination
    """
    destination = "corerl-reaktör-1"
    frames = build_message(
        destination=destination,
        msg_type=MessageType.REQUEST,
        correlation_id="",
        payload=b"test",
    )

    parsed = parse_message(frames)
    assert parsed.is_some()
    msg = parsed.expect()
    assert msg.destination == destination


def test_binary_payload():
    """
    Handle arbitrary binary data in payload
    """
    payload = bytes(range(256))
    frames = build_message(
        destination="test",
        msg_type=MessageType.REQUEST,
        correlation_id="",
        payload=payload,
    )

    parsed = parse_message(frames)
    assert parsed.is_some()
    msg = parsed.expect()
    assert msg.payload == payload
