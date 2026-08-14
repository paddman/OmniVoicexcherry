from omnivoice.cli.train_gui import parse_transcripts, split_records


def test_parse_transcript_with_session():
    parsed = parse_transcripts("001.wav|สวัสดี|th|session-a", 1)
    assert parsed["001.wav"]["text"] == "สวัสดี"
    assert parsed["001.wav"]["session_id"] == "session-a"


def test_session_split_avoids_leakage():
    rows = [
        {"id": f"a-{index}", "session_id": "a"} for index in range(3)
    ] + [
        {"id": f"b-{index}", "session_id": "b"} for index in range(3)
    ]
    train, dev = split_records(rows, 20, 42)
    train_sessions = {row["session_id"] for row in train}
    dev_sessions = {row["session_id"] for row in dev}
    assert train_sessions.isdisjoint(dev_sessions)
    assert train and dev
