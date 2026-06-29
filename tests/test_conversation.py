from pathlib import Path

from app.agent.conversation import ConversationManager
from app.agent.llm_client import LLMClient, LLMMetrics
from app.config import Settings
from app.db import create_session_factory, init_db


def build(tmp_path: Path, **overrides):
    settings = Settings(
        database_url=f"sqlite:///{tmp_path / 'c.db'}",
        water_dry_run=True,
        llm_enabled=False,
        llm_dry_run=True,
        **overrides,
    )
    session_factory = create_session_factory(settings.database_url)
    init_db(session_factory)
    return ConversationManager(session_factory, LLMClient(settings), settings)


def _result(text: str, prompt_tokens: int = 0, tool_calls=None, mission_ids=None) -> dict:
    metrics = LLMMetrics()
    metrics.last_prompt_tokens = prompt_tokens
    return {
        "assistant_response": text,
        "tool_calls": tool_calls or [],
        "created_task_ids": [],
        "created_mission_ids": mission_ids or [],
        "metrics": metrics,
    }


def test_record_and_build_history(tmp_path: Path):
    cm = build(tmp_path)
    sid = cm.create_session()
    cm.record_turn(
        sid,
        "go to kitchen",
        _result(
            "On my way.",
            tool_calls=[{
                "name": "move_to_location",
                "arguments": {"location_name": "Kitchen"},
                "payload": {"requested_target": "Kitchen", "task_id": "abcd1234"},
            }],
        ),
    )
    summary, history = cm.build_history(sid)
    assert summary is None
    assert history[0] == {"role": "user", "content": "go to kitchen"}
    assert history[1]["role"] == "assistant"
    assert "moved to Kitchen" in history[1]["content"]


def test_new_session_is_isolated(tmp_path: Path):
    cm = build(tmp_path)
    s1 = cm.create_session()
    cm.record_turn(s1, "remember this", _result("ok"))
    s2 = cm.create_session()
    summary, history = cm.build_history(s2)
    assert history == [] and summary is None


def test_title_fallback_when_llm_disabled(tmp_path: Path):
    cm = build(tmp_path)
    sid = cm.create_session()
    cm.record_turn(sid, "what is your battery level", _result("Battery is 82%."))
    title = cm.maybe_name_session(sid)
    assert title  # non-empty fallback derived from the first message
    assert cm.maybe_name_session(sid) == title  # idempotent once named


def test_context_fraction_and_should_compact(tmp_path: Path):
    cm = build(tmp_path, llm_context_window=1000, auto_compact_threshold=0.8)
    sid = cm.create_session()
    cm.record_turn(sid, "hi", _result("hello", prompt_tokens=900))
    assert cm.context_fraction(sid) == 0.9
    assert cm.should_compact(sid) is True


def test_compact_reduces_messages_and_sets_summary(tmp_path: Path):
    cm = build(tmp_path, compact_keep_recent_turns=1)
    sid = cm.create_session()
    for i in range(5):
        cm.record_turn(sid, f"msg {i}", _result(f"reply {i}"))
    before = len(cm.get_messages(sid))
    report = cm.compact(sid)
    assert report["compacted"] is True
    after = len(cm.get_messages(sid))
    assert after < before
    summary, history = cm.build_history(sid)
    assert summary  # offline fallback summary is stored
    assert len(history) == 2  # keep_recent_turns=1 -> last 2 messages retained


def test_list_sessions_orders_and_counts(tmp_path: Path):
    cm = build(tmp_path)
    s1 = cm.create_session()
    cm.record_turn(s1, "one", _result("a"))
    sessions = cm.list_sessions()
    assert sessions[0]["session_id"] == s1
    assert sessions[0]["message_count"] == 2
