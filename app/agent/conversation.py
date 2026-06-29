from __future__ import annotations

from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import sessionmaker

from app.agent.llm_client import LLMClient
from app.agent.prompts import build_compaction_prompt, build_title_prompt
from app.config import Settings
from app.models import ChatMessage, ChatRole, ChatSession, ChatSessionStatus, utcnow
from app.water.schemas import WaterClientError


class ConversationManager:
    """Persists chats (sessions + messages) and provides conversational memory.

    Only the dialogue is stored — the system prompt and live robot runtime
    context are rebuilt fresh each turn by the planner. Older turns can be
    compacted into a per-session summary to stay within the context window.
    """

    def __init__(self, session_factory: sessionmaker, llm_client: LLMClient, settings: Settings):
        self.session_factory = session_factory
        self.llm_client = llm_client
        self.settings = settings

    # --- session lifecycle -------------------------------------------------

    def create_session(self) -> str:
        session = self.session_factory()
        try:
            record = ChatSession(status=ChatSessionStatus.ACTIVE.value)
            session.add(record)
            session.commit()
            return record.id
        finally:
            session.close()

    def _session_dict(self, db, record: ChatSession) -> dict[str, Any]:
        count = db.scalar(
            select(func.count()).select_from(ChatMessage).where(ChatMessage.session_id == record.id)
        ) or 0
        return {
            "session_id": record.id,
            "title": record.title,
            "status": record.status,
            "updated_at": record.updated_at.isoformat() if record.updated_at else None,
            "message_count": int(count),
            "has_summary": bool(record.summary),
            "last_prompt_tokens": record.last_prompt_tokens,
        }

    def list_sessions(self, limit: int = 20) -> list[dict[str, Any]]:
        db = self.session_factory()
        try:
            rows = db.scalars(
                select(ChatSession).order_by(ChatSession.updated_at.desc()).limit(limit)
            ).all()
            return [self._session_dict(db, r) for r in rows]
        finally:
            db.close()

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        db = self.session_factory()
        try:
            record = db.get(ChatSession, session_id)
            return self._session_dict(db, record) if record else None
        finally:
            db.close()

    def get_messages(self, session_id: str) -> list[dict[str, Any]]:
        db = self.session_factory()
        try:
            rows = db.scalars(
                select(ChatMessage).where(ChatMessage.session_id == session_id).order_by(ChatMessage.seq)
            ).all()
            return [{"role": m.role, "content": m.content, "actions": m.actions or []} for m in rows]
        finally:
            db.close()

    # --- building context for the planner ----------------------------------

    def build_history(self, session_id: str) -> tuple[str | None, list[dict[str, Any]]]:
        """Return (summary, prior-turn messages) to inject before the new message."""
        db = self.session_factory()
        try:
            record = db.get(ChatSession, session_id)
            summary = record.summary if record else None
            rows = db.scalars(
                select(ChatMessage).where(ChatMessage.session_id == session_id).order_by(ChatMessage.seq)
            ).all()
            history: list[dict[str, Any]] = []
            for m in rows:
                content = m.content or ""
                if m.role == ChatRole.ASSISTANT.value and m.actions:
                    note = f"[actions: {'; '.join(m.actions)}]"
                    content = f"{content}\n{note}" if content else note
                history.append({"role": m.role, "content": content})
            return summary, history
        finally:
            db.close()

    # --- recording a turn ---------------------------------------------------

    @staticmethod
    def actions_from_result(result: dict[str, Any]) -> list[str]:
        actions: list[str] = []
        for call in result.get("tool_calls", []) or []:
            name = call.get("name")
            args = call.get("arguments") or {}
            payload = call.get("payload") or {}
            target = payload.get("requested_target") or args.get("location_name") or payload.get("marker_name")
            if name == "move_to_location" and target:
                actions.append(f"moved to {target}")
            elif name == "return_to_charger":
                actions.append(f"returned to charger {target}".strip())
            elif name == "create_mission":
                actions.append("created a mission")
            elif name == "cancel_current_task":
                actions.append("canceled current move")
            elif name == "emergency_stop":
                actions.append("emergency stop")
            elif name == "release_emergency_stop_confirmed":
                actions.append("released e-stop")
            elif name:
                actions.append(str(name))
            if payload.get("error"):
                actions.append(f"{name} failed")
        for mid in result.get("created_mission_ids", []) or []:
            actions.append(f"mission {mid[:8]}")
        return actions

    def record_turn(self, session_id: str, user_message: str, result: dict[str, Any]) -> None:
        db = self.session_factory()
        try:
            record = db.get(ChatSession, session_id)
            if record is None:
                return
            base = db.scalar(
                select(func.max(ChatMessage.seq)).where(ChatMessage.session_id == session_id)
            ) or 0
            db.add(ChatMessage(session_id=session_id, seq=base + 1, role=ChatRole.USER.value, content=user_message))
            actions = self.actions_from_result(result)
            db.add(
                ChatMessage(
                    session_id=session_id,
                    seq=base + 2,
                    role=ChatRole.ASSISTANT.value,
                    content=result.get("assistant_response", "") or "",
                    actions=actions or None,
                )
            )
            metrics = result.get("metrics")
            if metrics is not None:
                tokens = int(getattr(metrics, "last_prompt_tokens", 0) or 0)
                if tokens:
                    record.last_prompt_tokens = tokens
            record.updated_at = utcnow()
            db.commit()
        finally:
            db.close()

    # --- auto-naming --------------------------------------------------------

    @staticmethod
    def _clean_title(raw: str) -> str:
        title = (raw or "").strip().splitlines()[0] if raw and raw.strip() else ""
        title = title.strip().strip('"').strip("'").strip()
        if title.lower().startswith("title:"):
            title = title[6:].strip()
        return title[:48]

    @staticmethod
    def _fallback_title(first_user: str) -> str:
        words = (first_user or "").strip().split()
        return " ".join(words[:6])[:48].title() or "New chat"

    def maybe_name_session(self, session_id: str) -> str | None:
        db = self.session_factory()
        try:
            record = db.get(ChatSession, session_id)
            if record is None or record.title:
                return record.title if record else None
            rows = db.scalars(
                select(ChatMessage).where(ChatMessage.session_id == session_id).order_by(ChatMessage.seq).limit(2)
            ).all()
            first_user = next((m.content for m in rows if m.role == ChatRole.USER.value), "")
            first_assistant = next((m.content for m in rows if m.role == ChatRole.ASSISTANT.value), "")
            if not first_user:
                return None
            title = ""
            try:
                title = self._clean_title(self.llm_client.complete(build_title_prompt(first_user, first_assistant), max_tokens=24))
            except WaterClientError:
                title = ""
            record.title = title or self._fallback_title(first_user)
            db.commit()
            return record.title
        finally:
            db.close()

    # --- compaction ---------------------------------------------------------

    def context_fraction(self, session_id: str) -> float:
        db = self.session_factory()
        try:
            record = db.get(ChatSession, session_id)
            tokens = record.last_prompt_tokens if record else 0
            return tokens / max(1, self.settings.llm_context_window)
        finally:
            db.close()

    def should_compact(self, session_id: str) -> bool:
        return self.context_fraction(session_id) >= self.settings.auto_compact_threshold

    @staticmethod
    def _messages_to_text(rows: list[ChatMessage]) -> str:
        parts = []
        for m in rows:
            line = f"{m.role}: {m.content or ''}"
            if m.role == ChatRole.ASSISTANT.value and m.actions:
                line += f" [actions: {'; '.join(m.actions)}]"
            parts.append(line)
        return "\n".join(parts)

    def compact(self, session_id: str, instructions: str | None = None) -> dict[str, Any]:
        db = self.session_factory()
        try:
            record = db.get(ChatSession, session_id)
            if record is None:
                return {"compacted": False, "reason": "no session"}
            rows = db.scalars(
                select(ChatMessage).where(ChatMessage.session_id == session_id).order_by(ChatMessage.seq)
            ).all()
            keep = max(0, self.settings.compact_keep_recent_turns) * 2  # user+assistant per turn
            if len(rows) <= keep + 1:
                return {"compacted": False, "reason": "not enough history", "messages": len(rows)}
            older = rows[: len(rows) - keep] if keep else rows
            recent = rows[len(rows) - keep:] if keep else []
            before_fraction = (record.last_prompt_tokens or 0) / max(1, self.settings.llm_context_window)

            prior = f"Earlier summary:\n{record.summary}\n\n" if record.summary else ""
            history_text = prior + self._messages_to_text(older)
            new_summary = ""
            try:
                new_summary = self.llm_client.complete(build_compaction_prompt(history_text, instructions), max_tokens=512)
            except WaterClientError:
                new_summary = ""
            if not new_summary:
                # Offline/dry fallback: keep the prior summary plus a truncated tail.
                new_summary = (prior + self._messages_to_text(older))[:1500]

            record.summary = new_summary
            for m in older:
                db.delete(m)
            record.updated_at = utcnow()
            db.commit()
            return {
                "compacted": True,
                "removed": len(older),
                "kept": len(recent),
                "before_fraction": before_fraction,
            }
        finally:
            db.close()
