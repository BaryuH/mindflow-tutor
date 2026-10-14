import os
import tempfile
import pytest
from persistence.session_memory import SessionMemory


def test_session_memory_persistence_across_restart():
    """Verify that messages survive an in-memory cache clear by reading through SQLite."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_sessions.db")

        # Instance 1: writes messages
        mem1 = SessionMemory(db_path=db_path)
        session_id = "test_user_session_1"
        mem1.add_message(session_id, "user", "What is calculus?", user_id="user_123")
        mem1.add_message(session_id, "assistant", "Calculus is the study of continuous change.", user_id="user_123")

        history1 = mem1.get_history(session_id)
        assert len(history1) == 2
        assert history1[0]["content"] == "What is calculus?"

        # Simulate full server restart (Instance 2 with empty RAM cache)
        mem2 = SessionMemory(db_path=db_path)
        assert session_id not in mem2.sessions  # RAM is empty

        # Read-through should hydrate from persistent store
        history2 = mem2.get_history(session_id)
        assert len(history2) == 2
        assert history2[0]["role"] == "user"
        assert history2[0]["content"] == "What is calculus?"
        assert history2[1]["role"] == "assistant"

        summary = mem2.get_context_summary(session_id)
        assert "What is calculus?" in summary


def test_session_memory_clear():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_sessions.db")
        mem = SessionMemory(db_path=db_path)
        session_id = "to_clear"
        mem.add_message(session_id, "user", "Hello")
        assert len(mem.get_history(session_id)) == 1

        mem.clear_session(session_id)
        assert len(mem.get_history(session_id)) == 0
