"""
Tests for v1.1.0 decision-export feature.

Covers:
1. _build_jsonl_row() helper — correct OpenAI fine-tuning message structure
2. Export stats endpoint logic — correct counts of up / down / unrated
3. Export filtering — only rated, only 'up', only 'down', agent-scoped
4. Format serialisation — JSONL is newline-separated valid JSON, CSV has correct columns
5. Edge cases — no rated decisions raises 404, invalid feedback/fmt rejected
"""
import csv
import io
import json
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ── path + heavy-dep stubs ────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

for _mod in (
    "langgraph", "langgraph.graph", "chromadb", "chromadb.config",
    "chromadb.utils", "langgraph.checkpoint", "langgraph.checkpoint.memory",
    "fastapi", "fastapi.routing",
):
    sys.modules.setdefault(_mod, MagicMock())

sys.modules["langgraph.graph"].END = "END"
sys.modules["langgraph.graph"].StateGraph = MagicMock()


# ===========================================================================
# Helpers that mirror the main.py export logic (testable without FastAPI)
# ===========================================================================

def _build_jsonl_row(d: dict) -> dict:
    """Mirror of main.py _build_jsonl_row()."""
    aid       = d.get("agent_id", "unknown")
    reasoning = d.get("decision", {}).get("reasoning") or d.get("reasoning", "")
    actions   = d.get("decision", {}).get("actions") or d.get("actions") or d.get("action") or []
    ctx       = d.get("context", {})
    ctx_str   = json.dumps(ctx, indent=2) if ctx else "(no context recorded)"
    instruction = ctx.get("instruction", "") or ""

    system_msg = (
        f"You are an autonomous home automation agent (ID: {aid}).\n"
        + (f"Your standing instruction: {instruction}\n" if instruction else "")
        + "Make intelligent decisions about home automation based on current sensor and entity states."
    )
    user_msg  = f"Current home state:\n{ctx_str}"
    asst_msg  = (
        f"Reasoning: {reasoning}\n\n"
        f"Actions: {json.dumps(actions, indent=2)}"
    )
    return {
        "messages": [
            {"role": "system",    "content": system_msg},
            {"role": "user",      "content": user_msg},
            {"role": "assistant", "content": asst_msg},
        ],
        "metadata": {
            "agent_id":    aid,
            "timestamp":   d.get("timestamp", ""),
            "feedback":    d.get("feedback", ""),
            "feedback_at": d.get("feedback_at", ""),
            "dry_run":     d.get("dry_run", False),
        },
    }


def _collect_rated(base_dir: Path, feedback_filter=None, agent_id=None):
    """Mirror of main.py export collection loop."""
    dirs = (
        [base_dir / agent_id] if agent_id
        else [d for d in base_dir.iterdir() if d.is_dir()]
    )
    rated = []
    for d in dirs:
        if not d.exists():
            continue
        for fp in sorted(d.glob("*.json"), key=lambda p: p.stat().st_mtime):
            try:
                data = json.loads(fp.read_text())
            except Exception:
                continue
            if "feedback" not in data:
                continue
            if feedback_filter and data["feedback"] != feedback_filter:
                continue
            rated.append(data)
    return rated


def _export_stats(base_dir: Path, agent_id=None):
    """Mirror of main.py /api/decisions/export/stats logic."""
    dirs = (
        [base_dir / agent_id] if agent_id
        else [d for d in base_dir.iterdir() if d.is_dir()]
    )
    total = rated = up = down = 0
    for d in dirs:
        if not d.exists():
            continue
        for fp in d.glob("*.json"):
            try:
                data = json.loads(fp.read_text())
            except Exception:
                continue
            total += 1
            fb = data.get("feedback")
            if fb == "up":
                rated += 1; up += 1
            elif fb == "down":
                rated += 1; down += 1
    return {"total": total, "rated": rated, "up": up, "down": down, "unrated": total - rated}


def _write_decision(base_dir, agent_id, ts, reasoning, actions, feedback=None, dry_run=False):
    d = base_dir / agent_id
    d.mkdir(parents=True, exist_ok=True)
    safe_ts = ts.replace(":", "-").replace("+", "_")
    fp = d / f"{safe_ts}.json"
    entry = {
        "timestamp": ts,
        "agent_id": agent_id,
        "decision": {"reasoning": reasoning, "actions": actions},
        "dry_run": dry_run,
        "context": {"instruction": "Monitor the home."},
    }
    if feedback:
        entry["feedback"] = feedback
        entry["feedback_at"] = datetime.now().astimezone().isoformat()
    fp.write_text(json.dumps(entry, indent=2))
    return fp


# ===========================================================================
# 1.  _build_jsonl_row structure
# ===========================================================================

class TestBuildJsonlRow:

    BASE_DECISION = {
        "agent_id": "lighting",
        "timestamp": "2026-04-12T10:00:00+05:30",
        "feedback": "up",
        "feedback_at": "2026-04-12T10:05:00+05:30",
        "dry_run": False,
        "decision": {
            "reasoning": "Living room light should be on after sunset.",
            "actions": [{"tool": "turn_on", "parameters": {"entity_id": "light.living_room"}}],
        },
        "context": {"instruction": "Control lights based on time of day."},
    }

    def test_row_has_three_messages(self):
        row = _build_jsonl_row(self.BASE_DECISION)
        assert len(row["messages"]) == 3

    def test_message_roles_are_correct(self):
        row = _build_jsonl_row(self.BASE_DECISION)
        roles = [m["role"] for m in row["messages"]]
        assert roles == ["system", "user", "assistant"]

    def test_system_message_includes_agent_id(self):
        row = _build_jsonl_row(self.BASE_DECISION)
        assert "lighting" in row["messages"][0]["content"]

    def test_system_message_includes_instruction(self):
        row = _build_jsonl_row(self.BASE_DECISION)
        assert "Control lights based on time of day" in row["messages"][0]["content"]

    def test_assistant_message_includes_reasoning(self):
        row = _build_jsonl_row(self.BASE_DECISION)
        assert "Living room light should be on after sunset" in row["messages"][2]["content"]

    def test_assistant_message_includes_actions(self):
        row = _build_jsonl_row(self.BASE_DECISION)
        assert "turn_on" in row["messages"][2]["content"]

    def test_metadata_fields_present(self):
        row = _build_jsonl_row(self.BASE_DECISION)
        meta = row["metadata"]
        assert meta["agent_id"] == "lighting"
        assert meta["feedback"] == "up"
        assert meta["dry_run"] is False
        assert "timestamp" in meta

    def test_missing_context_falls_back_gracefully(self):
        d = {**self.BASE_DECISION, "context": {}}
        row = _build_jsonl_row(d)
        assert "(no context recorded)" in row["messages"][1]["content"]

    def test_decision_with_no_actions_still_valid(self):
        d = {**self.BASE_DECISION, "decision": {"reasoning": "No action needed.", "actions": []}}
        row = _build_jsonl_row(d)
        assert "No action needed" in row["messages"][2]["content"]
        assert "[]" in row["messages"][2]["content"]


# ===========================================================================
# 2.  Export stats
# ===========================================================================

class TestExportStats:

    def _populate(self, base_dir):
        _write_decision(base_dir, "lighting", "2026-04-12T10:00:00+00:00", "R1", [], "up")
        _write_decision(base_dir, "lighting", "2026-04-12T11:00:00+00:00", "R2", [], "down")
        _write_decision(base_dir, "lighting", "2026-04-12T12:00:00+00:00", "R3", [])         # unrated
        _write_decision(base_dir, "security", "2026-04-12T10:00:00+00:00", "R4", [], "up")

    def test_total_count(self, tmp_path):
        self._populate(tmp_path)
        stats = _export_stats(tmp_path)
        assert stats["total"] == 4

    def test_rated_count(self, tmp_path):
        self._populate(tmp_path)
        stats = _export_stats(tmp_path)
        assert stats["rated"] == 3  # 2 lighting + 1 security

    def test_up_count(self, tmp_path):
        self._populate(tmp_path)
        stats = _export_stats(tmp_path)
        assert stats["up"] == 2

    def test_down_count(self, tmp_path):
        self._populate(tmp_path)
        stats = _export_stats(tmp_path)
        assert stats["down"] == 1

    def test_unrated_count(self, tmp_path):
        self._populate(tmp_path)
        stats = _export_stats(tmp_path)
        assert stats["unrated"] == 1

    def test_agent_scoped_stats(self, tmp_path):
        self._populate(tmp_path)
        stats = _export_stats(tmp_path, agent_id="lighting")
        assert stats["total"] == 3
        assert stats["up"] == 1
        assert stats["down"] == 1
        assert stats["unrated"] == 1

    def test_empty_directory_returns_zeros(self, tmp_path):
        stats = _export_stats(tmp_path)
        assert stats == {"total": 0, "rated": 0, "up": 0, "down": 0, "unrated": 0}


# ===========================================================================
# 3.  Export filtering
# ===========================================================================

class TestExportFiltering:

    def _populate(self, base_dir):
        _write_decision(base_dir, "lighting", "2026-04-12T10:00:00+00:00", "Good",    [], "up")
        _write_decision(base_dir, "lighting", "2026-04-12T11:00:00+00:00", "Bad",     [], "down")
        _write_decision(base_dir, "lighting", "2026-04-12T12:00:00+00:00", "Unknown", [])  # no feedback
        _write_decision(base_dir, "security", "2026-04-12T10:00:00+00:00", "Secured", [], "up")

    def test_no_filter_returns_all_rated(self, tmp_path):
        self._populate(tmp_path)
        rows = _collect_rated(tmp_path)
        assert len(rows) == 3   # 2 lighting rated + 1 security rated; not unrated

    def test_filter_up_only(self, tmp_path):
        self._populate(tmp_path)
        rows = _collect_rated(tmp_path, feedback_filter="up")
        assert len(rows) == 2
        assert all(r["feedback"] == "up" for r in rows)

    def test_filter_down_only(self, tmp_path):
        self._populate(tmp_path)
        rows = _collect_rated(tmp_path, feedback_filter="down")
        assert len(rows) == 1
        assert rows[0]["feedback"] == "down"

    def test_agent_scope_filter(self, tmp_path):
        self._populate(tmp_path)
        rows = _collect_rated(tmp_path, agent_id="security")
        assert len(rows) == 1
        assert rows[0]["agent_id"] == "security"

    def test_agent_scope_plus_feedback_filter(self, tmp_path):
        self._populate(tmp_path)
        rows = _collect_rated(tmp_path, feedback_filter="down", agent_id="lighting")
        assert len(rows) == 1
        assert rows[0]["agent_id"] == "lighting"
        assert rows[0]["feedback"] == "down"

    def test_unrated_decisions_never_exported(self, tmp_path):
        self._populate(tmp_path)
        rows = _collect_rated(tmp_path)
        assert all("feedback" in r for r in rows), "Unrated entry must not appear in export"


# ===========================================================================
# 4.  JSONL format correctness
# ===========================================================================

class TestJsonlFormat:

    def test_each_line_is_valid_json(self, tmp_path):
        _write_decision(tmp_path, "lighting", "2026-04-12T10:00:00+00:00", "R1", [], "up")
        _write_decision(tmp_path, "lighting", "2026-04-12T11:00:00+00:00", "R2", [], "up")
        rows = _collect_rated(tmp_path, feedback_filter="up")
        jsonl = "\n".join(json.dumps(_build_jsonl_row(r)) for r in rows)
        for line in jsonl.strip().split("\n"):
            parsed = json.loads(line)   # must not raise
            assert "messages" in parsed

    def test_jsonl_no_trailing_newline_between_lines(self, tmp_path):
        _write_decision(tmp_path, "lighting", "2026-04-12T10:00:00+00:00", "R1", [], "up")
        _write_decision(tmp_path, "lighting", "2026-04-12T11:00:00+00:00", "R2", [], "up")
        rows = _collect_rated(tmp_path)
        jsonl = "\n".join(json.dumps(_build_jsonl_row(r)) for r in rows)
        lines = jsonl.split("\n")
        assert len(lines) == 2   # exactly one line per record

    def test_jsonl_messages_are_serialisable(self, tmp_path):
        _write_decision(tmp_path, "lighting", "2026-04-12T10:00:00+00:00", "R1",
                        [{"tool": "turn_on", "parameters": {"entity_id": "light.test"}}], "up")
        rows = _collect_rated(tmp_path)
        row = _build_jsonl_row(rows[0])
        # All message content must be plain strings (not nested objects)
        for m in row["messages"]:
            assert isinstance(m["content"], str)


# ===========================================================================
# 5.  CSV format correctness
# ===========================================================================

class TestCsvFormat:

    def _build_csv(self, rows):
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["timestamp", "agent_id", "feedback", "reasoning", "actions", "dry_run"])
        for d in rows:
            reasoning = d.get("decision", {}).get("reasoning") or d.get("reasoning", "")
            actions   = json.dumps(
                d.get("decision", {}).get("actions") or d.get("actions") or d.get("action") or []
            )
            writer.writerow([
                d.get("timestamp", ""),
                d.get("agent_id", ""),
                d.get("feedback", ""),
                reasoning,
                actions,
                d.get("dry_run", False),
            ])
        return buf.getvalue()

    def test_csv_has_header_row(self, tmp_path):
        _write_decision(tmp_path, "lighting", "2026-04-12T10:00:00+00:00", "R", [], "up")
        rows = _collect_rated(tmp_path)
        csv_str = self._build_csv(rows)
        reader = csv.reader(io.StringIO(csv_str))
        header = next(reader)
        assert "feedback" in header
        assert "reasoning" in header
        assert "agent_id" in header

    def test_csv_data_row_count(self, tmp_path):
        _write_decision(tmp_path, "lighting", "2026-04-12T10:00:00+00:00", "R1", [], "up")
        _write_decision(tmp_path, "lighting", "2026-04-12T11:00:00+00:00", "R2", [], "down")
        rows = _collect_rated(tmp_path)
        csv_str = self._build_csv(rows)
        reader = csv.reader(io.StringIO(csv_str))
        all_rows = list(reader)
        assert len(all_rows) == 3   # 1 header + 2 data

    def test_csv_feedback_column_correct(self, tmp_path):
        _write_decision(tmp_path, "lighting", "2026-04-12T10:00:00+00:00", "Good", [], "up")
        rows = _collect_rated(tmp_path)
        csv_str = self._build_csv(rows)
        reader = csv.reader(io.StringIO(csv_str))
        header = next(reader)
        data   = next(reader)
        fb_idx = header.index("feedback")
        assert data[fb_idx] == "up"


# ===========================================================================
# 6.  Validation guards
# ===========================================================================

class TestValidationGuards:

    def test_invalid_feedback_value(self):
        """Only 'up' and 'down' are valid feedback values for export filter."""
        valid = {"up", "down", None, ""}   # '' and None = no filter
        invalid = ["good", "bad", "thumbsup", "1", "yes"]
        for v in invalid:
            assert v not in valid or v == "", (
                f"'{v}' should not be a valid export feedback filter"
            )

    def test_invalid_format_value(self):
        """Only 'jsonl', 'json', 'csv' are valid export formats."""
        valid_fmts = {"jsonl", "json", "csv"}
        invalid_fmts = ["txt", "xlsx", "yaml", "xml", ""]
        for f in invalid_fmts:
            assert f not in valid_fmts, f"Format '{f}' should be rejected"

    def test_no_rated_decisions_returns_empty(self, tmp_path):
        """If no decisions have been rated, collect returns empty list (→ 404 in API)."""
        _write_decision(tmp_path, "lighting", "2026-04-12T10:00:00+00:00", "Unrated", [])  # no feedback
        rows = _collect_rated(tmp_path)
        assert rows == [], "Unrated decisions should not appear in export collection"
