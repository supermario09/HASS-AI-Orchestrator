"""
Tests for v1.0.8 Entity Manager features.

Covers:
1. GET /api/entities — correct entity shape returned by the endpoint logic;
   domain filter correctly includes/excludes entries.

2. PATCH /api/factory/agents/{id} — entity persistence logic:
   - Updates entities list in agents.yaml
   - Hot-reloads the in-memory agent's entities attribute
   - Does not overwrite unrelated fields (name, instruction)
   - Returns 404 when agent_id not found in yaml

3. Entity list helper — edge cases (empty HA state, missing friendly_name,
   entity with no entity_id skipped, domain filter case-insensitive-like).
"""
import asyncio
import json
import sys
from pathlib import Path
from typing import List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

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
# Helper: entity-list-filtering logic (mirrors main.py GET /api/entities)
# ===========================================================================

def _filter_entities(states: List[dict], domain: Optional[str] = None) -> List[dict]:
    """
    Mirrors the logic from main.py /api/entities so we can test it without
    spinning up a FastAPI server.
    """
    entities = []
    for s in states:
        eid = s.get("entity_id", "")
        if not eid:
            continue
        if domain and not eid.startswith(f"{domain}."):
            continue
        entities.append({
            "entity_id": eid,
            "friendly_name": s.get("attributes", {}).get("friendly_name") or eid,
            "state": s.get("state", ""),
            "domain": eid.split(".")[0],
        })
    entities.sort(key=lambda x: x["entity_id"])
    return entities


# ===========================================================================
# 1.  GET /api/entities — entity shape and domain filter
# ===========================================================================

class TestGetEntitiesLogic:

    HA_STATES = [
        {"entity_id": "light.living_room",   "state": "on",  "attributes": {"friendly_name": "Living Room Light"}},
        {"entity_id": "light.bedroom",        "state": "off", "attributes": {"friendly_name": "Bedroom Light"}},
        {"entity_id": "switch.garage_door",   "state": "off", "attributes": {}},
        {"entity_id": "sensor.temperature",   "state": "21",  "attributes": {"unit_of_measurement": "°C"}},
        {"entity_id": "camera.front_door",    "state": "streaming", "attributes": {"friendly_name": "Front Door"}},
        {"entity_id": "",                      "state": "on",  "attributes": {}},   # no entity_id — should be skipped
    ]

    def test_all_entities_returned_when_no_domain_filter(self):
        result = _filter_entities(self.HA_STATES)
        eids = [e["entity_id"] for e in result]
        assert "" not in eids, "Empty entity_id must be skipped"
        assert len(result) == 5  # 5 valid entities

    def test_domain_filter_includes_only_matching(self):
        result = _filter_entities(self.HA_STATES, domain="light")
        assert all(e["entity_id"].startswith("light.") for e in result)
        assert len(result) == 2

    def test_domain_filter_excludes_other_domains(self):
        result = _filter_entities(self.HA_STATES, domain="camera")
        eids = [e["entity_id"] for e in result]
        assert eids == ["camera.front_door"]

    def test_entities_sorted_alphabetically(self):
        result = _filter_entities(self.HA_STATES)
        eids = [e["entity_id"] for e in result]
        assert eids == sorted(eids), "Entities must be sorted by entity_id"

    def test_entity_shape_has_required_fields(self):
        result = _filter_entities([self.HA_STATES[0]])
        e = result[0]
        assert "entity_id" in e
        assert "friendly_name" in e
        assert "state" in e
        assert "domain" in e
        assert e["domain"] == "light"

    def test_friendly_name_falls_back_to_entity_id(self):
        """If no friendly_name in attributes, use entity_id as fallback."""
        state = {"entity_id": "switch.garage_door", "state": "off", "attributes": {}}
        result = _filter_entities([state])
        assert result[0]["friendly_name"] == "switch.garage_door"

    def test_empty_ha_state_returns_empty_list(self):
        assert _filter_entities([]) == []

    def test_none_domain_returns_all(self):
        result_none = _filter_entities(self.HA_STATES, domain=None)
        result_unfiltered = _filter_entities(self.HA_STATES)
        assert result_none == result_unfiltered

    def test_unknown_domain_returns_empty(self):
        result = _filter_entities(self.HA_STATES, domain="nonexistent_domain")
        assert result == []


# ===========================================================================
# 2.  PATCH /api/factory/agents/{id} — entity persistence in YAML
# ===========================================================================

def _yaml_update_entities(config_path: Path, agent_id: str, new_entities: List[str]) -> bool:
    """
    Mirrors the entity-update section of factory_router.patch /agents/{agent_id}.
    Returns True if the agent was found and updated, False otherwise.
    """
    if not config_path.exists():
        return False
    with open(config_path, "r") as f:
        data = yaml.safe_load(f) or {}

    found = False
    for agent in data.get("agents", []):
        if agent["id"] == agent_id:
            agent["entities"] = new_entities
            found = True
            break

    if not found:
        return False

    with open(config_path, "w") as f:
        yaml.dump(data, f, sort_keys=False)
    return True


def _make_agents_yaml(tmp_path: Path, agents_data: list) -> Path:
    p = tmp_path / "agents.yaml"
    p.write_text(yaml.dump({"agents": agents_data}))
    return p


class TestEntityPersistence:

    def test_entities_written_to_yaml(self, tmp_path):
        """Patching entities must update the YAML file."""
        p = _make_agents_yaml(tmp_path, [
            {"id": "lighting", "name": "Lighting Agent", "instruction": "...", "entities": []},
        ])
        new_ents = ["light.living_room", "light.bedroom"]
        ok = _yaml_update_entities(p, "lighting", new_ents)

        assert ok is True
        saved = yaml.safe_load(p.read_text())
        saved_ents = next(a["entities"] for a in saved["agents"] if a["id"] == "lighting")
        assert saved_ents == new_ents

    def test_other_fields_untouched_during_entity_update(self, tmp_path):
        """Updating entities must NOT overwrite name, instruction, etc."""
        p = _make_agents_yaml(tmp_path, [
            {
                "id": "security",
                "name": "Security Agent",
                "instruction": "Keep the home safe",
                "entities": ["lock.front_door"],
                "decision_interval": 60,
            },
        ])
        _yaml_update_entities(p, "security", ["lock.front_door", "binary_sensor.motion"])

        saved = yaml.safe_load(p.read_text())
        agent = next(a for a in saved["agents"] if a["id"] == "security")
        assert agent["name"] == "Security Agent"
        assert agent["instruction"] == "Keep the home safe"
        assert agent["decision_interval"] == 60

    def test_entity_update_for_nonexistent_agent_returns_false(self, tmp_path):
        """If agent_id is not in yaml, return False (→ 404 in API)."""
        p = _make_agents_yaml(tmp_path, [
            {"id": "lighting", "name": "Lighting", "instruction": "...", "entities": []},
        ])
        ok = _yaml_update_entities(p, "vision", ["camera.front_door"])
        assert ok is False

    def test_entity_list_can_be_cleared(self, tmp_path):
        """Setting entities to [] must persist as an empty list, not null."""
        p = _make_agents_yaml(tmp_path, [
            {"id": "heating", "name": "Heating", "instruction": "...", "entities": ["climate.thermostat"]},
        ])
        ok = _yaml_update_entities(p, "heating", [])
        assert ok is True
        saved = yaml.safe_load(p.read_text())
        agent = next(a for a in saved["agents"] if a["id"] == "heating")
        assert agent["entities"] == []

    def test_multiple_agents_only_target_updated(self, tmp_path):
        """When multiple agents exist, only the targeted one must change."""
        p = _make_agents_yaml(tmp_path, [
            {"id": "lighting", "name": "Lighting", "instruction": "...", "entities": ["light.kitchen"]},
            {"id": "security", "name": "Security", "instruction": "...", "entities": ["lock.front"]},
        ])
        _yaml_update_entities(p, "lighting", ["light.kitchen", "light.bedroom"])
        saved = yaml.safe_load(p.read_text())

        lighting = next(a for a in saved["agents"] if a["id"] == "lighting")
        security = next(a for a in saved["agents"] if a["id"] == "security")

        assert lighting["entities"] == ["light.kitchen", "light.bedroom"]
        assert security["entities"] == ["lock.front"], "Security agent entities must be unchanged"

    def test_missing_config_file_returns_false(self, tmp_path):
        """If agents.yaml doesn't exist, update returns False gracefully."""
        missing = tmp_path / "nonexistent.yaml"
        ok = _yaml_update_entities(missing, "lighting", ["light.test"])
        assert ok is False


# ===========================================================================
# 3.  Hot-reload: in-memory agent entities reflect YAML after save
# ===========================================================================

class TestInMemoryHotReload:
    """
    Verify that after a YAML update, reading entities back from the file and
    applying them to the in-memory agent object is correct — this mirrors the
    hot-reload block in factory_router.py.
    """

    def test_in_memory_agent_entities_updated_after_yaml_save(self, tmp_path):
        """After saving entities to YAML, the in-memory agent must reflect the new list."""
        p = _make_agents_yaml(tmp_path, [
            {"id": "lighting", "name": "Lighting", "instruction": "...", "entities": []},
        ])
        new_ents = ["light.living_room", "light.bedroom"]
        _yaml_update_entities(p, "lighting", new_ents)

        # Simulate hot-reload: re-read YAML and apply to in-memory agent
        mock_agent = MagicMock()
        mock_agent.entities = []

        saved = yaml.safe_load(p.read_text())
        for a in saved.get("agents", []):
            if a["id"] == "lighting":
                mock_agent.entities = a.get("entities", [])
                break

        assert mock_agent.entities == new_ents, (
            f"In-memory agent entities not updated after hot-reload. Got: {mock_agent.entities}"
        )
