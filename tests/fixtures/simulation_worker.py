"""Synthetic offline simulator smoke test: no Hermes imports, home, keys or API."""

import json
import os
import sys
from pathlib import Path

from hermes_ariadne.simulation import SourceMessage, simulate_request

home = Path(os.environ["HERMES_HOME"])
assert not home.exists()
history = [
    SourceMessage("old-question", {"role": "user", "content": "Compare A and B."}),
    SourceMessage("old-answer", {"role": "assistant", "content": "A is smaller."}),
    SourceMessage("previous-question", {"role": "user", "content": "Choose A."}),
    SourceMessage("previous-answer", {"role": "assistant", "content": "A chosen."}),
    SourceMessage("current-question", {"role": "user", "content": "Why?"}),
]
result = simulate_request(
    [{"role": "system", "content": "Synthetic host instructions."}],
    history,
    retain_from=2,
    summaries=[
        {
            "node_id": "synthetic-node",
            "event_ids": ["old-question", "old-answer"],
            "text": "Compared two toy choices; A was smaller. Expand for evidence.",
        }
    ],
    budget_units=2000,
    reserved_units=100,
)
assert result.status == "fits"
assert not home.exists()
assert "hermes_ariadne.engine" not in sys.modules
assert not any(name == "agent" or name.startswith("agent.") for name in sys.modules)
print(
    json.dumps(
        {
            "status": result.status,
            "measurement": "canonical ASCII-JSON bytes, not provider tokens",
            "used_units": result.used_units,
            "available_units": result.available_units,
            "archived_event_ids": result.archived_event_ids,
            "retained_event_ids": result.retained_event_ids,
            "hermes_imported": False,
            "home_created": home.exists(),
        },
        sort_keys=True,
    )
)
