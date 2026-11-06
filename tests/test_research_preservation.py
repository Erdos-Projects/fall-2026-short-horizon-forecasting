"""Guard the original research content and the published-result provenance."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_original_cells_outputs_and_figures_are_preserved():
    manifest = json.loads((ROOT / "results/research_preservation.json").read_text())
    assert len(manifest["cells"]) == 148
    assert sum(row["png_figures"] for row in manifest["cells"]) == 20
    archives = {}
    for row in manifest["cells"]:
        path = row["source_path"]
        if path not in archives:
            archives[path] = json.loads((ROOT / path).read_text())
        cell = archives[path]["cells"][row["source_cell"]]
        assert hashlib.sha256("".join(cell["source"]).encode()).hexdigest() == row["source_sha256"]
        outputs = cell.get("outputs", [])
        assert hashlib.sha256(json.dumps(outputs, sort_keys=True).encode()).hexdigest() == row["outputs_sha256"]
        assert len(outputs) == row["saved_outputs"]
        assert sum("image/png" in out.get("data", {}) for out in outputs) == row["png_figures"]
        for dest in row["visible_in"]:
            copied = json.loads((ROOT / dest["path"]).read_text())["cells"][dest["cell_index"]]
            source = "".join(cell["source"])
            if dest["adaptation"]:
                source = source.replace("Eloise", "Nhi Dinh")
            assert "".join(copied["source"]) == source
            assert copied.get("outputs", []) == outputs


def test_published_tables_point_to_exact_committed_results():
    notebook = json.loads((ROOT / "notebooks/01_forecasting_and_trading.ipynb").read_text())
    snapshots = [c for c in notebook["cells"] if "result_snapshot" in c.get("metadata", {})]
    assert len(snapshots) == 6
    for cell in snapshots:
        ref = cell["metadata"]["result_snapshot"]
        assert hashlib.sha256((ROOT / ref["path"]).read_bytes()).hexdigest() == ref["sha256"]
        assert cell["outputs"]
