"""Self-contained playback. Draw saved outputs; never reimplement control logic."""

import json
from pathlib import Path


def write_complete_review(path: Path, playback: list, manifest: dict):
    template = Path(__file__).with_name("complete_review.html").read_text(encoding="utf-8")
    payload = json.dumps({"playback": playback, "runs": manifest["runs"]}, ensure_ascii=False)
    recording = Path(__file__).with_name("recording.js").read_text(encoding="utf-8")
    template = template.replace("</script></html>", recording + "\n</script></html>")
    path.write_text(template.replace("/*PAYLOAD*/null", payload.replace("<", "\\u003c")),
                    encoding="utf-8")
