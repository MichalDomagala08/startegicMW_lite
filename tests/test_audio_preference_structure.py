# Non-importing smoke test for AudioTrialPreference structure
# Reads the source file and checks for key class/method occurrences.

from pathlib import Path
import re

SRC = Path(__file__).resolve().parents[1] / "Experiment_Natalia" / "classes" / "Audio.py"
text = SRC.read_text(encoding="utf-8")

assert "class AudioTrialPreference" in text, "AudioTrialPreference definition missing"
assert re.search(r"def\s+get_linear_scale_value_Preference\s*\(", text), "get_linear_scale_value_Preference missing"
assert re.search(r"class\s+Recording\s*\(", text) or re.search(r"class\s+Recording\s*:\", text), "Recording subclass missing"
assert re.search(r"def\s+run\s*\(", text), "run() seems missing"

print("SMOKE-OK: AudioTrialPreference skeleton and helpers present in Audio.py")
