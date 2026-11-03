"""Conservative embedded-sequence replacement for matched TNT/TPS pairs."""
from pathlib import Path


def replace_sequence(template_tnt: Path, template_tps: Path, edited_tps: bytes, output: Path) -> Path:
    standalone = b"PSEQ1.001.18 BIN"
    embedded = b"PSEQ" + (1).to_bytes(4, "little") + b"1.18 BIN"
    old = template_tps.read_bytes()
    if not old.startswith(standalone) or not edited_tps.startswith(standalone):
        raise ValueError("Matched-pair TNT editing requires the supported TPS signature")
    original = embedded + old[16:]
    replacement = embedded + edited_tps[16:]
    container = template_tnt.read_bytes()
    if container.count(original) != 1:
        raise ValueError("TNT must contain exactly one matching template TPS payload")
    if output.resolve() in {template_tnt.resolve(), template_tps.resolve()}:
        raise ValueError("Write the edited TNT to a separate output path")
    output.write_bytes(container.replace(original, replacement, 1))
    return output
