"""Public workflow API. Configuration checks never connect to an instrument."""
from __future__ import annotations

from pathlib import Path

from hardware_capture.configuration import Experiment, positive
from hardware_capture.pulses.conditions import generate
from hardware_capture.workflows import common, calibrate, analyze, sequence, tnmr, optimize

PROJECTS = {
    "capture": "Save raw RF from the scope or copy a compatible capture CSV",
    "calibrate-square": "Calibrate volts-to-Rabi from a known square pulse",
    "calibrate-bb1": "Calibrate volts-to-Rabi from a complete BB1 envelope",
    "analyze": "Analyze a capture or generated pulse using an explicit gate target",
    "pulse": "Generate/export a pulse from shared physical conditions",
    "sequence": "Inspect/edit TPS and matched TNT files offline",
    "tnmr": "Inspect or operate TNMR through COM",
    "phase-reference": "Measure a BB1 phase reference with TNMR and scope",
    "optimize-offline": "Optimize X(pi) in an explicit simulation model",
    "optimize-live": "Optimize X(pi) using measured TNMR/scope captures",
}


def _config(config) -> Experiment:
    if isinstance(config, Experiment):
        return config
    if isinstance(config, dict):
        return Experiment(config)
    return Experiment.load(config)


def plan(project: str, config) -> dict:
    """Validate input contracts and describe effects without opening devices."""
    config = _config(config)
    if project not in PROJECTS:
        raise ValueError(f"Unknown project {project!r}; choose {', '.join(PROJECTS)}")
    source = config.section("input").get("kind", "csv")
    raw_projects = {"capture", "calibrate-square", "calibrate-bb1", "analyze"}
    requires_hardware = project in {"tnmr", "phase-reference", "optimize-live"} or (project in raw_projects and source == "scope")
    if requires_hardware:
        for name in ["timeout_ms", "trigger_timeout_s", "armed_timeout_s", "poll_s"]:
            value = config.section("scope").get(name)
            if value is not None:
                positive(value, "scope." + name)
    if project == "capture":
        shots = config.section("input").get("shots", 1)
        if isinstance(shots, bool) or not isinstance(shots, int) or shots < 1:
            raise ValueError("input.shots must be a positive integer")
    if project in raw_projects:
        if source == "csv":
            path = Path(config.require("input", "path"))
            if not path.is_file():
                raise ValueError(f"Input capture does not exist: {path}")
        elif source == "scope":
            config.require("scope", "resource")
            common.capture_config(config)
        elif source == "pulse" and project == "analyze":
            generate(config)
        else:
            raise ValueError(f"Unsupported input.kind={source!r} for {project}")
        if project != "capture":
            demod = config.section("demodulation")
            order = demod.get("filter_order", 4)
            if isinstance(order, bool) or not isinstance(order, int) or order < 1:
                raise ValueError("demodulation.filter_order must be a positive integer")
            if demod.get("q_sign", -1) not in (-1, 1):
                raise ValueError("demodulation.q_sign must be +1 or -1")
            positive(demod.get("cutoff_hz", 25e6), "demodulation.cutoff_hz")
            if source != "pulse":
                positive(config.require("demodulation", "frequency_hz"), "demodulation.frequency_hz")
    if project == "calibrate-square":
        positive(config.calibration().get("rabi_at_100_hz", 0), "calibration.rabi_at_100_hz")
    elif project == "calibrate-bb1":
        positive(config.section("pulse").get("angle_deg", 180), "pulse.angle_deg")
    elif project == "analyze":
        options = config.section("analysis")
        steps = options.get("steps", 900)
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 2:
            raise ValueError("analysis.steps must be an integer >= 2")
        if options.get("method", "analytic") not in {"analytic", "scipy_expm"}:
            raise ValueError("analysis.method must be analytic or scipy_expm")
        if options.get("target", "fixed_xpi") not in {"nominal", "best_axis_pi", "fixed_xpi"}:
            raise ValueError("analysis.target must be nominal, best_axis_pi or fixed_xpi")
        if options.get("scaling", "fixed") not in {"fixed", "bb1_area"}:
            raise ValueError("analysis.scaling must be fixed or bb1_area")
        if source != "pulse" and options.get("scaling", "fixed") == "fixed":
            positive(config.calibration().get("omega_rad_s_per_volt", 0), "calibration.omega_rad_s_per_volt")
        config.scan_axes()
        for key in ["amplitude_path", "phase_path"]:
            if options.get(key) and not Path(options[key]).is_file():
                raise ValueError(f"analysis.{key} does not exist")
    elif project == "pulse":
        pulse = generate(config)
        reference = config.calibration().get("rabi_at_100_hz")
        if reference is not None and max(pulse.rabi_hz) > positive(reference, "calibration.rabi_at_100_hz") * (1 + 1e-12):
            raise ValueError("Pulse exceeds the calibrated amplitude-100 Rabi rate")
    elif project == "sequence":
        sequence.prepare(config)
    elif project == "tnmr":
        settings = config.section("tnmr")
        if not Path(config.require("tnmr", "path")).is_file():
            raise ValueError("tnmr.path must be an existing experiment file")
        if settings.get("action", "inspect") not in {"inspect", "update", "clear_tables", "zg", "go", "stop", "abort", "reset"}:
            raise ValueError("Unknown tnmr.action")
        if settings.get("action") == "clear_tables":
            if Path(config.require("tnmr", "save_path")).resolve() == Path(settings["path"]).resolve():
                raise ValueError("Clear tables requires a separate output TNT")
    elif project in {"optimize-offline", "optimize-live", "phase-reference"}:
        optimize.settings(config)
        optimize.objective(config)
        if project == "optimize-offline":
            optimize.initial_coefficients(config)
        else:
            optimize.hardware_config(config, Path(config.section("output").get("directory", "outputs")), phase_only=project == "phase-reference")
    return {"project": project, "description": PROJECTS[project], "hardware": requires_hardware, "input_kind": source if project in raw_projects else "configuration", "target": "fixed_xpi" if project.startswith("optimize") or project == "phase-reference" else config.section("analysis").get("target"), "output_directory": config.section("output").get("directory", str(Path.cwd() / "outputs")), "config": config.data}


def run(project: str, config, *, live: bool = False, scope=None, tnmr_device=None) -> dict:
    """Execute an offline workflow or, with live=True, an explicit live workflow.

    Live requests without live=True return a dry-run plan and perform no I/O.
    Fake devices can be injected for controlled integration tests.
    """
    config = _config(config)
    description = plan(project, config)
    if description["hardware"] and not live:
        return {"dry_run": True, "plan": description}
    if description["hardware"] and "SET_SCOPE" in config.section("scope").get("resource", ""):
        raise ValueError("Replace the scope placeholder in the lab profile before a live run")
    directory = common.run_directory(config, project)
    if project == "capture":
        result = common.capture(config, directory, scope=scope)
    elif project.startswith("calibrate-"):
        result = calibrate.calibrate(config, directory, project.removeprefix("calibrate-"), scope=scope)
    elif project == "analyze":
        result = analyze.analyze(config, directory, scope=scope)
    elif project == "pulse":
        result = generate(config).save(directory, rabi_at_100_hz=config.calibration().get("rabi_at_100_hz"))
    elif project == "sequence":
        result = sequence.edit(config, directory)
    elif project == "tnmr":
        result = tnmr.control(config, directory, tnmr=tnmr_device)
    elif project == "optimize-offline":
        result = optimize.offline(config, directory)
    else:
        result = optimize.live(config, directory, phase_only=project == "phase-reference", scope=scope, tnmr=tnmr_device)
    result = {"project": project, "output_directory": str(directory), **result}
    common.write_json(directory / "result.json", result)
    return result
