"""Oscilloscope capture and RF waveform analysis helpers."""

from hardware_capture.oscilloscope.rigol_dho import (
    DEFAULT_IP,
    DEFAULT_TIMEOUT_MS,
    RigolDHO,
    ScopePreamble,
    ScopeTimeoutError,
    ScopeWaveform,
    WaveformCaptureConfig,
    default_visa_resource,
    normalize_channel,
    parse_time_seconds,
    save_waveform_csv,
    timestamped_capture_path,
)

__all__ = [
    "DEFAULT_IP",
    "DEFAULT_TIMEOUT_MS",
    "RigolDHO",
    "ScopePreamble",
    "ScopeTimeoutError",
    "ScopeWaveform",
    "WaveformCaptureConfig",
    "default_visa_resource",
    "normalize_channel",
    "parse_time_seconds",
    "save_waveform_csv",
    "timestamped_capture_path",
]
