import numpy as np

import pytest

from hardware_capture.devices.rigol import RigolDHO, ScopeTimeoutError, WaveformCaptureConfig, chunk_ranges, normalize_waveform_format, parse_ascii_waveform, parse_preamble, parse_time_seconds, point_to_time, time_to_point

pytestmark = pytest.mark.unit

def test_parse_time_seconds_with_units():
    assert parse_time_seconds("-1us") == pytest.approx(-1e-6)
    assert parse_time_seconds("2 ms") == pytest.approx(2e-3)
    assert parse_time_seconds("3e-6") == pytest.approx(3e-6)

def test_preamble_point_time_roundtrip():
    preamble = parse_preamble("0,0,1000,1,1e-9,-5e-7,0,1,0,0")
    point = time_to_point(0.0, preamble)
    assert point_to_time(point, preamble) == pytest.approx(0.0)

def test_parse_ascii_waveform_strips_ieee_block():
    assert np.allclose(parse_ascii_waveform("#2111,2,3,4,5"), [1, 2, 3, 4, 5])

def test_chunk_ranges_are_inclusive():
    assert list(chunk_ranges(1, 10, 4)) == [(1, 4), (5, 8), (9, 10)]

class FakeSession:
    def __init__(self, trigger_statuses=None):
        self.timeout = None
        self.writes = []
        self.queries = []
        self.events = []
        self.trigger_statuses = list(trigger_statuses or ["STOP"])
        self.waveform_format = "WORD"
        self.start_point, self.stop_point = 1, 10
        self.codes = np.arange(99, 109, dtype=np.uint16)
        self.binary_options = []

    def query(self, command):
        self.queries.append(command)
        self.events.append(("query", command))
        if command == "*IDN?":
            return "RIGOL,DHO-test,123,0"
        if command == ":WAV:PRE?":
            data_format = {"BYTE": 0, "WORD": 1, "ASC": 2}[self.waveform_format]
            return f"{data_format},2,10,1,1e-6,0,0,1,0,100"
        if command == ":ACQ:MDEP?":
            return "10"
        if command == ":WAV:DATA?":
            assert self.waveform_format == "ASC", "Binary data used text query"
            return ",".join(str(float(v) - 100) for v in self.codes[self.start_point - 1:self.stop_point])
        if command == ":TRIGger:STATus?":
            if len(self.trigger_statuses) > 1:
                return self.trigger_statuses.pop(0)
            return self.trigger_statuses[0]
        raise AssertionError(f"unexpected query {command}")

    def write(self, command):
        self.writes.append(command)
        self.events.append(("write", command))
        if command.startswith(":WAV:FORM "):
            self.waveform_format = command.split()[-1]
        elif command.startswith(":WAV:STAR "):
            self.start_point = int(command.split()[-1])
        elif command.startswith(":WAV:STOP "):
            self.stop_point = int(command.split()[-1])

    def query_binary_values(self, command, **kwargs):
        assert command == ":WAV:DATA?"
        self.queries.append(command)
        self.events.append(("query", command))
        self.binary_options.append(kwargs)
        return self.codes[self.start_point - 1:self.stop_point].copy()

    def close(self):
        pass

class FakeResourceManager:
    def __init__(self, session=None):
        self.session = session or FakeSession()
        self.close_count = 0

    def open_resource(self, resource):
        return self.session

    def close(self):
        self.close_count += 1

def test_rigol_capture_uses_configured_point_window():
    rm = FakeResourceManager()
    scope = RigolDHO(resource="TCPIP::fake::INSTR", resource_manager=rm)
    waveform = scope.capture_waveform(
        WaveformCaptureConfig(
            channel=1,
            start_point=2,
            stop_point=5,
            chunk_points=10,
        )
    )

    assert waveform.source.startswith("RIGOL")
    assert waveform.channel == "CHAN1"
    assert waveform.start_point == 2
    assert waveform.stop_point == 5
    assert np.allclose(waveform.voltage, [0, 1, 2, 3])
    assert ":WAV:STAR 2" in rm.session.writes
    assert ":WAV:STOP 5" in rm.session.writes

def test_single_capture_arms_before_callback_and_reads_after_stop():
    session = FakeSession(trigger_statuses=["WAIT", "STOP", "WAIT", "WAIT"])
    rm = FakeResourceManager(session)
    events = []
    scope = RigolDHO(resource="TCPIP::fake::INSTR", resource_manager=rm)

    def start_shot():
        events.append(("callback", list(session.writes)))

    waveform = scope.capture_single_waveform(
        WaveformCaptureConfig(
            channel=1,
            start_point=2,
            stop_point=5,
            chunk_points=10,
        ),
        start_shot=start_shot,
        trigger_timeout_s=1.0,
        armed_timeout_s=1.0,
        poll_s=0.0,
        rearm_after_read=True,
    )

    assert waveform.start_point == 2
    single_indices = [
        index for index, command in enumerate(session.writes) if command == ":SINGle"
    ]
    assert len(single_indices) == 2
    assert ":STOP" not in session.writes
    assert events
    callback_writes = events[0][1]
    assert ":SINGle" in callback_writes
    assert ":WAV:DATA?" in session.queries
    assert session.queries.index(":TRIGger:STATus?") < session.queries.index(":WAV:DATA?")
    data_event_index = session.events.index(("query", ":WAV:DATA?"))
    second_single_event_index = [
        index
        for index, event in enumerate(session.events)
        if event == ("write", ":SINGle")
    ][1]
    assert data_event_index < second_single_event_index

def test_wait_until_stopped_times_out():
    session = FakeSession(trigger_statuses=["WAIT"])
    rm = FakeResourceManager(session)
    scope = RigolDHO(resource="TCPIP::fake::INSTR", resource_manager=rm)
    scope.open()

    with pytest.raises(ScopeTimeoutError):
        scope.wait_until_stopped(timeout_s=0.0, poll_s=0.0)


def test_word_transfer_preserves_time_window_and_scales_unsigned_codes():
    session = FakeSession()
    # Include embedded LF/CR bytes and codes on both sides of the signed boundary.
    codes = np.array([0, 10, 13, 32767, 32768, 65535], dtype=np.uint16)
    session.codes = np.r_[np.zeros(3, dtype=np.uint16), codes, np.zeros(3, dtype=np.uint16)]
    original_query = session.query

    def query(command):
        if command == ":WAV:PRE?":
            assert session.waveform_format == "WORD"
            return "1,2,12,1,2e-9,-10e-9,2,0.0025,10,32768"
        if command == ":ACQ:MDEP?":
            return "12"
        return original_query(command)

    session.query = query
    scope = RigolDHO(resource="FAKE", resource_manager=FakeResourceManager(session))
    result = scope.capture_waveform(WaveformCaptureConfig(
        window_start_s=-8e-9, window_end_s=2e-9, chunk_points=4, allow_clamp=False,
    ))
    assert (result.start_point, result.stop_point) == (4, 9)
    np.testing.assert_allclose(result.time, [-8e-9, -6e-9, -4e-9, -2e-9, 0, 2e-9], atol=1e-23)
    np.testing.assert_allclose(result.voltage, (codes.astype(float) - 10 - 32768) * 0.0025)
    assert ":WAV:STAR 4" in session.writes
    assert ":WAV:STOP 7" in session.writes
    assert ":WAV:STAR 8" in session.writes
    assert ":WAV:STOP 9" in session.writes
    assert len(session.binary_options) == 2
    for options in session.binary_options:
        assert options == dict(datatype="H", is_big_endian=False, container=np.array,
                               header_fmt="ieee", expect_termination=True, chunk_size=1024*1024)


@pytest.mark.parametrize("waveform_format", ["ASC", "ascii", "WORD", "BYTE"])
def test_transfer_formats_produce_the_same_voltages_and_times(waveform_format):
    session = FakeSession()
    scope = RigolDHO(resource="FAKE", resource_manager=FakeResourceManager(session))
    result = scope.capture_waveform(WaveformCaptureConfig(
        start_point=2, stop_point=7, chunk_points=4, waveform_format=waveform_format,
    ))
    np.testing.assert_allclose(result.voltage, np.arange(6))
    np.testing.assert_allclose(result.time, np.arange(1, 7) * 1e-6)
    assert bool(session.binary_options) == (normalize_waveform_format(waveform_format) != "ASC")
    if waveform_format == "BYTE":
        assert session.binary_options[0]["datatype"] == "B"


@pytest.mark.parametrize("count", [0, 3, 5])
def test_binary_read_rejects_incomplete_or_extra_samples(count):
    session = FakeSession()
    session.query_binary_values = lambda *args, **kwargs: np.zeros(count)
    scope = RigolDHO(resource="FAKE", resource_manager=FakeResourceManager(session))
    with pytest.raises(ValueError, match="expected 4"):
        scope.capture_waveform(WaveformCaptureConfig(start_point=2, stop_point=5))


def test_rejected_word_setting_is_detected_before_reading_samples():
    session = FakeSession()
    original_query = session.query
    session.query = lambda command: "2,2,10,1,1e-6,0,0,1,0,0" if command == ":WAV:PRE?" else original_query(command)
    scope = RigolDHO(resource="FAKE", resource_manager=FakeResourceManager(session))
    with pytest.raises(ValueError, match="Refusing to decode mismatched data"):
        scope.capture_waveform(WaveformCaptureConfig(start_point=2, stop_point=5))
    assert ":WAV:DATA?" not in session.queries


def test_invalid_format_is_rejected_before_device_access():
    scope = RigolDHO(resource="FAKE")
    with pytest.raises(ValueError, match="waveform_format"):
        scope.capture_waveform(WaveformCaptureConfig(waveform_format="BIN"))
    with pytest.raises(ValueError, match="waveform_format"):
        scope.capture_single_waveform(WaveformCaptureConfig(waveform_format="BIN"))


@pytest.mark.parametrize("scale", ["nan", "0", "-1"])
def test_invalid_voltage_scaling_is_rejected(scale):
    session = FakeSession()
    original_query = session.query
    session.query = lambda command: f"1,2,10,1,1e-6,0,0,{scale},0,100" if command == ":WAV:PRE?" else original_query(command)
    scope = RigolDHO(resource="FAKE", resource_manager=FakeResourceManager(session))
    with pytest.raises(ValueError, match="invalid timing or voltage scaling"):
        scope.capture_waveform(WaveformCaptureConfig(start_point=2, stop_point=5))


def test_binary_failure_propagates_without_an_ascii_retry():
    session = FakeSession()
    def fail(*args, **kwargs):
        raise TimeoutError("synthetic transfer timeout")
    session.query_binary_values = fail
    scope = RigolDHO(resource="FAKE", resource_manager=FakeResourceManager(session))
    with pytest.raises(TimeoutError, match="synthetic transfer timeout"):
        scope.capture_waveform(WaveformCaptureConfig(start_point=2, stop_point=5))
    assert ":WAV:FORM ASC" not in session.writes


def test_word_read_through_pyvisa_binary_decoder_without_a_device():
    """Exercise real IEEE-block decoding with an in-memory transport only."""
    messagebased = pytest.importorskip("pyvisa.resources.messagebased")

    class WireSession(FakeSession):
        query_delay = 0.0
        _read_termination = "\n"

        def write(self, command):
            super().write(command)
            if command == ":WAV:DATA?":
                payload = self.codes[self.start_point - 1:self.stop_point].astype("<u2").tobytes()
                length = str(len(payload)).encode("ascii")
                self.pending = b"#" + str(len(length)).encode("ascii") + length + payload + b"\n"

        def query_binary_values(self, *args, **kwargs):
            return messagebased.MessageBasedResource.query_binary_values(self, *args, **kwargs)

        def read_binary_values(self, *args, **kwargs):
            return messagebased.MessageBasedResource.read_binary_values(self, *args, **kwargs)

        def _read_raw(self, *args, **kwargs):
            # Simulate a low-level read stopping on an LF inside a sample.
            length = self.pending.index(b"\n") + 1
            value, self.pending = self.pending[:length], self.pending[length:]
            return bytearray(value)

        def read_bytes(self, count, **kwargs):
            assert count <= len(self.pending)
            value, self.pending = self.pending[:count], self.pending[count:]
            return value

    session = WireSession()
    session.codes = np.array([0, 10, 13, 256, 32768, 65535, 0, 0, 0, 0], dtype=np.uint16)
    scope = RigolDHO(resource="FAKE", resource_manager=FakeResourceManager(session))
    result = scope.capture_waveform(WaveformCaptureConfig(start_point=2, stop_point=6, chunk_points=3))
    np.testing.assert_allclose(result.voltage, [-90, -87, 156, 32668, 65435])
    assert session.pending == b""


def test_capture_config_defaults_to_word_and_allows_ascii_override():
    from hardware_capture.configuration import Experiment
    from hardware_capture.workflows.common import capture_config

    config = Experiment({"scope": {"window_start_s": -1e-6, "window_end_s": 2e-6}})
    assert capture_config(config).waveform_format == "WORD"
    fallback = capture_config(config.with_updates(scope={"waveform_format": "ASCII"}))
    assert fallback.waveform_format == "ASC"
    assert fallback.window_start_s == -1e-6
    assert fallback.window_end_s == 2e-6
