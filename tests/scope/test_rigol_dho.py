import numpy as np

import pytest

from hardware_capture.devices.rigol import RigolDHO, ScopeTimeoutError, WaveformCaptureConfig, chunk_ranges, parse_ascii_waveform, parse_preamble, parse_time_seconds, point_to_time, time_to_point

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

    def query(self, command):
        self.queries.append(command)
        self.events.append(("query", command))
        if command == "*IDN?":
            return "RIGOL,DHO-test,123,0"
        if command == ":WAV:PRE?":
            return "0,0,10,1,1e-6,0,0,1,0,0"
        if command == ":ACQ:MDEP?":
            return "10"
        if command == ":WAV:DATA?":
            return "0,1,2,3"
        if command == ":TRIGger:STATus?":
            if len(self.trigger_statuses) > 1:
                return self.trigger_statuses.pop(0)
            return self.trigger_statuses[0]
        raise AssertionError(f"unexpected query {command}")

    def write(self, command):
        self.writes.append(command)
        self.events.append(("write", command))

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
