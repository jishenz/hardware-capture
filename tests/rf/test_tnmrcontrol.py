"""Tests for the Redstone TNMR control helper."""

from __future__ import annotations

import importlib.util
import threading
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[2] / "hardware_capture/devices/tnmr.py"
SPEC = importlib.util.spec_from_file_location("redstone_tnmrcontrol", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not load {MODULE_PATH}.")
tnmrcontrol = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tnmrcontrol)


class FakeApp:
    def __init__(self, active_doc_path: str = "") -> None:
        self.active_doc_path = active_doc_path
        self.zg_count = 0
        self.opened_paths: list[str] = []
        self.thread_ids: list[int] = []

    def OpenFile(self, path: str) -> None:
        self.thread_ids.append(threading.get_ident())
        self.opened_paths.append(path)

    def GetActiveDocPath(self) -> str:
        return self.active_doc_path

    def GetDocumentList(self) -> str:
        return f"{self.active_doc_path},C:\\TNMR\\data\\other.tnt,"

    def ZG(self) -> None:
        self.thread_ids.append(threading.get_ident())
        self.zg_count += 1


class FakePropertyApp:
    def __init__(self, active_doc_path: str = "") -> None:
        self.GetActiveDocPath = active_doc_path
        self.GetDocumentList = f"{active_doc_path},C:\\TNMR\\data\\other.tnt,"


class FakeDocument:
    def __init__(self) -> None:
        self.parameters: dict[str, object] = {}
        self.tables: dict[str, str] = {}
        self.zg_count = 0
        self.stop_count = 0
        self.saved_paths: list[str] = []
        self.saved_sequence_paths: list[str] = []
        self.loaded_sequence_paths: list[str] = []
        self.thread_ids: list[int] = []

    def _record_thread(self) -> None:
        self.thread_ids.append(threading.get_ident())

    def GetNMRParameter(self, name: str) -> object:
        self._record_thread()
        return self.parameters.get(name)

    def SetNMRParameter(self, name: str, value: object) -> None:
        self._record_thread()
        self.parameters[name] = value

    def SetTable(self, name: str, entries: str) -> None:
        self._record_thread()
        self.tables[name] = entries

    def ZG(self) -> None:
        self._record_thread()
        self.zg_count += 1

    def Stop(self) -> None:
        self._record_thread()
        self.stop_count += 1

    def SaveAs(self, path: str) -> None:
        self._record_thread()
        self.saved_paths.append(path)

    def LoadSequence(self, path: str) -> None:
        self._record_thread()
        self.loaded_sequence_paths.append(path)

    def SaveSequence(self) -> None:
        self._record_thread()
        return None

    def SaveSequenceAs(self, path: str) -> None:
        self._record_thread()
        self.saved_sequence_paths.append(path)

    def GetSequenceName(self) -> str:
        self._record_thread()
        return "sequence.tps"

    @property
    def CheckAcquisition(self) -> bool:
        return True

    @property
    def GetTableList(self) -> str:
        return "ph0,phmodsmoke,"


class TNMRControlTests(unittest.TestCase):
    def _path(self, leaf: str) -> str:
        return str((Path(__file__).resolve().parent / leaf).resolve())

    def _control(self, path: str, document: FakeDocument) -> object:
        return tnmrcontrol.TNMRControl(
            app=FakeApp(active_doc_path=path),
            document=document,
            path=path,
        )

    def test_control_stores_bound_document(self) -> None:
        path = self._path("active.tnt")
        document = FakeDocument()

        control = self._control(path, document)

        self.assertEqual(control.path, path)
        self.assertIs(control.document, document)

    def test_document_list_handles_comma_delimited_property_or_method(self) -> None:
        path = self._path("active.tnt")
        method_control = tnmrcontrol.TNMRControl(app=FakeApp(active_doc_path=path))
        property_control = tnmrcontrol.TNMRControl(app=FakePropertyApp(active_doc_path=path))

        self.assertEqual(
            method_control.document_list(),
            [path, "C:\\TNMR\\data\\other.tnt"],
        )
        self.assertEqual(
            property_control.document_list(),
            [path, "C:\\TNMR\\data\\other.tnt"],
        )

    def test_snapshot_reads_property_style_members(self) -> None:
        path = self._path("snapshot.tnt")
        document = FakeDocument()
        document.parameters.update(
            {
                "Observe Freq.": 601.0100113,
                "SW +/-": 1000000.0,
                "Scans 1D": 4,
                "Points 1D": 1024,
            }
        )
        control = tnmrcontrol.TNMRControl(
            app=FakePropertyApp(active_doc_path=path),
            document=document,
        )

        snapshot = control.snapshot()

        self.assertTrue(snapshot["connected"])
        self.assertEqual(snapshot["active_document_path"], path)
        self.assertEqual(snapshot["sequence_name"], "sequence.tps")
        self.assertEqual(snapshot["parameters"]["Scans 1D"], 4)
        self.assertEqual(snapshot["tables"], ["ph0", "phmodsmoke"])
        self.assertEqual(snapshot["warnings"], [])

    def test_set_parameter_and_zero_and_go(self) -> None:
        path = self._path("dirty.tnt")
        document = FakeDocument()
        control = self._control(path, document)

        control.set_parameter("Observe Freq.", 400.13)

        control.zero_and_go()
        self.assertEqual(document.parameters["Observe Freq."], 400.13)
        self.assertEqual(document.zg_count, 1)

    def test_zero_and_go_prefers_bound_document_over_app(self) -> None:
        path = self._path("active_run.tnt")
        app = FakeApp(active_doc_path=path)
        document = FakeDocument()
        control = tnmrcontrol.TNMRControl(app=app, document=document, path=path)

        control.zero_and_go()

        self.assertEqual(document.zg_count, 1)
        self.assertEqual(app.zg_count, 0)

    def test_document_command_failure_does_not_fall_back_to_application(self) -> None:
        class FailingDocument(FakeDocument):
            def ZG(self) -> bool:
                self._record_thread()
                return False

        path = self._path("active_run.tnt")
        app = FakeApp(active_doc_path=path)
        control = tnmrcontrol.TNMRControl(
            app=app,
            document=FailingDocument(),
            path=path,
        )

        with self.assertRaisesRegex(RuntimeError, "reported failure: ZG"):
            control.zero_and_go()

        self.assertEqual(app.zg_count, 0)

    def test_open_failure_preserves_the_existing_bound_document(self) -> None:
        class FailingOpenApp(FakeApp):
            def OpenFile(self, path: str) -> bool:
                self.thread_ids.append(threading.get_ident())
                self.opened_paths.append(path)
                return False

        old_path = self._path("active.tnt")
        old_document = FakeDocument()
        control = tnmrcontrol.TNMRControl(
            app=FailingOpenApp(active_doc_path=old_path),
            document=old_document,
            path=old_path,
        )

        with self.assertRaisesRegex(RuntimeError, "reported failure: OpenFile"):
            control.open(MODULE_PATH)

        self.assertIs(control.document, old_document)
        self.assertEqual(control.path, old_path)

    def test_connected_control_uses_one_com_owner_thread(self) -> None:
        path = str(MODULE_PATH)
        app = FakeApp(active_doc_path=path)
        document = FakeDocument()

        class FakeComClient:
            def GetObject(self, *args, **kwargs):
                if kwargs.get("Class") == tnmrcontrol.TNMR_APPLICATION_PROGID:
                    return app
                return document

        original_com_client = tnmrcontrol._com_client
        tnmrcontrol._com_client = lambda: FakeComClient()
        control = tnmrcontrol.TNMRControl()
        try:
            control.connect()
            owner_thread_id = control._worker_thread_id
            self.assertIsNotNone(owner_thread_id)
            self.assertNotEqual(owner_thread_id, threading.get_ident())

            control.open(path)
            control.set_parameter("Observe Freq.", 400.13)

            errors: list[BaseException] = []

            def call_zero_and_go() -> None:
                try:
                    control.zero_and_go()
                except BaseException as exc:
                    errors.append(exc)

            caller = threading.Thread(target=call_zero_and_go)
            caller.start()
            caller.join(timeout=2)

            self.assertFalse(caller.is_alive())
            self.assertEqual(errors, [])
            self.assertEqual(app.opened_paths, [path])
            self.assertEqual(document.parameters["Observe Freq."], 400.13)
            self.assertEqual(document.zg_count, 1)
            self.assertEqual(set(app.thread_ids + document.thread_ids), {owner_thread_id})
        finally:
            control.close()
            tnmrcontrol._com_client = original_com_client

    def test_constructor_can_launch_tnmr_when_no_instance_is_running(self) -> None:
        app = FakeApp()

        class FakeComClient:
            def __init__(self) -> None:
                self.dispatched_progid = ""

            def GetObject(self, *args, **kwargs):
                raise RuntimeError("TNMR is not running")

            def Dispatch(self, progid: str):
                self.dispatched_progid = progid
                return app

        client = FakeComClient()
        original_com_client = tnmrcontrol._com_client
        tnmrcontrol._com_client = lambda: client
        try:
            control = tnmrcontrol.TNMRControl(connect=True, launch=True)

            self.assertIs(control.app, app)
            self.assertEqual(client.dispatched_progid, tnmrcontrol.TNMR_APPLICATION_PROGID)
        finally:
            control.close()
            tnmrcontrol._com_client = original_com_client

    def test_set_table_formats_iterables_as_crlf_text(self) -> None:
        path = self._path("table.tnt")
        document = FakeDocument()
        control = self._control(path, document)

        control.set_table("phase0", [0, 90, 180])

        self.assertEqual(document.tables["phase0"], "0\r\n90\r\n180\r\n")

    def test_stop_and_save_sequence(self) -> None:
        path = self._path("run.tnt")
        output_path = self._path("run_out.tnt")
        sequence_path = str(MODULE_PATH)
        output_sequence_path = self._path("sequence_out.tps")
        document = FakeDocument()
        control = self._control(path, document)

        control.zero_and_go()
        control.stop()
        control.save_as(output_path)
        control.load_sequence(sequence_path)
        control.save_sequence()
        control.save_sequence_as(output_sequence_path)

        self.assertEqual(document.stop_count, 1)
        self.assertEqual(document.saved_paths, [output_path])
        self.assertEqual(document.loaded_sequence_paths, [sequence_path])
        self.assertEqual(document.saved_sequence_paths, [output_sequence_path])

    def test_stop_requires_rf_tune_inactive(self) -> None:
        path = self._path("rf_guard.tnt")
        document = FakeDocument()
        control = self._control(path, document)

        with self.assertRaisesRegex(RuntimeError, "RF tune is active"):
            control.stop(rf_tune_active=True)


if __name__ == "__main__":
    unittest.main()
