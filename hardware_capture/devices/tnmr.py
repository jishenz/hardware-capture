r"""Simple TNMR OLE automation control.

Use one public class:

    control = TNMRControl()
    control.connect()
    control.open(r"C:\TNMR\data\file.tnt")
    control.zero_and_go()

The wrapper intentionally keeps public methods close to the TNMR command names.
Live COM work is serialized on one owner thread and failures are raised as plain
``RuntimeError`` exceptions.
"""

from __future__ import annotations

import concurrent.futures
import threading
from collections.abc import Iterable
from pathlib import Path
from threading import RLock
from typing import Any

DEFAULT_TNMR_APPLICATION_PROGID = "NTNMR.Application"
DASHBOARD_TAB_ORDER = (
    "Acquisition",
    "Frequency",
    "Processing",
    "Grad. Preemph.",
    "Misc.",
    "Sequence",
)


from hardware_capture.config import TNMR_APPLICATION_PROGID


def _path(path: str | Path) -> str:
    return str(Path(path).expanduser().resolve())


def _existing_path(path: str | Path) -> str:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise RuntimeError(
            f"File does not exist: {resolved}. Use a raw Windows string, for "
            "example r'C:\\TNMR\\data\\file.tnt'."
        )
    return str(resolved)


def _com_client():
    try:
        import pythoncom
        import win32com.client
    except ImportError as exc:
        raise RuntimeError("pywin32 is required for live TNMR COM access.") from exc
    pythoncom.CoInitialize()
    return win32com.client


def _call(target: Any, command: str, *args: Any) -> Any:
    member = getattr(target, command, None)
    if callable(member):
        try:
            result = member(*args)
        except Exception as exc:
            raise RuntimeError(f"TNMR command failed: {command}: {exc}") from None
        if result is False:
            raise RuntimeError(f"TNMR command reported failure: {command}")
        return result
    if args or member is None:
        raise RuntimeError(f"TNMR command failed: {command}")
    return member


def _read_member(target: Any, name: str, *args: Any) -> Any:
    member = getattr(target, name, None)
    if callable(member):
        try:
            return member(*args)
        except Exception as exc:
            raise RuntimeError(f"TNMR command failed: {name}: {exc}") from None
    if member is None:
        raise RuntimeError(f"TNMR command failed: {name}")
    if args:
        raise RuntimeError(f"TNMR member is not callable: {name}")
    return member


def _name_list(value: Any) -> list[str]:
    return _split_tnmr_list(value)


def _split_tnmr_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        prefix, separator, rest = text.partition("=")
        if separator and (
            prefix.strip() in DASHBOARD_TAB_ORDER
            or prefix.strip().lower().endswith("list")
        ):
            text = rest
        raw_items = text.replace("\r", "\n").replace(",", "\n").splitlines()
    else:
        try:
            raw_items = list(value)
        except TypeError:
            raw_items = [value]
    return [
        name
        for name in (str(item).strip() for item in raw_items)
        if name and name not in {":", "::"}
    ]


COMMON_STATUS_PARAMETERS = (
    "Observe Freq.",
    "SW +/-",
    "Scans 1D",
    "Points 1D",
)


class TNMRControl:
    """Connect to TNMR, bind one document, and send TNMR commands.

    Live COM access is serialized on one owner thread so GUI, measurement, and
    notebook callers can share this object without touching COM directly.
    """

    DASHBOARD_TAB_ORDER = DASHBOARD_TAB_ORDER

    def __init__(
        self,
        *,
        app: Any | None = None,
        document: Any | None = None,
        path: str | Path | None = None,
        connect: bool = False,
        launch: bool = False,
    ) -> None:
        self.app = app
        self.document = document
        self.path = "" if path is None else _path(path)
        self._lock = RLock()
        self._executor: concurrent.futures.ThreadPoolExecutor | None = None
        self._worker_thread_id: int | None = None
        self.last_message = "Not connected."
        if connect:
            self.connect(launch=launch)

    def _use_worker_thread(self) -> bool:
        return self._executor is not None or self.app is None

    def _in_worker_thread(self) -> bool:
        return self._worker_thread_id == threading.get_ident()

    def _run_on_worker(self, method: Any, *args: Any, **kwargs: Any) -> Any:
        if self._in_worker_thread():
            return method(*args, **kwargs)
        if self._executor is None:
            self._executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="TNMRControlCOM",
            )
        future = self._executor.submit(self._worker_call, method, args, kwargs)
        return future.result()

    def _worker_call(self, method: Any, args: tuple[Any, ...], kwargs: dict[str, Any]):
        self._worker_thread_id = threading.get_ident()
        return method(*args, **kwargs)

    def connect(self, *, launch: bool = False) -> "TNMRControl":
        """Attach to TNMR. By default, never launch a new TNMR instance."""

        if self._use_worker_thread() and not self._in_worker_thread():
            return self._run_on_worker(self.connect, launch=launch)
        with self._lock:
            client = _com_client()
            try:
                self.app = client.GetObject(Class=TNMR_APPLICATION_PROGID)
            except Exception as exc:
                if not launch:
                    raise RuntimeError(
                        f"Could not attach to running TNMR application: {exc}"
                    ) from None
                self.app = client.Dispatch(TNMR_APPLICATION_PROGID)
            self.last_message = "Connected to TNMR application."
            return self

    def bind_active_document(self) -> "TNMRControl":
        """Bind the active TNMR document without opening or saving files."""

        if self._use_worker_thread() and not self._in_worker_thread():
            return self._run_on_worker(self.bind_active_document)
        with self._lock:
            if self.app is None:
                self.connect(launch=False)
            active_path = self._active_document_path_unlocked()
            self.document = _com_client().GetObject(active_path)
            self.path = active_path
            self.last_message = f"Bound active TNMR document: {active_path}"
            return self

    def open(self, path: str | Path) -> "TNMRControl":
        """Open a TNMR file and bind the document COM object."""

        if self._use_worker_thread() and not self._in_worker_thread():
            return self._run_on_worker(self.open, path)
        with self._lock:
            document_path = _existing_path(path)
            if self.app is None:
                self.connect(launch=False)
            _call(self.app, "OpenFile", document_path)
            try:
                document = _com_client().GetObject(document_path)
            except Exception as exc:
                raise RuntimeError(
                    f"Could not bind TNMR document: {document_path}: {exc}"
                ) from None
            self.document = document
            self.path = document_path
            self.last_message = f"Opened TNMR document: {document_path}"
            return self

    def _doc(self) -> Any:
        if self.document is None:
            raise RuntimeError("No TNMR document is bound.")
        return self.document

    def _app_command(self, command: str, *args: Any) -> Any:
        if self.app is None:
            raise RuntimeError("No TNMR application is attached.")
        return _read_member(self.app, command, *args)

    def _document_command(self, command: str) -> Any:
        return _call(self._doc(), command)

    def active_document_path(self) -> str:
        if self._use_worker_thread() and not self._in_worker_thread():
            return self._run_on_worker(self.active_document_path)
        with self._lock:
            return self._active_document_path_unlocked()

    def _active_document_path_unlocked(self) -> str:
        if self.path:
            return self.path
        active_path = str(self._app_command("GetActiveDocPath")).strip()
        if not active_path:
            raise RuntimeError("TNMR did not report an active document path.")
        return _path(active_path)

    def document_list(self) -> list[str]:
        if self._use_worker_thread() and not self._in_worker_thread():
            return self._run_on_worker(self.document_list)
        with self._lock:
            value = self._app_command("GetDocumentList")
            return _split_tnmr_list(value)

    def application_info(self) -> dict[str, Any]:
        if self._use_worker_thread() and not self._in_worker_thread():
            return self._run_on_worker(self.application_info)
        if self.app is None:
            raise RuntimeError("No TNMR application is attached.")
        return {
            "progid": TNMR_APPLICATION_PROGID,
            "application_object": repr(self.app),
            "bound_document_path": self.path,
        }

    def parameter_page_list(self) -> list[str]:
        if self._use_worker_thread() and not self._in_worker_thread():
            return self._run_on_worker(self.parameter_page_list)
        with self._lock:
            return _name_list(self._app_command("GetParameterPageList"))

    def parameter_list_in_page(self, page: str) -> list[str]:
        if self._use_worker_thread() and not self._in_worker_thread():
            return self._run_on_worker(self.parameter_list_in_page, page)
        with self._lock:
            return _name_list(self._app_command("GetParameterListInPage", page))

    def get_parameter(self, name: str) -> Any:
        if self._use_worker_thread() and not self._in_worker_thread():
            return self._run_on_worker(self.get_parameter, name)
        with self._lock:
            return _read_member(self._doc(), "GetNMRParameter", name)

    def set_parameter(self, name: str, value: Any) -> "TNMRControl":
        if self._use_worker_thread() and not self._in_worker_thread():
            return self._run_on_worker(self.set_parameter, name, value)
        with self._lock:
            _call(self._doc(), "SetNMRParameter", name, value)
            self.last_message = f"Set TNMR parameter {name}."
            return self

    def zero_and_go(self) -> "TNMRControl":
        if self._use_worker_thread() and not self._in_worker_thread():
            return self._run_on_worker(self.zero_and_go)
        with self._lock:
            self._document_command("ZG")
            self.last_message = "Sent ZG."
            return self

    def zg(self) -> "TNMRControl":
        return self.zero_and_go()

    def go(self) -> "TNMRControl":
        if self._use_worker_thread() and not self._in_worker_thread():
            return self._run_on_worker(self.go)
        with self._lock:
            self._document_command("Go")
            self.last_message = "Sent Go."
            return self

    def stop(self, *, rf_tune_active: bool = False) -> "TNMRControl":
        if self._use_worker_thread() and not self._in_worker_thread():
            return self._run_on_worker(self.stop, rf_tune_active=rf_tune_active)
        if rf_tune_active:
            raise RuntimeError("RF tune is active. Stop RF tune first.")
        with self._lock:
            self._document_command("Stop")
            self.last_message = "Sent Stop."
            return self

    def abort(self) -> "TNMRControl":
        if self._use_worker_thread() and not self._in_worker_thread():
            return self._run_on_worker(self.abort)
        with self._lock:
            self._document_command("Abort")
            self.last_message = "Sent Abort."
            return self

    def reset(self) -> "TNMRControl":
        if self._use_worker_thread() and not self._in_worker_thread():
            return self._run_on_worker(self.reset)
        with self._lock:
            self._document_command("Reset")
            self.last_message = "Sent Reset."
            return self

    def sequence_name(self) -> str:
        if self._use_worker_thread() and not self._in_worker_thread():
            return self._run_on_worker(self.sequence_name)
        with self._lock:
            return str(_read_member(self._doc(), "GetSequenceName"))

    def load_sequence(self, path: str | Path) -> "TNMRControl":
        if self._use_worker_thread() and not self._in_worker_thread():
            return self._run_on_worker(self.load_sequence, path)
        with self._lock:
            sequence_path = _existing_path(path)
            _call(self._doc(), "LoadSequence", sequence_path)
            self.last_message = f"Loaded sequence: {sequence_path}"
            return self

    def save_sequence(self) -> "TNMRControl":
        if self._use_worker_thread() and not self._in_worker_thread():
            return self._run_on_worker(self.save_sequence)
        with self._lock:
            _call(self._doc(), "SaveSequence")
            self.last_message = "Saved sequence."
            return self

    def save_sequence_as(self, path: str | Path) -> "TNMRControl":
        if self._use_worker_thread() and not self._in_worker_thread():
            return self._run_on_worker(self.save_sequence_as, path)
        with self._lock:
            output_path = _path(path)
            _call(self._doc(), "SaveSequenceAs", output_path)
            self.last_message = f"Saved sequence as: {output_path}"
            return self

    def get_table_list(self) -> list[str]:
        if self._use_worker_thread() and not self._in_worker_thread():
            return self._run_on_worker(self.get_table_list)
        with self._lock:
            return _name_list(_read_member(self._doc(), "GetTableList"))

    def get_table(self, name: str) -> Any:
        if self._use_worker_thread() and not self._in_worker_thread():
            return self._run_on_worker(self.get_table, name)
        with self._lock:
            return _read_member(self._doc(), "GetTable", name)

    def set_table(self, name: str, entries: str | Iterable[object]) -> "TNMRControl":
        if self._use_worker_thread() and not self._in_worker_thread():
            return self._run_on_worker(self.set_table, name, entries)
        with self._lock:
            if isinstance(entries, str):
                table_entries = entries
            else:
                lines = [str(entry) for entry in entries]
                table_entries = "" if not lines else "\r\n".join(lines) + "\r\n"
            _call(self._doc(), "SetTable", name, table_entries)
            self.last_message = f"Set TNMR table {name}."
            return self

    def save(self) -> "TNMRControl":
        if self._use_worker_thread() and not self._in_worker_thread():
            return self._run_on_worker(self.save)
        with self._lock:
            _call(self._doc(), "Save")
            self.last_message = "Saved TNMR document."
            return self

    def save_as(self, path: str | Path) -> "TNMRControl":
        if self._use_worker_thread() and not self._in_worker_thread():
            return self._run_on_worker(self.save_as, path)
        with self._lock:
            self.path = _path(path)
            _call(self._doc(), "SaveAs", self.path)
            self.last_message = f"Saved TNMR document as: {self.path}"
            return self

    def close(self) -> None:
        executor = self._executor
        if executor is not None and not self._in_worker_thread():
            try:
                self._run_on_worker(self.close)
            finally:
                executor.shutdown(wait=True)
                self._executor = None
                self._worker_thread_id = None
            return

        with self._lock:
            self.document = None
            self.app = None
            self.last_message = "Disconnected from TNMR application."

    def snapshot(self, parameter_names: Iterable[str] | None = None) -> dict[str, Any]:
        """Return a plain read-only status snapshot for GUI use."""

        if self._use_worker_thread() and not self._in_worker_thread():
            return self._run_on_worker(self.snapshot, parameter_names)
        with self._lock:
            warnings: list[str] = []
            connected = self.app is not None
            if self.app is None:
                try:
                    self.connect(launch=False)
                    connected = True
                except Exception as exc:
                    warnings.append(str(exc))
                    return {
                        "connected": False,
                        "active_document_path": "",
                        "bound_document_path": self.path,
                        "document_list": [],
                        "sequence_name": "",
                        "acquisition_complete": None,
                        "parameters": {},
                        "tables": [],
                        "warnings": warnings,
                        "last_message": self.last_message,
                    }

            active_path = ""
            saved_path = self.path
            try:
                self.path = ""
                active_path = self._active_document_path_unlocked()
            except Exception as exc:
                warnings.append(f"Active document path unavailable: {exc}")
            finally:
                self.path = saved_path

            if self.document is None and active_path:
                try:
                    self.document = _com_client().GetObject(active_path)
                    self.path = active_path
                except Exception as exc:
                    warnings.append(f"Could not bind active document: {exc}")

            documents: list[str] = []
            try:
                documents = _split_tnmr_list(self._app_command("GetDocumentList"))
            except Exception as exc:
                warnings.append(f"Document list unavailable: {exc}")

            sequence_name = ""
            acquisition_complete = None
            tables: list[str] = []
            parameters: dict[str, Any] = {}
            if self.document is not None:
                try:
                    sequence_name = str(_read_member(self.document, "GetSequenceName"))
                except Exception as exc:
                    warnings.append(f"Sequence name unavailable: {exc}")
                try:
                    acquisition_complete = bool(
                        _read_member(self.document, "CheckAcquisition")
                    )
                except Exception as exc:
                    warnings.append(f"Acquisition state unavailable: {exc}")
                try:
                    tables = _split_tnmr_list(
                        _read_member(self.document, "GetTableList")
                    )
                except Exception as exc:
                    warnings.append(f"Table list unavailable: {exc}")
                for name in tuple(parameter_names or COMMON_STATUS_PARAMETERS):
                    try:
                        parameters[name] = _read_member(
                            self.document,
                            "GetNMRParameter",
                            name,
                        )
                    except Exception as exc:
                        parameters[name] = f"<error: {exc}>"

            if connected:
                self.last_message = "Refreshed TNMR status."
            return {
                "connected": connected,
                "active_document_path": active_path,
                "bound_document_path": self.path,
                "document_list": documents,
                "sequence_name": sequence_name,
                "acquisition_complete": acquisition_complete,
                "parameters": parameters,
                "tables": tables,
                "warnings": warnings,
                "last_message": self.last_message,
            }
