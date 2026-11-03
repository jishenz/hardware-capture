"""Local connection settings; override with environment variables before a live run."""

import os

TNMR_APPLICATION_PROGID = os.environ.get("TNMR_APPLICATION_PROGID", "NTNMR.Application")
