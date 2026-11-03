"""Local connection settings; override with environment variables before a live run."""

import os

TNMR_APPLICATION_PROGID = os.environ.get("TNMR_APPLICATION_PROGID", "NTNMR.Application")
WF_NV_PORT = os.environ.get("WF_NV_PORT", "COM4")
NI_ch_FM_Analog = os.environ.get("NI_CH_FM_ANALOG", "Dev1/ao3")
NI_ch_FM_Trig = os.environ.get("NI_CH_FM_TRIG", "/Dev1/PFI1")
