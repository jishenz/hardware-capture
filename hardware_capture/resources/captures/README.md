# Capture fixtures

`pulse1_008.csv` is the extracted laboratory BB1 capture; its original example assumes a 90-degree target and 601.0100113 MHz demodulation. Verify those assumptions before quantitative use.

`synthetic_square.csv` is synthetic, not a measurement: 20 MHz sampling, 2 MHz cosine carrier, 30 mV amplitude on [0, 40) microseconds, zero elsewhere. The example assigns 100 kHz Rabi at that plateau solely to exercise the calibration workflow.
