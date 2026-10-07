# Measurement upload contract

CSV files contain the following columns **in exactly this order**:

```text
unit,cycle,setting_1,setting_2,setting_3,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15,s16,s17,s18,s19,s20,s21
```

Use numeric source units with the C-MAPSS channel meanings. The user selects the matching simulation scenario; it is not inferred. The three settings are required. Blank sensor cells are supported and receive a missingness indicator plus causal forward fill within the 30-cycle window; no available history uses a training-condition mean. Infinite values are rejected.

`unit` and `cycle` must be positive integers. Each engine's rows must appear in consecutive increasing cycle order with no duplicates. Several engines may share a file, but unit identifiers must identify one trajectory each. Maximum: 10 MiB and 50,000 rows. Labels, total lifetime and future measurements are not input columns. A short trajectory is allowed with an explicit left-padding warning.

The interface retains uploads in session memory. It does not save them to disk or log their contents. The original measurements do not leave the chosen hosting server for model inference. The displayed trajectory and CSV export contain at most 300 evenly spaced past endpoints (including the final endpoint) to bound memory and computation. The current prediction always uses the full available past to construct its last 30-cycle window.

PNG exports show the displayed prediction trajectory. Inspection JSON includes the exact model asset hashes, last observed cycle, window length, missingness and any applied stress. Latency is measured per request and will vary; it is not an accuracy measure. Uploaded files have no actual-RUL reference label.
