# Data provenance

Sources checked 2026-10-07:

- [NASA catalog](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data)
- [Direct archive](https://data.nasa.gov/docs/legacy/CMAPSSData.zip)
- [NASA PCoE source and mirror attribution](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)

The verified inner archive SHA-256 is `74bef434a34db25c7bf72e668ea4cd52afe5f2cf8e44367c55a82bfd91a5a34f`. The mirror wraps that archive in an outer ZIP. The downloader validates the inner archive and all twelve numeric files before extraction. It writes only named expected files, not arbitrary archive paths.

| Subset | Train engines / rows | Test engines / rows | RUL rows |
|---|---:|---:|---:|
| FD001 | 100 / 20,631 | 100 / 13,096 | 100 |
| FD002 | 260 / 53,759 | 259 / 33,991 | 259 |
| FD003 | 100 / 24,720 | 100 / 16,596 | 100 |
| FD004 | 249 / 61,249 | 248 / 41,214 | 248 |

The source prose reverses the FD004 train/test engine counts. Numeric files, checked IDs and RUL row counts establish the counts above. Each measurement row contains 26 fields: unit, cycle, three operating settings and 21 sensor readings. The source's final-column description contains a numbering typo; there are 21 sensors, not 26.

Structural checks cover finite values, ordered consecutive integer cycles, expected unit IDs, duplicate rows, identical complete trajectories and label vector alignment. Test structure is inspected during the audit, but no model outcomes or test-distribution plots are used in development.

Training engines are split using NumPy's PCG64 generator seeded with 20261007, independently for each subset. The first floor(0.8 × engine count) shuffled IDs form training. Validation IDs and four endpoints per motor are recorded in the versioned split manifest. Training/validation counts: 80/20, 208/52, 80/20 and 199/50.

## Rights and citation

The NASA data catalog states "License not specified". This project does not assign MIT to the dataset or imply NASA endorsement. Download the original files from the linked source; do not add raw archives or CSVs to Git. NASA PCoE requests acknowledgment of the repository and data donors and provides the data without liability for resulting systems.

A. Saxena and K. Goebel (2008), *Turbofan Engine Degradation Simulation Data Set*, NASA Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA.

Method reference: A. Saxena, K. Goebel, D. Simon and N. Eklund, [Damage Propagation Modeling for Aircraft Engine Prognostics](https://ntrs.nasa.gov/citations/20090029214), PHM 2008.
