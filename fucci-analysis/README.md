# fucci-analysis

## Purpose

This is the primary project workspace for FUCCI timelapse data quality validation and preprocessing checks prior to segmentation and tracking.

## Key Files

- `src/check_data.py`: Compares raw versus cleaned image counts by channel and verifies alignment targets for BF/561/647 data.
- `src/diagnose_mismatch.py`: Diagnoses channel-index mismatches by reporting channel-specific unique indexes and duplicated index groups.
- `src/visual_qc.py`: Generates visual quality-control figures from selected aligned frames to inspect image quality and channel consistency.
