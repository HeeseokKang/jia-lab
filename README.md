# Jia Lab Image Analysis Pipelines

Lead: Heeseok Kang  
Advisor: Bill Jia

## Overview

This repository contains reproducible image-analysis workflows for cell-cycle and voltage-imaging experiments in Jia Lab. The current implementation focuses on FUCCI timelapse data quality control, channel alignment, and preprocessing utilities that support downstream segmentation and tracking analyses.

## Project Structure

- `configs/`: Global project configuration, including canonical data paths and shared runtime parameters.
- `shared/`: Reusable helper modules used across analysis pipelines.
- `fucci-analysis/`: Primary analysis workspace for FUCCI timelapse data validation and quality control.
- `voltage-imaging/`: Placeholder for upcoming ratiometric voltage-imaging analysis workflows.

## Data Paths

- Scratch storage: `/data/Project_Data/Voltage_CellCycle/`
- NAS storage: `/mnt/nas1/Projects/Voltage_CellCycle/Data/`

## Current Priorities (FUCCI)

- Internal consistency metrics (e.g., division and edge-case checks)
- 647-channel background subtraction using media blank references
- Dark-frame imputation informed by brightfield segments
- Multi-well scalability for higher-throughput experiments
