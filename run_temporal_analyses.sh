#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p execution_logs
python3 revision_analysis/fusion_conformance.py > execution_logs/fusion.txt
python3 revision_analysis/bridge_conformance.py > execution_logs/bridge.txt
python3 revision_analysis/kappa/inspect_cohort.py > execution_logs/kappa_load.txt
python3 revision_analysis/kappa/analyze_kappa.py > execution_logs/kappa.txt
python3 revision_analysis/kappa/audit_gaps_and_prediction.py > execution_logs/kappa_audit.txt
python3 revision_analysis/thresholds/compare_confirmation.py > execution_logs/thresholds.txt
python3 revision_analysis/rate_phase/run_rate_phase.py > execution_logs/rate_phase.txt
python3 revision_analysis/rate_phase/cohort_audit.py > execution_logs/cohort.txt
python3 revision_analysis/rate_phase/p5_censor_check.py > execution_logs/censor.txt
python3 revision_analysis/field_feedback.py > execution_logs/field_feedback.txt
printf '%s\n' 'Completed offline analyses; see execution_logs and generated JSON/CSV.'
