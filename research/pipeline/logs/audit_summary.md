# Data Audit and Completeness Gate (G1) Summary Report

**Status:** PASS
**Timestamp:** 2026-07-17 11:50:58

## 1. StressID Dataset (Primary)
- **Total Subjects:** 65 (verified 65 subjects)
- **Class Balance (Task-level):**
  - Stressed: 368
  - Non-stressed: 332
  - Ratio (Stressed/Total): 52.57%
- **Modality Completeness:**
  - Face (Video): 53/65 (81.54%)
  - Voice (Audio): 54/65 (83.08%)
  - Physiology: 65/65 (100.00%)

## 2. EmpathicSchool Dataset (Supplementary)
- **Total Subjects:** 30 (verified 30 subjects)
- **Class Balance (Interval/Task-level):**
  - Stressed: 35
  - Non-stressed: 56
  - Ratio (Stressed/Total): 38.46%
- **Modality Completeness:**
  - Face (Video/Landmarks): 27/30 (90.00%)
  - Physiology: 30/30 (100.00%)

## 3. Gate G1 Evaluation
- **Condition:** Every required modality must be present for at least 80% of each dataset's subjects.
- **Evaluation Result:**
  - StressID Face: PASS (81.54%)
  - StressID Voice: PASS (83.08%)
  - StressID Physio: PASS (100.00%)
  - EmpathicSchool Face: PASS (90.00%)
  - EmpathicSchool Physio: PASS (100.00%)
- **Verdict:** **PASSED**
