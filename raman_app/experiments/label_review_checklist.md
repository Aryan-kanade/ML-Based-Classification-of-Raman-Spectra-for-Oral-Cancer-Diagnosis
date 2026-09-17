# Label-error review checklist — for HUMAN verification
Generated 2026-09-17 from `experiments/label_errors.txt`
(3SSE-d2 winner chain, 287 OOF rows, 2 flagged — tier REVIEW, none
auto-confirmed).  Cleaning a CONFIRMED mislabel is the only remaining
honest lever that raises every downstream number (SUMMARY.md L11).

## How to read this
The chain assigns each spectrum an out-of-fold probability.  A flag
means the GIVEN label got a near-zero probability while the other
class dominates — either a mislabel, or simply a hard/atypical
spectrum.  Only histopathology / patient records can decide.

## Flagged spectra (verify against patient records)

| # | File (under `Data/`) | Given label | Model says (OOF) | Action |
|---|---|---|---|---|
| 1 | `Tumor/TDOC083 Spectra pro/22042026TDOC083TH02.csv` | Normal* | Tumor 0.832 | Open the CSV + check patient TDOC083's report |
| 2 | `Tumor/TDOC085 Spectra pro/29042026TDOC085TH0.csv` | Normal* | Tumor 0.764 | Open the CSV + check patient TDOC085's report |

\* **Mapping caveat**: both rows sit in `Tumor/` folders with Tumor
site tokens (TH*), yet the given label reads 0 (= Normal in the
sorted class order).  That contradiction has two explanations —
check BOTH before touching anything:
  1. **Row-map shift** (most likely): the flagging script rebuilds
     names with a truncated tail map (`eval_label_errors.py:52-53`)
     because paired feature-building may drop leading rows; the
     printed given-label may belong to a different row.  Re-run with
     the corrected map before concluding.
  2. **Genuine mislabel**: the spectrum truly is Tumor tissue and
     the given label is wrong, or vice versa.

## Decision rule (pre-registered, Brain.md §60 discipline)
- If the patient record CONFIRMS the given label → keep; the spectrum
  is a hard case, no change.
- If the patient record CONTRADICTS it → fix the label in the source
  folder (rename/move), re-run the loader + winner chain, re-adjudicate
  with 3 seeds + McNemar.  Only a confirmed, documented mislabel may
  be changed.
- If unclear → no change (do not tune the dataset on model output).

## Why this matters
Two confirmed mislabels on ~290 rows is ~0.7% of the data; past
experience on such datasets puts the honest lift at roughly
+0.005–0.015 F1 — real, but modest.  The record of every change must
go into Brain.md with the patient-record justification.
