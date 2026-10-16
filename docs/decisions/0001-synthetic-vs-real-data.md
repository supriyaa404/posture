# Decision 0001: Role of synthetic data versus real data

- **Status:** Accepted
- **Date:** 2026-10-16
- **Deciders:** Project team (Nilesh Sharma, Supriya Yadav)

## Context

Phase 2 produced a synthetic dataset of 2,100 stylised illustrations (1,470 train / 315 validation / 315 test) with exact 13-joint annotations. All 10 automated verifier checks passed and the preview grid was inspected by the team.

The images are rendered figures. They are not real people, not webcam frames and not MediaPipe output. Their form labels come from generator parameters and have not been clinically validated.

## Decisions

1. **Retain** the synthetic dataset for initial data-pipeline and preprocessing development only.
2. **Use real webcam frames processed by MediaPipe** for the real-time demonstration, and collect genuine landmark sequences (numbers only, with volunteer consent) for future evaluation.
3. **Keep synthetic and real data separate.** Synthetic results are never presented as real-world validation.

## Rules that follow from these decisions

- Synthetic data lives only under `dataset/synthetic/`. Real landmark recordings live only under `dataset/landmarks_real/`. Nothing is copied between them.
- Every table, figure, log line and report section that uses synthetic data must say **"synthetic"** in its title or caption.
- Metrics computed on synthetic data may be described as "pipeline checks" or "synthetic benchmark results". They must not be described as accuracy, validation or clinical performance.
- Any statement about real-world accuracy must come from real recordings only.
- The synthetic dataset is not regenerated unless the team explicitly decides so. If it is regenerated, the new seed and settings are recorded here.

## Consequences

- Early development (loading, normalisation, angle maths, baseline plumbing) can proceed without waiting for volunteer recordings.
- Real-world performance is unknown until real recordings are collected and evaluated.
- Real MediaPipe landmarks are noisier and differently distributed than the exact synthetic annotations, so results will not transfer automatically.

## Not decided here

The Presentation II slides describe training PostureNet on UI-PRMD with synthetic viewpoint augmentation. This record does not change or confirm that plan. It will be settled in a later phase.