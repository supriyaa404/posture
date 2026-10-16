# POSTURE

**Pose-based Online System for Tracking, User Rehabilitation & Exercise**

A privacy-first, browser-based aid that helps patients follow therapist-prescribed rehabilitation exercises at home. Pose tracking runs entirely in the browser; only numeric telemetry is sent to the server.

> **Scope:** POSTURE is a self-guided rehabilitation support tool. It is **not** a diagnostic system, does **not** predict recovery time, and does **not** replace professional medical advice.

## Status

🚧 **Phase 1: repository structure only.** No application code has been written yet. Nothing in this repository has been tested or evaluated at this stage.

## Team

- Nilesh Sharma (BEB243)
- Supriya Yadav (BEB261)
- Guide: Prof. Megha Madhavi
- Lokmanya Tilak College of Engineering, B.E. Computer Engineering, Major Project 2026-27

## Planned features

- Patient registration, login and Google Sign-In
- Upload of a doctor's prescription PDF, extraction of exercises and targets, patient review before confirming
- Personalised exercise schedule with rest days, reminders (Google Calendar) and session tracking
- In-browser pose tracking (MediaPipe Pose Landmarker), joint angles, repetition counting and real-time feedback
- PostureNet: a compact temporal 1D CNN (proposal, to be validated) for form quality, camera yaw and calibrated confidence, with a "can't judge from here" option
- Pain ratings, check-ins, reason-logged skipped sessions, progress reports and a therapist dashboard
- Numeric telemetry only; no raw video is stored

## Planned technology

| Layer | Technology |
|---|---|
| Frontend | Next.js, React, TypeScript |
| Pose tracking | MediaPipe Pose Landmarker (WebAssembly, Web Worker) |
| Model | PyTorch (training), ONNX, ONNX Runtime Web (in-browser inference) |
| Backend | Python, FastAPI |
| Database | PostgreSQL |
| Auth / integrations | Google Sign-In, Google Calendar API |
| Tooling | VS Code, Git, GitHub, Docker |

## Repository structure

```
frontend/     Next.js web application
backend/      FastAPI service (api, core, db, models, schemas, services, tests)
database/     PostgreSQL init scripts and schema documentation
dataset/      Data files (contents are git-ignored; folders are kept)
ml/           Synthetic data generation, preprocessing, baselines, training, evaluation, ONNX export
tests/        Integration and end-to-end tests
docs/         Architecture, privacy, ethics, decisions, report, presentations
demo/         Screenshots and fake sample files for demonstrations
scripts/      Helper scripts
```

## Privacy principles

- Webcam video is processed in the browser and is never uploaded or stored.
- The server stores numeric telemetry only (repetitions, range of motion, scores, pain ratings, skip reasons).
- Secrets (`.env`, API keys) are never committed.
- Only fake data is used in `demo/`.

## Getting started

Setup instructions will be added as each component is built.

## License

To be decided.