# Dataset directory

Data files are git-ignored. Only the folders and this README are tracked.

| Folder | Source | Contents |
|---|---|---|
| `synthetic/` | **SYNTHETIC** (generated illustrations) | 2,100 images and annotations from `ml/synthetic_data/` |
| `landmarks_real/` | **REAL** (volunteer recordings, MediaPipe landmarks) | Numeric landmark sequences only, collected with consent. No video. |
| `external/ui_prmd/` | External public dataset | Reserved. Its use is still to be decided. |
| `processed/` | Derived | Windows or arrays produced by the pipeline. Each output must record which source it came from. |

## Separation rules

1. Never copy files between `synthetic/` and `landmarks_real/`.
2. Anything in `processed/` must be named or tagged with its source (`synthetic` or `real`).
3. Any result computed from `synthetic/` is labelled "synthetic" and is not real-world validation.
4. Never store raw webcam video or images of volunteers. Real recordings are landmarks only.
5. Volunteer data requires recorded consent before collection.