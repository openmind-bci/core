# User profiles

A profile lets a returning user skip most of the calibration. It is managed by
`openmind_core.profiles.ProfileStore` and the `openmind-core profile` command.

## What is stored

| File | Content |
|---|---|
| `profile.json` | pseudonymized id, creation/update dates, channels, sampling rate, model name, free notes |
| `reference.npz` | reference covariance matrix, channel labels, optional resting band power |
| `scores.jsonl` | progression scores, one JSON object per line |
| `weights/` | weights of the personal model (never in Git) |

EEG data are health data (GDPR). Nothing else is stored: no name, no raw
recording.

## Where

Profiles live in `$OPENMIND_DATA_DIR/profiles/<user>/` (default
`~/openmind-data/profiles/`), outside every repository. The store refuses a root
located inside a Git working tree. Directories are created with mode `0700` where
the OS supports it. Encryption at rest is not implemented; it may be added later
as an option.

## Pseudonymization

Users are identified by ids such as `P01`, `P02` (`^P[0-9]{2,}$`, the same pattern
as the protocol). Names are refused. Keep the mapping between ids and people
outside the project.

## Deletion (right to erasure)

```bash
openmind-core profile delete P01          # asks for confirmation
openmind-core profile delete P01 --yes
```

This removes the whole `<user>/` directory (metadata, reference, scores, weights)
and prints the number of files erased. Models registered under
`OPENMIND_DATA_DIR/models/` are not touched: do not register personal weights
there.

## Command line

```bash
openmind-core profile create P01 --model base --srate 250 --channels C3,Cz,C4
openmind-core profile list
openmind-core profile show P01
```

Errors are printed as `error: ...` with exit code 1.
