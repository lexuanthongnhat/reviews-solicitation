# Review Solicitation

## To run experiments

Key file: `soli_start.py`

Get help: `python soli_start.py --help`

Some typical runs can be found in `start.sh`, `pipeline.sh` script

## To plot experimental results

Key file: `visualizer.py`

Get help: `python visualizer.py --help`

For example:

```bash
  python visualizer.py edmunds --experiment edmunds_l100_p300_q2_r200
```

where *"edmunds_l100_p300_q2_r200"* is the experiment output's file name without file extension (.pickle)

## Note

Directories *output/* and *plots/* are marked in `.gitignore` file since they are used by default for storing experimental results, plots (just a personal choice.)
