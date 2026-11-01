# E2 within-prompt samples (`e2_samples`)

These are extra constrained runs of the E1 protocol, used to measure layout diversity within a prompt. Each prompt gets
5 samples: the E1 run plus 4 extra.
- **Records:** `dataset/e2_samples/deepseek-v4.1-flash_cloud/` (`records.jsonl`, `runs/`). This is its own set: the
  runs are not E1 records.
- **Code:** run from the `e1-freeze-v7` worktree (`../worldc-e1-freeze-v7`, commit `6f9647d`, clean). `run_e1.sh`
  refuses anything else. The protocol is E1's: effort high, max 120 turns, build seed 0, 10 repairs, 3600 s timeout,
  final physics check. Samples of one prompt differ only through the model's sampling.
- **Concurrency:** the batch runs beside the E1 batch (tmux `e1`). Isaac Sim work (physics checks) queues
  machine-wide on the GPU lock.

## Running batch

Started **2026-10-29 20:21:08 EET** in tmux session `e2samples`, from the repo root
`/home/sebnae/ws/ws_startup/claude_scene_generator`, with exactly:

```
R=/home/sebnae/ws/ws_startup/claude_scene_generator
tmux new-session -d -s e2samples -n main "bash -c '$R/scripts/e2_samples.sh P05,P08,P15,P16 4 2>&1 | tee -a $R/dataset/e2_samples/main.log'"
tmux new-window -t e2samples -n p23 "bash -c 'E2_WAIT_FOR=P23 $R/scripts/e2_samples.sh P23 4 2>&1 | tee -a $R/dataset/e2_samples/p23.log'"
```

- **Window `main`:** 4 passes over P05, P08, P15 and P16, 16 runs. Each pass runs
  `cd ../worldc-e1-freeze-v7 && E1_DATASET=$R/dataset/e2_samples E1_PROMPTS=<ids> scripts/run_e1.sh ollama:deepseek-v4.1-flash:cloud`.
- **Window `p23`:** waits, checking every 300 s, until `dataset/e1/deepseek-v4.1-flash_cloud/records.jsonl` has an
  accepted constrained P23 record. Then it runs 4 passes over P23. If E1 rejects P23, it keeps waiting.
- **Logs:** `dataset/e2_samples/main.log` and `p23.log`.

- **Pass 4 redo:** pass 4 of window `main` (started 2026-10-30 00:53:18) was killed by mistake at 01:13 during its first
  run (P05, which wrote no record; its folder `runs/20261029T225320_3f5c9b00` is incomplete and ignored), and the pass
  skipped P08, P15 and P16. It was rerun from 2026-10-30 01:14:20 EET in tmux `e2samples` (log `redo.log`) with
  `scripts/e2_samples.sh P05,P08,P15,P16 1`.

## GLM batch (`glm-5.3-flash:cloud`)

The same design for a second model: 4 extra constrained runs for each of the 5 prompts, its E1 run being sample 1.
Records go to `dataset/e2_samples/glm-5.3-flash_cloud/`. GLM runs at its Ollama thinking default (max), directly
like DeepSeek. Started 2026-10-30 09:19:45 EET in tmux `e2samples_glm`, after GLM's E1 batch and the DeepSeek redo had
both finished (a queued start at 01:38 never fired: its wait loop matched its own tmux session by name prefix):

```
E2_MODEL=ollama:glm-5.3-flash:cloud $R/scripts/e2_samples.sh P05,P08,P15,P16,P23 4 2>&1 | tee -a $R/dataset/e2_samples/glm.log
```

Analysis: `scripts/e2_within.py --e1 dataset/e1/glm-5.3-flash_cloud --samples dataset/e2_samples/glm-5.3-flash_cloud --out results/e2_within_glm`.

## Prompts

| prompt | cell type | size | difficulty | E1 sample |
|---|---|---|---|---|
| P05 | ev_battery_dismantling | M 14 x 9 | hard | `20261029T143353_3f5c9b00` |
| P08 | electronics_weee | M 16 x 10 | medium | `20261029T152717_3dc2eed0` (also the template's prompt) |
| P15 | engine_gearbox_reman | L 26 x 14 | hard | `20261029T172753_91fdb0bb` |
| P16 | appliance_repair | S 9 x 6 | easy | `20261029T174357_1b71d579` |
| P23 | sorting_grading | M 16 x 10 | medium | pending in E1 |

The recipe gets 5 seeds per prompt (base..base+4, base = the prompt number). The template stays the IoU = 1.0
reference.
