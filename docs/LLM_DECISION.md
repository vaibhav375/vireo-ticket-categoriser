# Should the tool use an LLM? Measured answer: no, not on this data

The first version had an optional Claude API step for low-confidence tickets. It was replaced with a
free local LLM (Ollama) so there's no key and no per-call cost. Then it was measured against the
default model. Machine: Apple M1, 8 GB RAM, CPU/GPU shared memory.

| Model | Accuracy, 150 audited tickets | Accuracy, 26 low-confidence tickets* | Median latency / ticket | p95 | Cold start |
|---|---|---|---|---|---|
| **TF-IDF + logistic regression (default at the time)** | **100%** | **88.5%** | **0.0014 s** | — | — |
| qwen2.5:3b (1.9 GB) | 86.7% | 69.2% | 1.5 s | 2.5 s | 2.4 s |
| llama3.2:3b (2.0 GB) | 84.0% | 53.8% | 1.5 s | 2.3 s | 8.5 s |
| qwen2.5:7b (4.7 GB) | 87.3% | 76.9% | 12.2 s | 39.8 s | 21.7 s |

\* The only tickets `--llm` would send: fast-model confidence below 0.6, and the agent's note gives a label to check against.
Measured before the classifier was upgraded to a linear SVM (`docs/MODEL_IMPROVEMENT.md`). The upgrade only widens the gap.
Reproduce with `python run.py --benchmark` (detail and every error in `out/llm_benchmark.md`).

## Why the LLMs lose

The mistakes are about Vireo's routing conventions, not language. For example:
- "Got a different colour than ordered" or "box was crushed": the LLMs say Returns or Warranty, but Vireo routes these to Logistics.
- "Nobody came for the pickup": the LLMs say Delivery, but Returns Desk owns reverse pickups.
- "Typo in the flat number": the LLMs say Delivery, but it's an Order Change handled by frontline.

The fast model learned these conventions from 10,000+ agent notes. A zero-shot 3B model can't. Prompt
tuning might close some of the gap, but the only clean way to measure it is the audit set, and tuning
against that would contaminate the one independent check. So I didn't.

## What happened on the laptop

Running the 7B model froze an 8 GB machine. Ollama keeps a model in memory for 5 minutes after use,
so the benchmark's 7B model (4.7 GB) was still loaded when `--llm` loaded a 3B model (2 GB) next to
the pipeline. macOS swapped until it killed the run (exit 137). Fixes:
- **Size guard.** Models larger than 35% of RAM are refused before loading (7B on 8 GB is refused with a message).
- **One model at a time.** Any other loaded model is unloaded first.
- **Short keep-alive** (30 s), plus an explicit unload when the step finishes, even on error.
- The 7B model is removed from the default benchmark list.

After the fixes, `python run.py --llm` with qwen2.5:3b ran end to end. Free system memory never went
below 19%, Python peaked at 0.4 GB, and the model was unloaded straight after.

## Decision

- The normal run (`python run.py`) uses no LLM. It takes about 0.63 GB of RAM and about 50 s for all 18 months, and it
  would take about 0.9 ms per ticket at intake.
- `--llm` stays as a documented, off-by-default experiment. It is free, but on this data it makes the
  uncertain tickets worse (69% vs 88%).
- When an LLM would earn its place: live messages that look nothing like the training data (new
  products, Hindi/Hinglish, long emails), or to *explain* a routing decision to an agent. Re-run
  `--benchmark` on a hand-labelled sample of live tickets before switching it on.
