"""Post-hoc evaluation of built scenes (E1 LLM runs, E2 baselines): prompt adherence and layout plausibility.

Evaluation only: nothing here runs inside ``check_scene``, emits issues to the generation / repair loop, or changes what an
accepted run is.  ``worldc eval --runs DIR --prompts prompts.csv --out eval.csv`` (see ``run.py``).
"""
