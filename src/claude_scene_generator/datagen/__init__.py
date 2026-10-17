"""Dataset generation: an LLM writes scene specs through the tool surface; every run is collected as dataset records.

    csg generate "a small machine shop with two lathes"      # one prompt -> built scene + dataset record
    csg dataset export --format prompt_spec,sft              # training files from everything collected
"""
