"""Dataset generation: an LLM writes scene specs through the tool surface; every run is collected as dataset records.

    worldc generate "a small machine shop with two lathes"      # one prompt -> built scene + dataset record
    worldc dataset export --format prompt_spec,sft              # training files from everything collected
"""
