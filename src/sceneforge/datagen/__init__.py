"""Dataset generation: an LLM writes scene specs through the tool surface; every run is collected as dataset records.

    sceneforge generate "a small machine shop with two lathes"      # one prompt -> built scene + dataset record
    sceneforge dataset export --format prompt_spec,sft              # training files from everything collected
"""
