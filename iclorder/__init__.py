"""One cell of arXiv:2511.09700 ("Order Matters: Rethinking Prompt Construction in
In-Context Learning"): GPT-5-nano on AG News, ordering vs selection sensitivity.

dataset        AG News constants; build the demo pool / dev / test files; sample demonstration sets
prompts        default ordering, random permutations, prompt text
openai_scorer  call the OpenAI API, read the label from the reply, cache every completion
metrics        ordering sensitivity, selection sensitivity, r (paper Eqs. 1-3)
"""
__version__ = "1.0.0"
