"""Replication harness for arXiv:2511.09700, "Order Matters: Rethinking Prompt
Construction in In-Context Learning" (Li, Wang, Wang, Shang, Nov 2025).

Modules
-------
specs      dataset registry: which HF dataset, how an example is rendered, label words
data       build balanced dev/test splits and demonstration pools; sample demo sets
prompts    default ordering, random permutations, prompt text
scorer     model wrapper: encode the shared demonstration prefix once, score queries
metrics    ordering / selection sensitivity and r (paper Eqs. 1-3)
search     Section 4 aggregates: highest-dev, average, oracle, curves
runner     evaluate an M x P grid of (demo set, permutation) on a list of examples
parse      answer extraction and numeric match for generation tasks
"""
__version__ = "0.1.0"
