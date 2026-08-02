"""Quantization policies and precision carve-outs.

Owned by bead ``t-quant-plan``. The central concern is that GDN recurrent
state is fed back through every decode step, so quantization error accumulates
over the sequence instead of staying local the way KV-cache error does. State
and gating tensors are therefore the prime candidates for FP16 carve-outs.
"""
