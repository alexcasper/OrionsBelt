# Tests

Two distinct jobs, and conflating them is a mistake:

1. **Unit tests** — metric computation, results-schema conformance, manifest capture,
   and the partitioning logic. Fast, no hardware, run in CI (`t-harness-tests`).
2. **Correctness oracle** — golden logits and perplexity compared against the x86
   reference with explicit, justified tolerances, including long-context prompts where
   recurrent-state drift compounds (`t-oracle`).

The oracle is what makes PLAN.md section 9's rule enforceable: *speed that changes
outputs is not speed*. Every optimization bead — NPU quantization, the GPU scan kernel,
the dispatcher — is gated on it.
