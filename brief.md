

**Mission**: Build a reproducible research repository that deploys and optimizes a Qwen3.5-family model (leveraging its Gated DeltaNet hybrid architecture) on the Radxa Orion O6, with a stretch goal of implementing or benchmarking Gated DeltaNet-2 as a forward-looking architectural comparison, targeting the Physical AI or Cloud AI track of the Arm challenge.

**Phase 1 — Model and scale selection.** Qwen3.5 ships from 0.8B to 397B-A17B parameters, so the agent should target a small dense or lightly-sparse variant (0.8B–4B range) that plausibly fits within the O6's 64GB LPDDR5 and its "within ten billion parameters" NPU inference ceiling [7][2]. Because GDN's linear-attention layers decode with O(1) memory per token versus full attention's growing KV-cache, this is precisely the architecture that benefits most from edge deployment — the agent should frame the project around demonstrating that advantage concretely on constrained hardware [5][8].

**Phase 2 — GDN operator mapping to Orion O6 heterogeneous compute.** The core technical challenge is mapping GDN's chunkwise WY-style recurrent update (delta rule + gated decay + causal Conv1D) onto the P1 SoC's NPU (INT4/INT8/FP16), GPU (Vulkan/OpenCL compute shaders), and CPU big.LITTLE cores, since GDN layers are not standard attention and may not have out-of-box NPU kernel support in the CIX NOE Compiler [9][3]. The agent should profile where GDN's linear recurrence bottlenecks occur (state update sequentiality vs. the chunk-parallel training formulation) and decide whether to run GDN layers on GPU/CPU while offloading the periodic full-attention layers and MoE FFN blocks to the NPU, since NPU accelerators are typically tuned for dense matmuls rather than recurrent scans [3][9].

**Phase 3 — GDN-2 stretch goal.** As an ambitious differentiator, the agent should clone the NVLabs GDN-2 reference implementation and either (a) benchmark GDN-2's channel-wise erase/write gating against standard GDN purely as a research comparison on the O6, or (b) attempt a small-scale GDN-2 layer swap into an open Qwen3.5-architecture checkpoint to test whether the decoupled gating improves long-context retrieval quality at edge-appropriate model sizes, directly citing the RULER retrieval gains reported in the original paper as the hypothesis to validate [6]. This targets the "WOW factor" (25 points) and "Potential Impact" (20 points) judging criteria by producing genuinely novel, reusable research rather than a standard inference demo [10].

**Phase 4 — Benchmarking and metrics.** The agent should measure and report tokens/sec, time-to-first-token, and memory footprint at increasing context lengths (e.g., 4K, 32K, 128K, 262K if feasible) to showcase GDN's near-linear scaling advantage over quadratic attention directly on Orion O6 hardware, using Arm Performix for standardized reporting as required by the challenge brief [10][11]. A comparison table of full-attention-only vs. hybrid GDN vs. (if implemented) GDN-2 across these metrics would make a compelling core result for the write-up.

**Phase 5 — Repo and submission structure.** Structure the repository with a clear README explaining the GDN/GDN-2 background for judges unfamiliar with the architecture, reproducible setup scripts for the CIX NOE Compiler and Python 3.8 NPU toolchain, benchmark scripts outputting CSV/plots, and an OSI license (MIT/Apache 2.0) visible in the GitHub About section from day one [9][10]. The write-up should explicitly frame the project as "optimizing a genuinely novel architecture (Gated DeltaNet) for Arm edge silicon before broad tooling support exists," which plays directly to the "Arm-specific optimization" and technological implementation criteria (40 points) [10].

### Architecture Comparison Reference

| Feature | Gated DeltaNet (Qwen3.5) | Gated DeltaNet-2 |
|---|---|---|
| Gating | Single scalar gate ties erase + write | Separate channel-wise erase (b_t) and write (w_t) gates [6] |
| Relation | Combines Mamba2 decay + delta rule [2] | Generalizes GDN and KDA as special cases [6] |
| Layer ratio in Qwen3.5 | 3:1 linear:full attention across 60 layers [12] | Not yet integrated into a released Qwen checkpoint [6] |
| Best benchmark gains | Long-context efficiency, 262K–1M context [8] | Long-context RULER multi-key retrieval [6] |
| Reference code | huggingface.co/collections/Qwen/qwen35 [13] | github.com/NVlabs/GatedDeltaNet-2 [6] |

Citations:
[1] Qwen3.5: Nobody Agrees on Attention Anymore https://huggingface.co/blog/mlabonne/qwen35
[2] Qwen 3.5: A Complete Model Family from 0.8B to 397B https://enclaveai.app/blog/2026/03/08/qwen-3-5-complete-model-family-local-ai/
[3] qwen3.5-gated-deltanet-analysis https://gist.github.com/justinchuby/0213aa253664fb72e9adb0089816de15
[4] Qwen 3.5 Medium Models: Benchmarks, Pricing, and Guide https://www.digitalapplied.com/blog/qwen-3-5-medium-model-series-benchmarks-pricing-guide
[5] Qwen — Open Foundation Models https://qwen.moe/
[6] Gated DeltaNet-2: Decoupling Erase and Write in Linear ... https://arxiv.org/abs/2605.22791
[7] Radxa Orion O6 https://radxa.com/products/orion/o6/
[8] Qwen 3.5: 397B MoE, 17B Active, 262K Context (2026 Guide) https://www.morphllm.com/qwen-3-5
[9] NPU SDK Installation - Orion O6 - Radxa Docs https://docs.radxa.com/en/orion/o6/app-development/artificial-intelligence/npu-introduction
[10] Rules - Arm Create: AI Optimization Challenge https://arm-ai-optimization-challenge.devpost.com/rules
[11] Arm Create: AI Optimization Challenge: One challenge. Three ... https://arm-ai-optimization-challenge.devpost.com/
[12] Hybrid Attention (Linear + Full) Qwen3.5 is a native vision-language model ... https://x.com/ZhihuFrontier/status/2031686944040915152
[13] Alibaba Launches Qwen3.5: The 397B MoE Giant Built for the Agent Era https://www.youtube.com/watch?v=-R7usl9tBlU
[14] [Deep Dive] Qwen 3.5 Brings Native Multimodality and Long ... https://trilogyai.substack.com/p/deep-dive-qwen-35-brings-native-multimodality
[15] Qwen3.5 https://huggingface.co/docs/transformers/model_doc/qwen3_5
[16] raw - Hugging Face https://huggingface.co/Qwen/Qwen3.5-35B-A3B/raw/main/README.md
[17] while GLM-5 largely follows DeepSeek-V3, Qwen went in a ... https://x.com/zhuokaiz/status/2023790799666770361
[18] Qwen3.5: Towards Native Multimodal Agents https://simonwillison.net/2026/Feb/17/qwen35/
[19] 【Qwen3.5 系列技术解析】2026 年原生多模态旗舰模型全景报告 https://developer.aliyun.com/article/1730072

Competition brief
The arm competition is here 
https://arm-ai-optimization-challenge.devpost.com/

## Competition Context

The Arm Create: AI Optimization Challenge 2026 runs its submission period through August 14, 2026, with judging concluding by September 4 and winners announced around September 15 [1]. It offers $8,000 total across an Overall Winner ($3,000), Overall Runner Up ($2,000), and three category prizes ($1,000 each) for Physical AI, Cloud AI, and Mobile AI [1]. Given your edge computing and embedded systems background, the **Physical AI track** (robotics, embedded devices, edge environments) is the best fit for an Orion O6-based project, though the board's server-class specs also make it viable for Edge/Cloud-adjacent inference work [1].

## Target Hardware Profile

The Orion O6 is built around the Cix P1 SoC (6nm, Armv9.2) with a 12-core CPU (4x Cortex-A720 big, 4x A720 medium, 4x A520 little), an Arm Immortalis G720 MC10 GPU with ray-tracing, and up to 64GB LPDDR5 delivering over 100GB/s bandwidth [2]. Its combined AI throughput reaches up to 45 TOPS (NPU + CPU + GPU), with the NPU alone offering ~28.8 TOPS across INT4/INT8/INT16/FP16/BF16/TF32 precisions, and it's rated to process roughly 30 tokens/sec on a Qwen2-1.5B model [3][4]. It ships with native Debian 12 support, UEFI/BIOS boot, PCIe Gen4 x16 expansion, dual 5GbE, and Wi-Fi 7, making it a genuinely capable "AI PC" form factor rather than a typical SBC [3].

## Agent Brief

**Mission**: Build and document a reproducible research repository that optimizes an AI workload specifically for the Orion O6's heterogeneous CPU+GPU+NPU architecture, producing measurable performance gains and reusable artifacts suitable for the Physical AI track submission.

**Phase 1 — Environment and toolchain setup.** The agent should first establish the CIX P1 NPU SDK toolchain, which requires enrolling in the CIX Early Bird Program to access the NOE Compiler and CIX AI Model Hub, and setting up a Python 3.8 environment (via conda or venv) since the NOE Compiler has this hard version dependency [4]. It should also pull the CIX SDK from GitLab via `repo init`/`repo sync` for kernel and BSP-level work if deeper hardware access is needed [5], and clone the Radxa-maintained `orion-o6` GitHub repo for base system images [6]. Document every setup step meticulously since judges score "Developer Experience" (15 points) on how clearly others can reproduce the build [1].

**Phase 2 — Define the optimization target.** Rather than a generic demo, the agent should pick one concrete, benchmarkable workload aligned with your existing expertise — a strong candidate is quantizing and deploying a small-to-mid LLM (in the Qwen2/Llama-3.2/Phi-3 class, given the board's stated ~10B parameter ceiling) or a vision/graph-based edge inference model, then optimizing it across the NPU (INT4/INT8), GPU (Vulkan/OpenCL compute), and CPU (big.LITTLE scheduling) to hit measurable improvements in tokens/sec, latency, or model size, which are exactly the metrics the challenge highlights [7][4]. A heterogeneous-scheduling demo — dynamically routing inference across NPU+GPU+CPU depending on load — would strongly showcase "Arm-specific optimization" and stand out on the "WOW factor" criterion worth 25 points [7].

**Phase 3 — Benchmarking and repo structure.** The agent should use Arm Performix (referenced in the challenge brief) to generate standardized Arm performance benchmarks and report before/after numbers transparently [7]. The repository must include an OSI-approved license (MIT or Apache 2.0) visible in the GitHub "About" section immediately, since this is a hard submission requirement, not optional polish [1]. Structure the repo with clear setup instructions runnable on the actual Orion O6 hardware, a documented model/optimization pipeline, benchmark scripts producing reproducible CSV/plots, and a written project overview covering purpose, functionality/output, and why it should win — mirroring the exact write-up sections judges expect [1].

**Phase 4 — Submission polish.** Since a demo video under 3 minutes is optional but explicitly said to help judges substantially, the agent should plan to script and record a short clip showing the optimization running live on the physical Orion O6 board, timestamped clearly against the "Functionality" judging point [1]. Given the board's availability guarantee until at least September 2029, framing the project as a durable, reusable reference implementation (rather than a one-off hack) will help on the "Potential Impact" criterion (20 points), which rewards reusable optimized models, migration templates, or learning-ready content [2][1].

### Submission Requirements Checklist

| Requirement | Detail | Source |
|---|---|---|
| Track selection | Choose one of Physical AI, Cloud AI, Mobile AI | [1] |
| Repository | Public, open-source, MIT or Apache 2.0 license visible in About section | [1] |
| Write-up | Project overview, functionality/output, setup instructions | [1] |
| Demo video (optional) | Under 3 minutes, shows device functioning, hosted on YouTube/Vimeo/Youku | [1] |
| Benchmarking | Use Arm Performix for standardized results | [7] |
| Deadline | Submissions close Aug 14, 2026, 4:00pm PT | [1] |
| Newness | Must be newly built or significantly updated during submission period | [1] |


Citations:
[1] Rules - Arm Create: AI Optimization Challenge https://arm-ai-optimization-challenge.devpost.com/rules
[2] Radxa Orion O6 Product Brief https://dl.radxa.com/orion/o6/docs/radxa_orion_o6_product_brief.pdf
[3] Radxa Orion O6 https://radxa.com/products/orion/o6/
[4] NPU SDK Installation - Orion O6 - Radxa Docs https://docs.radxa.com/en/orion/o6/app-development/artificial-intelligence/npu-introduction
[5] Obtain the Source Code | Radxa Docs https://docs.radxa.com/en/orion/o6/cix-sdk/get-source
[6] GitHub - radxa-build/orion-o6: Orion O6 https://github.com/radxa-build/orion-o6
[7] Arm Create: AI Optimization Challenge: One challenge. Three ... https://arm-ai-optimization-challenge.devpost.com/
[8] Artificial Intelligence https://docs.radxa.com/en/orion/o6/app-development/artificial-intelligence
[9] Orion O6 - Radxa Docs https://docs.radxa.com/en/orion/o6
[10] Orion O6 https://www.radxa.com/products/orion/o6/
[11] Download Resources https://docs.radxa.com/en/orion/download
[12] Latest O6 topics - Radxa Community https://forum.radxa.com/c/orion/o6
[13] 【“星睿O6”AI PC开发套件评测】GPU矩阵指令算力，GPU带宽和NPU算力测试 https://blog.csdn.net/weixin_47569031/article/details/147414324
[14] Build Guide: Radxa Orion O6 AI PC Development Kit https://www.youtube.com/watch?v=LP1kXGeJGBI
[15] Welcome to Arm Community https://developer.arm.com/community
[16] Not Your Usual ARM Board https://www.youtube.com/watch?v=GDDTN421Zl8
