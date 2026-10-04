# Conversation memory: follow-up rewrites (D-71)

Model: `openai/gpt-oss-20b` (`small` role). Produced by `eval/conversation_memory.py`.

**Rewrites judged correct: 12/12. Controls (no call, unchanged): 2/2. Rewrite latency p50: 609 ms.**

A rewrite is correct when the slot extractor, run on the standalone question, finds the expected metric / formula / topic / fiscal period / ask type and no subject the follow-up did not refer to. No model judges another model here.

| Case | Follow-up | Read as | Result |
|---|---|---|---|
| metric swap | And gross profit? | What was the gross profit growth rate in fiscal 2026? | correct |
| metric swap, 'what about' | What about net income? | What was the net income growth rate in fiscal 2026? | correct |
| reason for the previous figure | Why did it grow so fast? | What was the reason for the revenue growth rate in fiscal 2026? | correct |
| period shift | And the year before? | What were total assets as of January 25, 2025? | correct |
| bare period | and FY2025? | What were cash and cash equivalents in fiscal 2025? | correct |
| formula swap | How about the quick ratio? | What is the quick ratio at the latest balance sheet date? | correct |
| bare why | Why? | Why did inventories change year over year in fiscal 2026? | correct |
| topic reference | What are the risks to it going forward? | What are the risks to gross margin in fiscal 2026 going forward? | correct |
| compare with earlier subject | Compare that with operating income | How does net income compare with operating income in fiscal 2026? | correct |
| figure follow-up | Which layer is at the bottom? | Which layer is at the bottom of NVIDIA's five-layer cake? | correct |
| event follow-up | Can shareholders attend it online? | Can shareholders attend NVIDIA's 2026 annual meeting online? | correct |
| formula, change over time | Is that higher than last year? | Is the total debt to equity ratio for fiscal 2026 higher than that for fiscal 2025? | correct |
| control: self-contained | What is NVIDIA's five-layer cake? | *(unchanged, no call)* | correct |
| control: no conversation | And gross profit? | *(unchanged, no call)* | correct |

Caveats: 14 cases, written by hand against this report; a single run of a sampled model. The cases cover the follow-up shapes seen in use (metric swap, period shift, bare *why*, reference to an earlier subject or topic, comparison) — not every phrasing a visitor will type. The rules that decide whether to call the model are covered by `tests/test_condense.py`.
