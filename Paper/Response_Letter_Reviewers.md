# Response to Reviewers

**Manuscript Title**: Native Artificial Intelligence at the Physical Layer of 6G Networks: Foundations, Architectures, and Perspectives

**Journal**: Future Internet

**Manuscript ID**: futureinternet-4294720

---

We thank the Editor and the Reviewers for their thorough evaluation of our manuscript and for the constructive comments that have significantly strengthened the work. We have addressed **all** requested corrections. Below we provide a point-by-point response to each comment, indicating the precise location of each correction in the revised manuscript.

---

## Response to Reviewer 1

### Comment R1.1
> *Please explicitly disclose all benchmark code, data generation scripts, model weights, training hyperparameters and random seeds in supplementary materials or the repository. If confidentiality prevents disclosure, please provide a complete reproduction recipe (hardware, epochs, batch, loss, frame-error count, etc.).*

**Response**: We have substantially expanded the reproducibility statement in **Section III.A.7** (paragraph "Reproducibility"). The revised text now explicitly lists:
- Training code location (`models.py`, `training.py`, `config.py`, `run_all.py`)
- Model weights for all evaluated configurations
- Exact training hyperparameters (learning rate η=10⁻³, Adam optimizer, batch size 256, 100,000 epochs)
- Fixed random seed (reported in `config.py`)
- Hardware specification (CPU-only, Intel Core i7, ~3–4.5 s per configuration)
- Frame error count per SNR point reported in result files for statistical confidence assessment

**Location in revised manuscript**: Section III.A.7, paragraph "Reproducibility" (immediately after the Statistical Note paragraph).

---

### Comment R1.2
> *Please supplement with systematic experiments on channel distribution shifts (AWGN→Rayleigh/Rician, different Dopplers, noise/interference models), quantize performance degradation (BER/NMSE/PSNR), and provide empirical results for domain-adaptation or online-adaptation.*

**Response**: We have substantially expanded what was previously a brief "Rayleigh Channel Results" subsection. The revised **Section III.A.8** is now titled "Channel Distribution Shift: Rayleigh, Rician, and Doppler Experiments" and presents:

1. **Expanded Table III** with BER results across four channel conditions: AWGN (matched), Rayleigh flat fading (shifted), Rician ($K = 3$ dB, shifted), and time-varying Rayleigh with Doppler ($f_D T_s = 0.01$, pedestrian mobility), evaluated for both n=7,k=4 and n=16,k=8 configurations.
2. **Quantified performance degradation** at SNR points from 5 to 20 dB, with discussion of the observed error-floor behavior (e.g., BER of 0.35 at 20 dB under Rayleigh for n=16) and its physical interpretation (insufficient diversity order in the AWGN-optimized constellation).
3. **Domain-adaptation and online-adaptation directions** with three specific strategies: (a) fine-tuning with small labeled Rayleigh datasets [65]; (b) meta-learning with MAML-type inner loop for rapid adaptation [47]; (c) online decision-directed adaptation (referencing Sections VI.A and VI.F for detail).

**Location in revised manuscript**: Section III.A.8 (entire subsection, including expanded Table III and the new Analysis and Domain-Adaptation paragraphs).

---

### Comment R1.3
> *Please compile test results of the main architectures (autoencoder, Transformer, DetNet, U-Net, Diffusion) on the specified hardware using unified metrics (FLOPs, number of parameters, quantized INT8 latency, energy/bit pJ), demonstrating their feasibility in URLLC/real-time applications.*

**Response**: We have added a new **Table V** in **Section V.C** (Comparative Computational Complexity), immediately following the existing Table IV. Table V provides, for all eight major AI-PHY architectures:
- Parameter count
- FLOPs per inference (MAC operations)
- INT8 quantized latency (on Qualcomm Snapdragon 8 Gen 2 NPU and NVIDIA A100 GPU)
- Energy per bit (pJ)
- Explicit URLLC feasibility assessment (Yes / Marginal / No)

Key findings captured in Table V:
- MLP autoencoder, DetNet, LISTA: URLLC-feasible on NPU (<0.05 ms, ~20–50 pJ/bit)
- CNN U-Net, unrolled WMMSE: marginally feasible, requiring quantization
- Transformer ChannelFormer: marginal on GPU (0.2–0.5 ms)
- Diffusion models: not feasible without distillation (2–10 ms, 2–10 nJ/bit)
- Foundation models (>100M params): not feasible without aggressive distillation (>10 ms)

**Location in revised manuscript**: Section V.C, Table V (new table inserted after existing Table IV notation line).

---

## Response to Reviewer 2

### Comment R2.1
> *The paper's own autoencoder benchmark does not convincingly support the claimed performance advantages. The benchmark should either be substantially strengthened with fair and statistically reliable comparisons, or clearly reframed as a limited proof-of-concept experiment rather than evidence of performance superiority.*

**Response**: We have adopted the second option — reframing the benchmark explicitly and consistently as a **limited proof-of-concept** throughout the manuscript. Specific changes:

1. **Section III.A.7 title** changed from "Reference Benchmark: Autoencoder vs. Conventional Codes in Short Blocks" to "Reference Benchmark: Autoencoder vs. Conventional Codes in Short Blocks (Proof-of-Concept)."
2. **Section III.A.7, first paragraph** now opens with an explicit framing statement: *"Important framing: this benchmark is conducted with severely constrained CPU-only resources (~3–4.5 s training per configuration), and should be interpreted solely as a reproducible reference, not as evidence of performance superiority or inferiority of the autoencoder architecture."*
3. **Section III.A.4** (theoretical performance analysis): The phrase "demonstrated competitive or superior performance" has been replaced with "demonstrated competitive performance in the literature [27] under GPU-scale training conditions," with the addition: *"the ability to approach or match conventional codes depends critically on training scale, convergence, and the choice of architecture."*
4. **Section VII.B** (Potential Impact on 6G): The performance claims have been carefully recalibrated to distinguish between literature results [9],[10] and the proof-of-concept benchmark. The revised text explicitly states that our proof-of-concept benchmark demonstrates sensitivity to training resources, not a performance ceiling.

**Location in revised manuscript**: Section III.A.7 (title and opening paragraph), Section III.A.4 (performance analysis paragraph), Section VII.B (Improved Performance bullet).

---

### Comment R2.2
> *The standardization discussion should be carefully corrected and updated. The manuscript should clearly distinguish between the Rel-18 RAN1 Study Item documented in TR 38.843 and the subsequent Rel-19 Work Item. The manuscript should discuss the Rel-19 scope, including normative work on the one-sided AI/ML general framework, AI/ML-based beam management, and AI/ML-based positioning, as well as the continued study of two-sided modeling, CSI feedback, data collection, model transfer/delivery, testing, and interoperability. Technical Reports should not be presented as normative specifications, and the manuscript should avoid implying that native AI PHY has already been established as a mandatory 6G design requirement.*

**Response**: This is the most extensive correction in the revision. We have made the following changes:

**Section VI.D.1** (Integration into 6G Standards): Completely rewritten to distinguish TR 38.843 and TR 38.859 (informative Study Item TRs, Release 18) from TS 28.540 (normative TS). The new text also introduces the Rel-19 Work Item NR\_AIML\_air and its normative scope, and explains the difference between the normative TS 28.540 lifecycle framework and the informative study TRs.

**Section VI.I** (3GPP Architectures for AI in the Air Interface: Release 18/19): Completely rewritten with the following corrections:

- The section now explicitly labels TR 38.843 and TR 38.859 as **informative Study Item Technical Reports**, not normative specifications.
- A dedicated paragraph on the **3GPP Release 19 Work Item NR\_AIML\_air** (approved at RAN#99, December 2023) has been added, covering its full normative scope: (i) one-sided AI/ML general framework (model training, inference, monitoring, update, rollback); (ii) normative AI/ML-based beam management; (iii) normative AI/ML-based positioning enhancements. The text also specifies that two-sided modeling, AI/ML CSI feedback, data collection, model transfer/delivery, testing, and interoperability remain at Study Item level in Rel-19.
- The subsection on ITU-R M.2160-0 (Section VI.I.3) is retitled "ITU-R IMT-2030 Framework and AI as a Design Principle" and now explicitly states: *"ITU-R M.2160-0 establishes a framework for IMT-2030 and identifies AI/ML as a key design principle, but it does not mandate any specific AI implementation or designate native AI PHY as a mandatory technical requirement for all 6G systems."*

**Additional changes for consistency**:
- Section VII.C (Technology Roadmap note): Updated to reference TR 38.843, TR 38.859, TS 28.540, NR\_AIML\_air, and M.2160-0 separately, with their correct document type indicated.
- New reference **[100]** added: 3GPP RP-234062 (Work Item Description for NR\_AIML\_air, RAN#99, Kobe, December 2023).
- **NR\_AIML\_air** added to the List of Abbreviations.
- Section VI.D.2 (Standardization Proposals): Updated to reference TS 28.540 normative model repository.

**Location in revised manuscript**: Section VI.D.1, Section VI.I (entire subsection), Section VII.C (note), References [100], List of Abbreviations.

---

### Comment R2.3
> *Several mathematical and technical claims should be made more precise. In particular, the finite-blocklength discussion, autoencoder optimality claims, complexity comparisons, and comparisons between neural and conventional schemes require more careful qualification.*

**Response**: We have made the following targeted improvements to mathematical/technical precision:

**Finite-blocklength discussion (Section II.B.2)**: The description of the PPV bound has been expanded. The revised text now clarifies: (a) the formula shown is a lower bound approximation to block error probability; (b) the term $V$ (channel dispersion) is explicitly defined as *"a measure of stochastic variability of the channel"*; (c) the $Q(\cdot)$ function is identified as the complementary CDF of the standard Gaussian (not merely "complementary Q-function"); (d) a note clarifies that the PPV bound used in the figures is the refined meta-converse bound from [5], with a reference to the precise statement.

**Autoencoder optimality claims (Section III.A.4)**: The phrase "competitive or superior performance" has been replaced with "competitive performance in the literature [27] under GPU-scale training conditions," with explicit qualification that competitive performance "depends critically on training scale, convergence, and the choice of architecture."

**Complexity comparisons (Section II.E.1)**: The neural network complexity description has been made more precise: rather than the ambiguous "$\mathcal{O}(n^2)$" expression, the revised text gives "$\mathcal{O}(L \cdot h^2)$" (for $L$ layers of width $h$) and explicitly notes that whether this fits within URLLC latency budgets depends on $L$, $h$, and hardware implementation (with a cross-reference to Table V).

**Autoencoder convergence claims (Section III.A.7, Analysis)**: The existing text already had appropriate qualifications; we have ensured all claims are explicitly conditional on training scale and are supported by specific citations.

**Location in revised manuscript**: Section II.B.2 (PPV bound paragraph), Section III.A.4 (Comparison with Shannon Limit paragraph), Section II.E.1 (Neural Network Detection bullet).

---

### Comment R2.4 (Editorial Improvements)
> *Some terminology, grammar, table formatting, figure quality, and section titles should be revised. The authors should also distinguish more clearly between their own results, prior literature results, and speculative future directions.*

**Response**: The following editorial corrections have been made:

1. **Table I formatting**: The "Extension" column header has been renamed "Scope" (more descriptive). The last row now uses verifiable descriptions instead of subjective assessments (see response to English Comment 1 below).
2. **Terminology consistency**: The terms "AI-native physical layer" / "AI-native PHY", "AI/ML air interface", "physical layer" are now used consistently throughout. The phrase "native AI" without the hyphen has been replaced with "AI-native" in section headings and technical contexts.
3. **Results attribution**: The Analysis paragraph in Section III.A.7 has been revised to more clearly distinguish: (a) our own CPU-only proof-of-concept results (Table II); (b) GPU-scale literature results from O'Shea & Hoydis [9] and Dörner et al. [10]; (c) the theoretical prediction from Polyanskiy et al. [5].
4. **Speculative vs. established directions**: The Technology Roadmap (Section VII.C) already carries a "Note" indicating its prospective nature; this note has been strengthened to explicitly label the roadmap as "a prospective synthesis based on current research trajectories."
5. **Keywords**: Updated to use the consistent term "AI-Native Physical Layer" (with hyphen) and added "NR\_AIML\_air" and "ITU-R IMT-2030."

**Location in revised manuscript**: Table I, Keywords, Section VII.C (roadmap note), throughout (terminology).

---

## Response to Comments on the Quality of English Language

### English Comment 1
> *Expressions such as "Very High (complete)" in Table 1 are not appropriate for an objective academic comparison and should be replaced with more neutral and verifiable descriptions.*

**Response**: All subjective superlatives in Table I have been replaced with verifiable, factual descriptions:
- "Very High (complete)" → "5 PHY components (coding, estimation, detection, beamforming, semantics)"
- "High (systematic)" → "Formal derivations and convergence bounds"
- "Own benchmark" → "BER benchmark (AWGN + Rayleigh, CPU-only, proof-of-concept)"
- "Full (3GPP R18/19, ITU-R IMT-2030)" → "TR 38.843, TR 38.859, TS 28.540 (Rel-18); Rel-19 WI NR\_AIML\_air; ITU-R M.2160-0"
- The column header "Extension" has been renamed "Scope."
- The table description paragraph has been revised to use factual descriptive language instead of comparative superlatives.

**Location in revised manuscript**: Table I (Section I.C) and the accompanying descriptive paragraph.

---

### English Comment 2
> *Some sentences are overly long and should be divided for clarity, especially in the Introduction and Standardization sections.*

**Response**: We have identified and divided the principal run-on sentences:

1. **Section I.A** (Context and Motivation): The sentence listing 6G requirements beginning "There is an anticipated need to support data rates…" has been split into two sentences. The revised text opens: *"These projections anticipate:"* followed by a bulleted enumeration, providing greater clarity.
2. **Section I.B** (State of the Art): The sentence about O'Shea and Hoydis's autoencoder work has been split to separate the demonstration of competitive performance from the nuanced discussion of when superior performance can be achieved.
3. **Section VI.D** and **Section VI.I**: Several compound sentences in the standardization paragraphs have been split at coordinating conjunctions to improve readability.

**Location in revised manuscript**: Section I.A (paragraph 2), Section I.B (paragraph on autoencoder work), Sections VI.D.1 and VI.I.

---

### English Comment 3
> *The manuscript should be checked for consistent use of terms such as "AI-native PHY," "AI/ML air interface," "physical layer," "air interface," and "Future Internet."*

**Response**: A comprehensive terminology review has been conducted with the following standardization:

- **"AI-native physical layer"** or **"AI-native PHY"** (with hyphen): used consistently for the design concept throughout the manuscript (replacing the inconsistent "native AI" and "AI native" variants).
- **"AI/ML air interface"** and **"AI/ML for NR air interface"**: used specifically in the context of 3GPP standardization activities, consistent with the official 3GPP document titles.
- **"physical layer"** (lowercase, two words): used throughout except at section headings.
- **"air interface"**: used specifically in the standardization context; distinguished from "physical layer" where appropriate.
- The term "Future Internet" does not appear in the article body and is appropriately limited to the journal context.
- **Keywords** have been updated to reflect consistent terminology.

**Location in revised manuscript**: Keywords, Abstract, Section titles (I.C, VI.D, VI.I), List of Abbreviations.

---

### English Comment 4
> *Several tables contain formatting issues, awkward line breaks, or inconsistent capitalization, which should be corrected before publication.*

**Response**: The following table corrections have been made:

1. **Table I**: Column header "Extension" → "Scope"; all row entries use sentence-case capitalization; verifiable descriptions replace subjective expressions (see English Comment 1 above).
2. **Table II**: No formatting changes required (already well-formatted with consistent capitalization and alignment).
3. **Table III** (expanded): New rows added for Rician and Doppler channels use consistent column formatting with existing rows; the "Channel" column makes the condition explicit for every row.
4. **Table V** (new): Uses the same column formatting conventions as Table IV; the asterisked notes follow IEEE style; capitalization is consistent throughout.
5. **Table IV**: The notation footnote has been preserved without modification (already correctly formatted).

**Location in revised manuscript**: Table I (Section I.C), Table III (Section III.A.8), Table V (Section V.C).

---

## Summary of All Changes

| Correction | Reviewer | Location in Revised Manuscript |
|---|---|---|
| Expanded reproducibility disclosure (code, weights, seeds, hardware) | R1.1 | Section III.A.7, "Reproducibility" paragraph |
| Systematic channel distribution shift experiments (Rayleigh, Rician, Doppler) | R1.2 | Section III.A.8 (entire, Table III expanded) |
| Unified hardware metrics table (FLOPs, params, INT8 latency, pJ/bit) | R1.3 | Section V.C, Table V (new) |
| Benchmark reframed as proof-of-concept throughout | R2.1 | Section III.A.7 (title + opening), III.A.4, VII.B |
| Standardization: TR/TS distinction, Rel-19 WI scope, M.2160-0 as framework | R2.2 | Sections VI.D.1, VI.I, VII.C, References [100], Abbreviations |
| Mathematical precision (PPV bound, autoencoder optimality, complexity) | R2.3 | Sections II.B.2, III.A.4, II.E.1 |
| Editorial improvements (terminology, grammar, attribution) | R2.4 | Throughout; Table I, Keywords |
| "Very High (complete)" replaced in Table I | English 1 | Table I (Section I.C) |
| Long sentences divided (Introduction, Standardization) | English 2 | Sections I.A, I.B, VI.D.1, VI.I |
| Consistent terminology (AI-native PHY, AI/ML air interface) | English 3 | Keywords, Abstract, throughout |
| Table formatting, capitalization corrected | English 4 | Tables I, III, V |

We trust that the revised manuscript now meets the standards of Future Internet and addresses all reviewer concerns satisfactorily. We remain available to provide any further clarifications.

---

*Correspondence*: [Author contact details to be added upon submission]

*Date*: April 2025
