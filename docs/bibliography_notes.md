# Bibliography Study Notes (reference only, not the submission)

> These are **reading notes** to help us understand the papers. Per the assignment, the
> annotated-bibliography *summaries* must be written by us, not by AI. Use these notes to check
> your understanding, then write each summary yourself from your own reading.
> Items marked ⚠ could not be confirmed against the paper's full text; check them in the PDF.

## Requirement checklist (7 papers)
| # | Paper | Peer-reviewed venue | ≥ Nov 2022? | Cites another paper on this list (verified) |
|---|---|---|---|---|
| 1 | Sagawa et al. 2020 | ICLR 2020 | no | no (cites Hashimoto 2018, not on list) |
| 2 | Sohoni et al. 2020 | NeurIPS 2020 | no | **yes**: Sagawa |
| 3 | Koenecke et al. 2020 | PNAS (journal) | no | no |
| 4 | Dheram et al. 2022 | Interspeech 2022 (Sept) | no (just outside) | **yes**: Koenecke |
| 5 | Veliche & Fung 2023 | ICASSP 2023 | **yes** | **yes**: Dheram, Koenecke |
| 6 | Radford et al. 2023 | ICML 2023 | **yes** | **yes**: Koenecke ⚠ (seen in an extracted reference list, not the PDF) |
| 7 | Swain et al. 2026 | ICASSP 2026 | **yes** | **yes**: Sagawa, Radford, Dheram, Koenecke |

The requirements are met with margin: 7 ≥ 5 papers; 4–5 ≥ 3 cite another paper on the list; 3 ≥ 3 recent (Veliche, Radford, Swain).
Optional 8th paper: van Zee et al. (2024, Findings of NAACL), which found that bigger Whisper models had *larger* worst-case group gaps. We couldn't read its PDF, so its reference list hasn't been checked.

---

## 1. Sagawa, Koh, Hashimoto & Liang (2020): Group DRO
**APA:** Sagawa, S., Koh, P. W., Hashimoto, T. B., & Liang, P. (2020). Distributionally robust neural networks for group shifts: On the importance of regularization for worst-case generalization. *International Conference on Learning Representations (ICLR 2020)*. https://openreview.net/forum?id=ryxGuJrFvS

- **Problem:** models with high average accuracy can fail badly on rare groups when they rely on spurious correlations (e.g. "waterbird" ↔ water background).
- **Method:** groups are defined by (label, spurious attribute) pairs. Instead of minimizing average loss, minimize the *worst group's* loss. In practice, an online algorithm keeps a weight per group and increases the weight of high-loss groups each step.
- **Key finding (the subtitle):** naive Group DRO on large networks ≈ ERM, because the model drives *every* group's training loss to ~0, which leaves nothing to reweight. It only works with **strong regularization**: much stronger L2 (λ = 1.0 on Waterbirds vs. the default 0.0001) or early stopping.
- **Numbers (worst-group test accuracy, early stopping), ERM → Group DRO:** Waterbirds 6.7 → 86.0; CelebA 25.0 → 88.3; MultiNLI 66.0 → 77.7. Average accuracy drops only about 1–3%.
- **Also:** a "group adjustment" term C/√n_g that favors small groups.
- **Hard parts / honest "didn't understand" candidates:** why zero training loss means worst-group loss is also zero (the generalization-gap argument); the convergence proof (convex case only); how C was tuned.
- **Cross-refs:** cites Hashimoto et al. 2018 ("fairness without demographics") as related DRO work. **Is cited by** Sohoni (2020) and Swain (2026).
- **Bearing on project:** this is the training algorithm for conditions 2–4. Its regularization finding tells us to use strong weight decay and early stopping, or Group DRO will collapse to ERM.

## 2. Sohoni, Dunnmon, Angus, Gu & Ré (2020): GEORGE
**APA:** Sohoni, N., Dunnmon, J., Angus, G., Gu, A., & Ré, C. (2020). No subclass left behind: Fine-grained robustness in coarse-grained classification problems. *Advances in Neural Information Processing Systems 33 (NeurIPS 2020)*, 19339–19352. ⚠ (check the page numbers on papers.nips.cc)

- **Problem:** "hidden stratification." A class contains unlabeled subclasses (e.g. "malignant" lesions with and without some visual feature). A model can do well overall but badly on a subclass nobody labeled.
- **Method (2 steps):** (1) train ERM on the coarse labels, take penultimate-layer features, reduce dimensions (UMAP ⚠), and cluster *within each class* (Gaussian mixture ⚠, k chosen by silhouette score). (2) Treat the clusters as groups and train Group DRO on them.
- **Key results:** reduces average worst-subclass *error* by 22% across 4 image datasets (U-MNIST, Waterbirds, ISIC, CelebA). Group DRO with *true* subclass labels (the "oracle") reduces it by 51%, so GEORGE closes roughly half the gap. Up to +22 points worst-subclass accuracy. With better pretrained features (BiT), it sometimes nearly matches the oracle.
- **Caveats:** the clusters are imperfect (e.g. low precision on a Waterbirds subclass ⚠); no gain on ISIC ⚠; the theory assumes Gaussian-mixture features.
- **Hard parts:** the theory section (Gaussian-mixture assumptions, total-variation bounds); why clustering ERM features finds subclasses at all.
- **Cross-refs:** builds directly on Sagawa et al. 2020. Group DRO is step 2, and true-label Group DRO is its oracle baseline.
- **Bearing on project:** this is *the recipe we transfer to speech*. Our conditions 2 vs. 3 mirror its oracle-vs-GEORGE comparison, and "fraction of gap closed" is the same headline metric.

## 3. Koenecke et al. (2020): racial disparities in ASR
**APA:** Koenecke, A., Nam, A., Lake, E., Nudell, J., Quartey, M., Mengesha, Z., Toups, C., Rickford, J. R., Jurafsky, D., & Goel, S. (2020). Racial disparities in automated speech recognition. *Proceedings of the National Academy of Sciences, 117*(14), 7684–7689. https://doi.org/10.1073/pnas.1915768117

- (We haven't fetched this one yet. Read it yourself.) Known headline: five commercial ASR systems (Amazon, Apple, Google, IBM, Microsoft) had an average WER of about 0.35 for Black speakers vs. 0.19 for white speakers, on matched interview audio (CORAAL vs. VOC corpora). The gap traced to acoustic and phonological differences rather than vocabulary.
- **Cross-refs:** cited by Dheram, Veliche & Fung, Swain, and (⚠) Radford. This is the field's standard motivation citation.
- **Bearing on project:** motivation. It shows that ASR gaps across speaker groups are large and real, which is why fairness methods matter.

## 4. Dheram et al. (2022): discovering and mitigating ASR disparities (Amazon)
**APA:** Dheram, P., Ramakrishnan, M., Raju, A., Chen, I.-F., King, B., Powell, K., Saboowala, M., Shetty, K., & Stolcke, A. (2022). Toward fairness in speech recognition: Discovery and mitigation of performance disparities. *Proc. Interspeech 2022*, 1268–1272. https://doi.org/10.21437/Interspeech.2022-10816

- **Discovery, 2 ways:** (a) *geodemographic*: ZIP codes where at least 75% of residents share one census category; (b) *automatic*: speaker embeddings → k-means (K = 50) → rank clusters by mean ASR confidence; the bottom 10% form the "bottom cohort."
- **Finding:** automatic clusters found a *bigger* gap (65.0% relative WER gap) than geography did (41.7%). Unsupervised clustering surfaces underserved speakers.
- **Mitigation** (geographic cohorts only): (1) oversample bottom-cohort data with machine-generated (teacher) transcripts, cutting the gap from 56.3% to 46.2%; (2) give the model a one-hot "cohort" input, cutting it to 38.5%. Combining both didn't help further, which the authors call counterintuitive.
- **Caveat worth noting:** top-cohort WER got slightly worse (−1.6% WERR) in every mitigation run. The data is internal Alexa data and was not released.
- **Hard parts:** the PDR human-disagreement metric; using ASR confidence as a stand-in for accuracy; RNN-T details.
- **Cross-refs:** cites Koenecke 2020. **Is cited by** Veliche & Fung (2023) and Swain (2026).
- **Bearing on project:** evidence that speaker-embedding k-means clusters capture fairness-relevant structure. It never tried Group DRO on those clusters, which is our gap.

## 5. Veliche & Fung (2023): unsupervised clustering for fair ASR (Meta)
**APA:** Veliche, I.-E., & Fung, P. (2023). Improving fairness and robustness in end-to-end speech recognition through unsupervised clustering. *IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP 2023)*. ⚠ (add the DOI/pages from IEEE Xplore)

- **Constraint:** privacy. No demographics, ZIP codes, or raw speaker embeddings may be used in training.
- **Method:** 192-dim speaker-ID embeddings for each 10 s chunk → PCA → k-means (K = 50, elbow method). The cluster ID goes into the RNN-T *encoder* as an extra one-hot input. 10% of the time it is replaced with "unknown", and at test time *every* utterance gets "unknown", so no speaker model is needed at inference.
- **Results:** WER improved for *every* demographic group on an in-house voice-command set: e.g. non-native US-English 8.52 → 8.44; East Asian 6.39 → 5.89; female 6.42 → 6.02. On Casual Conversations, male speakers saw 17.54 → 15.58. About 80M-parameter model trained from scratch on about 22k hours.
- **Caveats:** they can't say *what* the clusters capture (by design); non-native accents gained least (only 0.93% relative); no Group DRO.
- **Hard parts:** why an input that is always "unknown" at test time still helps (it seems to act as a training regularizer or curriculum); Emformer/RNN-T.
- **Cross-refs:** cites Dheram 2022 (as prior cohort-discovery work) and Koenecke 2020.
- **Bearing on project:** the most direct evidence that clusters of voice embeddings help ASR fairness. We test the *other* way of using such clusters: as Group DRO groups, following Sohoni.

## 6. Radford et al. (2023): Whisper
**APA:** Radford, A., Kim, J. W., Xu, T., Brockman, G., McLeavey, C., & Sutskever, I. (2023). Robust speech recognition via large-scale weak supervision. *Proceedings of the 40th International Conference on Machine Learning (ICML 2023)*, PMLR 202, 28492–28518.

- **What it is:** an encoder–decoder Transformer trained on about 680k hours of weakly labeled internet audio. The input is a 30 s, 80-channel log-mel spectrogram. The decoder predicts text tokens after special tokens (language, transcribe/translate, timestamps).
- **Sizes:** tiny 39M, **base 74M** (ours), small 244M, medium 769M, large 1550M parameters.
- **Key result:** "robustness." Zero-shot Whisper-large matches wav2vec 2.0 on LibriSpeech-clean (2.7 WER) but makes 55% fewer errors averaged across other datasets (e.g. Common Voice 29.9 → 9.0).
- **Accents:** no dedicated accent-fairness analysis in the main text.
- **Hard parts:** the multitask token format; the text normalizer used for scoring; scaling-law plots.
- **Cross-refs:** ⚠ cites Koenecke 2020 (seen in an extracted reference list; confirm in the PDF). **Is cited by** Swain (2026).
- **Bearing on project:** the model we fine-tune. Its loss is token-level cross-entropy over 30 s-padded audio, which is why the CTC-length issue fixed by CTC-DRO doesn't apply to us.

## 7. Swain et al. (2026): fairness-prompted fine-tuning for L2 speakers
**APA:** Swain, M., Maji, B., Mishra, J., Schedl, M., Søgaard, A., & Jensen, J. R. (2026). Towards fair ASR for second language speakers using fairness prompted finetuning. In *Proceedings of the 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP 2026)*. (arXiv:2510.18374 states "Accepted in ICASSP 2026"; add the DOI/pages if they're on Xplore)

- **Setup:** Whisper (tiny–large) and SeamlessM4T fine-tuned on **EdAcc** (26 accent groups) *with true group labels*. They compare ERM, Spectral Decoupling, Group DRO, IRM, and a weighted "Fusion" of all of them.
- **Results (macro-avg WER / max–min gap):** Whisper-large: zero-shot 58.3/114.0 → ERM 26.7/30.1 → **DRO 33.3/37.6 (worse than ERM)** → Fusion 24.1/30.8. On Seamless, DRO *did* beat ERM (e.g. Seamless-medium 40.5 → 26.8 macro WER).
- **Caveats (from reading the table):** Fusion beats ERM on average WER everywhere but doesn't always shrink the gap (Whisper-large gap 30.1 → 30.8). No seeds or significance tests are reported. Urdu and Indian English accents didn't improve.
- **Hard parts:** IRM and spectral decoupling penalties; how the loss weights were tuned; the adapter architecture (not described).
- **Cross-refs:** cites Sagawa 2020 (Group DRO), Radford 2023 (Whisper), Dheram 2022, and Koenecke 2020.
- **Bearing on project:** the closest prior work and our "true labels" reference point. Its finding that Group DRO hurt Whisper is a warning. We need to regularize carefully (per Sagawa), and a negative result for condition 2 is plausible.
