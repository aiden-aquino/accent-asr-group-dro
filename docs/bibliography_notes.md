# Bibliography Study Notes

> **AI-generated study notes, not our annotated bibliography.**
> These notes were produced with Claude (an AI assistant) to help us *understand* the papers:
> facts, numbers, and connections checked against the papers themselves. Per CS 445 policy, the
> annotation text we submit is written entirely by us (it lives in our Overleaf project, with the
> citations in [annote.bib](annote.bib)). Items marked ⚠ have not been confirmed against the paper.

## Requirement checklist (7 papers)
| # | Paper | Peer-reviewed venue | Since Nov 2022? | Cites another paper on this list (verified) |
|---|---|---|---|---|
| 1 | Sagawa et al. 2020 | ICLR 2020 | no | no |
| 2 | Sohoni et al. 2020 | NeurIPS 2020 | no | **yes**: Sagawa |
| 3 | Koenecke et al. 2020 | PNAS (journal) | no | no |
| 4 | Dheram et al. 2022 | Interspeech 2022 (Sept) | no | **yes**: Koenecke |
| 5 | Veliche & Fung 2023 | ICASSP 2023 | **yes** | **yes**: Dheram, Koenecke |
| 6 | Radford et al. 2023 | ICML 2023 | **yes** | ⚠ probably Koenecke (seen in an extracted reference list, not the PDF) |
| 7 | Swain et al. 2026 | ICASSP 2026 | **yes** | **yes**: Sagawa, Radford, Dheram, Koenecke |

The requirements are met: 7 ≥ 5 papers; 4 ≥ 3 cite another paper on the list; 3 ≥ 3 recent. The recent count has **no margin**: Swain is so far only an arXiv preprint marked "Accepted in ICASSP 2026", and the arXiv version of Radford is from Dec 2022. An optional 8th paper for safety is van Zee, van Zee & Søgaard (2024), *Findings of NAACL 2024*, pp. 2213–2226. It found that larger Whisper models had *larger* worst-case group gaps. Its reference list hasn't been checked.

**Citation graph:** Koenecke ← Dheram ← Veliche & Fung; Sagawa ← Sohoni; Sagawa, Radford, Dheram, Koenecke ← Swain.

---

## 1. Sagawa, Koh, Hashimoto & Liang (2020): Group DRO for neural networks
**APA:** Sagawa, S., Koh, P. W., Hashimoto, T. B., & Liang, P. (2020). Distributionally robust neural networks for group shifts: On the importance of regularization for worst-case generalization. *International Conference on Learning Representations (ICLR 2020)*. https://openreview.net/forum?id=ryxGuJrFvS
(PDF header confirms "Published as a conference paper at ICLR 2020".)

- **Problem:** models with high *average* accuracy can fail on rare groups when they learn a spurious shortcut. Waterbirds example: 95% of waterbirds in training are on water backgrounds, so the model learns "water background → waterbird."
- **Group DRO did NOT originate here.** The paper adopts the group DRO setting of Hu et al. (2018) and Oren et al. (2019) (p. 3). Its contributions are:
  - showing that naive Group DRO fails on large ("overparameterized") networks, and that **strong regularization** fixes it
  - a stable online training algorithm (Algorithm 1)
  - the "group adjustment" term C/√n_g
- **Groups must be labeled in advance:** each training example comes as (x, y, g). The groups are (label × spurious attribute), e.g. 4 groups for Waterbirds.
- **Why naive Group DRO ≈ ERM:**
  - A large network memorizes the training set, so every group's *training* loss goes to about 0.
  - With nothing to upweight, Group DRO trains like ERM.
  - The gap appears only on unseen test data.
  - Strong L2 or early stopping keeps the model from memorizing, so the rare groups' training loss stays visible.
- **Waterbirds, Table 1 (test accuracy, average / worst-group):**

  | Setting | ERM | Group DRO |
  |---|---|---|
  | Standard regularization | 97.3 / 60.0 | 97.4 / 76.9 |
  | Strong L2 (λ = 1.0) | 95.7 / 21.3 | 96.6 / 84.6 |
  | Early stopping | 93.8 / 6.7 | 93.2 / 86.0 |

  - CelebA, standard regularization: worst group 41.1 for both methods.
  - MultiNLI, early stopping: worst group 66.0 → 77.7.
  - So naive Group DRO is *similar* to ERM, not identical: on Waterbirds it still helped (60.0 → 76.9).
- **Which group is worst?**
  - The paper **never names the ERM worst group** for Waterbirds. The 60.0% is labeled only "worst-group accuracy."
  - Waterbirds on land is the **smallest** group (56 of 4,795 training images; p. 3). The two rare groups are waterbirds on land and landbirds on water (Appendix C.1).
  - Figure 3 names waterbirds on land as dragging down worst-group accuracy, but **for the regularized Group DRO model**, not ERM.
- **Algorithm 1 (Section 5):** each step:
  1. sample a group uniformly at random
  2. sample an example from that group
  3. multiply that group's weight by exp(η·loss) and renormalize
  4. take an SGD step weighted by q_g

  In practice they use minibatches. It shifts weight *gradually* toward high-loss groups instead of picking the single worst group (which was unstable in Oren et al.).
- **BERT / MultiNLI:** strong L2 did *not* help (similar or worse); **early stopping did**. Relevant to us, since Whisper is also a transformer.
- **Model selection:** benchmark models were picked at the best early-stopping epoch by validation worst-group accuracy.
- **Hard parts:**
  - the convex-case convergence proof (Appendix A.2, online mirror descent)
  - the generalization-gap argument
  - how C was tuned
- **Cross-refs:** cites Hashimoto et al. 2018 (not on our list). **Cited by** Sohoni (as [48]) and Swain (as [18]).
- **Bearing on project:** Group DRO is the training method for conditions 2–4. Whisper-base (74M params) on about 10k clips is overparameterized, so without early stopping or strong weight decay our Group DRO runs could all collapse to ERM.

## 2. Sohoni, Dunnmon, Angus, Gu & Ré (2020): GEORGE
**APA:** Sohoni, N., Dunnmon, J., Angus, G., Gu, A., & Ré, C. (2020). No subclass left behind: Fine-grained robustness in coarse-grained classification problems. *Advances in Neural Information Processing Systems 33 (NeurIPS 2020)*, 19339–19352. ⚠ (check the page numbers on papers.nips.cc)

- **Problem: "hidden stratification."** Labeled classes ("superclasses") contain unlabeled subclasses. Example: 48% of benign ISIC skin-lesion images contain a colored patch, which makes them easy. Group DRO needs subclass labels, which are usually unavailable.
- **GEORGE**, a name the paper never expands (Figure 4, Section 4):
  1. Train an ERM model on the superclass labels.
  2. Take its penultimate-layer activations for every training example.
  3. Reduce them with UMAP.
  4. Cluster *within each superclass*.
  5. Train a **new** model with Sagawa's Group DRO, using the **cluster IDs as groups**.

  Validation points are assigned to the training clusters, and the checkpoint is chosen by worst-*cluster* validation accuracy.
- **Clustering detail (p. 6):** plain k-means and Gaussian mixture models "often failed to capture smaller clusters," and small clusters tend to be the rare, poorly served subclasses. Their fix: over-cluster, then use the silhouette score to choose k and filter out spurious clusters.
- **Results, Table 1 (worst-subclass / overall accuracy):**

  | Dataset | ERM | GEORGE | Group DRO + true labels |
  |---|---|---|---|
  | Waterbirds | 63.3 / 97.3 | 76.2 / 95.7 | 90.7 / 92.7 |
  | U-MNIST | 93.9 / 98.7 | 95.7 / 98.1 | 96.8 / 98.0 |
  | CelebA | 40.3 / 95.7 | 53.7 / 94.6 | 89.3 / 92.8 |
  | ISIC (AUROC) | no meaningful difference | | |

  - On average GEORGE cuts worst-case subclass **error** by 22%, versus 51% for the true-label oracle.
  - "Up to 22 percentage points" refers to *accuracy* (CelebA). These are two different measurements.
  - With pretrained BiT features, CelebA nearly matches the oracle: **feature quality matters**.
  - Clusters are imperfect (high recall, much lower precision) but still help. Clustering also surfaced an unlabeled ISIC "needed a biopsy" subgroup.
- **Datasets:** reuses Sagawa's **Waterbirds** and **CelebA** ("introduced to evaluate GDRO in [48]"), plus U-MNIST and ISIC.
- **Hard parts:**
  - Section 5 theory (Gaussian-mixture assumptions, Theorem 1 sample complexity)
  - "soft" group labels (Appendix D)
- **Cross-refs:** cites Sagawa [48] for the Group DRO algorithm and the oracle baseline.
- **Bearing on project:** this is the idea we **adapt** (we don't use GEORGE as-is). Our conditions 1 / 3 / 2 mirror its ERM / GEORGE / oracle table, and the headline question is the same: how much of the gap do the clusters close?

### GEORGE vs. our project
| | GEORGE | Ours |
|---|---|---|
| Task | image classification (one label) | speech-to-text (sentence output) |
| Hidden groups | subclasses inside a label | accents (not tied to the output) |
| Features | the ERM model's own penultimate layer | a separate pretrained speaker-ID model (ECAPA-TDNN) |
| Clustered | images, within each class | all clips together (no classes) |
| Method | UMAP + over-clustering + silhouette k | k-means, k = 6 (plus a sweep; optional UMAP/silhouette) |
| Group DRO model | a new image model | Whisper-base, fine-tuned |
| Loss seen by Group DRO | one classification loss | per-token cross-entropy (noisier) |
| Comparisons | ERM, GEORGE, oracle | the same + a **random-groups** control |
| Data | thousands to hundreds of thousands of images | 24 speakers |

## 3. Koenecke et al. (2020): racial disparities in ASR
**APA:** Koenecke, A., Nam, A., Lake, E., Nudell, J., Quartey, M., Mengesha, Z., Toups, C., Rickford, J. R., Jurafsky, D., & Goel, S. (2020). Racial disparities in automated speech recognition. *Proceedings of the National Academy of Sciences, 117*(14), 7684–7689. https://doi.org/10.1073/pnas.1915768117
(Read via PMC7149386. Detailed Methods/SI were not reached.)

- **Question:** do 5 commercial ASR systems (Amazon, Apple, Google, IBM, Microsoft) transcribe Black and white speakers equally well?
- **Data:**
  - CORAAL: African American speakers in Princeville NC, Rochester NY, Washington DC.
  - VOC: white speakers in Sacramento and Humboldt County, CA.
  - Clips are matched on speaker **age, gender, and clip length** (nearest-neighbor matching), *not* on sentence content.
  - Matched set: 2,141 clips per group; 19.8 h; 73 Black and 42 white speakers.
- **Results:**
  - Average WER **0.35 (Black) vs. 0.19 (white)**, "nearly twice as large in every case."
    - Microsoft 0.27 / 0.15
    - Apple 0.45 / 0.23
  - Clips with WER > 0.5: **23% vs. 1.6%**.
  - By gender: Black men 0.41, Black women 0.30, white men 0.21, white women 0.17.
- **Cause:**
  - **Language model?** Mostly no. 98–99% of the words spoken were in vocabulary. The LMs did prefer Standard English grammar ("he a pastor" got perplexity 305 vs. 67 for "he's a pastor").
  - **Acoustic model?** Yes. On **206 identical phrases** spoken by both groups the gap remained (e.g. Apple 0.28 vs. 0.12).
  - Higher AAVE dialect density was associated with higher WER.
  - The authors conclude the gap arises "primarily" from the acoustic models (too little training audio from Black speakers).
- **Recommendations:** more diverse training data (including "nonnative accents"); regularly assess and publicly report these gaps.
- **Limitations:**
  - the groups came from different regions (East Coast vs. California)
  - different recording years and equipment
  - the commercial LMs weren't available, so public LMs were used as stand-ins
- **Not an accent paper.** It is about race and dialect (AAVE); it only mentions non-native accents in its recommendations.
- **Hard parts:**
  - perplexity
  - nearest-neighbor matching
  - separating the language-model and acoustic-model contributions
- **Cross-refs:** cites none of our papers (it is the earliest). **Cited by** Dheram, Veliche & Fung, Swain (and ⚠ probably Radford).
- **Bearing on project:** motivation: ASR gaps between speaker groups are large and real. Also, because the gap comes from how speech *sounds*, clustering voice embeddings is a sensible way to find the underserved groups. Its identical-phrase test parallels L2-ARCTIC, where everyone reads the same sentences.

## 4. Dheram et al. (2022): discovering and mitigating ASR disparities (Amazon)
**APA:** Dheram, P., Ramakrishnan, M., Raju, A., Chen, I.-F., King, B., Powell, K., Saboowala, M., Shetty, K., & Stolcke, A. (2022). Toward fairness in speech recognition: Discovery and mitigation of performance disparities. *Proc. Interspeech 2022*, 1268–1272. https://doi.org/10.21437/Interspeech.2022-10816

- **Data:** internal, de-identified Alexa US-English far-field recordings (not released).
- **Two separate studies:**
  1. **Discovery.** Can we find underserved users without demographic labels?
     - *Geographic:* ZIP codes where at least 75% of residents share one census category. The lowest-confidence ones form the "bottom cohort" (431 ZIPs, confidence 11.7% below average).
     - *Automatic:*
       - speaker embeddings from the wake word → **k-means (K = 50)**
       - k-means only groups similar voices
       - clusters are **then ranked by mean ASR confidence** (cheap, no transcripts needed), and the lowest 10% form the bottom cohort
       - WER was then confirmed with human transcription (3 blind passes plus adjudication)
     - **Result:** the relative WER gap between the bottom and top cohorts was **65.0% for clusters vs. 41.7% for ZIP codes**. Clustering voices found a *bigger* gap.
  2. **Mitigation: ZIP-code (geographic) cohorts only, never the clusters.**
     - **Oversampling:** the bottom cohort's share of training data went from 0.8% to 9.0%, using machine (teacher) transcripts. Relative gap 56.3% → 46.2%.
     - **Cohort embedding:** a one-hot top/bottom flag concatenated to the acoustic features. Gap → 38.5%, the best.
     - **Both together:** gap 40.0%, no extra gain ("counterintuitive").
     - Top-cohort WER got slightly worse (−1.6% WERR) in all runs.
     - The paper doesn't say how the cohort flag is set at test time. That's an open question, and it matters because it requires group info at inference.
     - The paper doesn't explain why the clusters weren't used for mitigation. Our guesses: the ZIP cohorts are interpretable, membership is easy to look up, and the two discovery methods used different data.
- **Numbers caveat:** the 41.7% / 65.0% discovery gaps come from a hybrid RNN-HMM model; the 56.3% mitigation baseline comes from an RNN-T. The models differ, so the numbers aren't directly comparable.
- **Hard parts:**
  - confidence as a stand-in for accuracy (systems can be confidently wrong)
  - the PDR transcriber-disagreement metric
  - RNN-T
- **Cross-refs:** cites Koenecke [8]. **Cited by** Veliche & Fung [2] and Swain [5].
- **Bearing on project:** evidence that clustering speaker embeddings surfaces groups the system serves worse, which is the assumption our clustering step relies on. It never trained with the clusters (and never used Group DRO). Veliche & Fung then fed clusters in as an input; we use them as Group DRO groups.

## 5. Veliche & Fung (2023): unsupervised clustering for fair ASR (Meta)
**APA:** Veliche, I.-E., & Fung, P. (2023). Improving fairness and robustness in end-to-end speech recognition through unsupervised clustering. *IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP 2023)*. ⚠ (add the DOI/pages from IEEE Xplore)

- **Constraint:** privacy. No demographics, ZIP codes, or raw speaker embeddings may be used in training.
- **Method:**
  - 192-dim speaker-ID embeddings per 10 s chunk → PCA → k-means (K = 50, elbow method)
  - the cluster ID is fed into the RNN-T **encoder** as an extra one-hot input
  - 10% of the time it is replaced with "unknown"
  - at **test time every utterance gets "unknown"**, so no group info is needed at inference
- **Results:**
  - WER improved for every demographic group on an in-house voice-command set, e.g. non-native 8.52 → 8.44, East Asian 6.39 → 5.89, female 6.42 → 6.02.
  - On Casual Conversations, male speakers improved 17.54 → 15.58.
  - About 80M-parameter model trained from scratch on about 22k hours.
- **Caveats:**
  - they don't know what the clusters capture (by design)
  - non-native speakers gained least (0.93% relative)
  - no Group DRO
- **Hard parts:** why an input that is always "unknown" at test time still helps; Emformer/RNN-T.
- **Cross-refs:** cites Dheram 2022 and Koenecke 2020.
- **Bearing on project:** the most direct evidence that voice-embedding clusters help ASR fairness. We test the *other* way to use them: as Group DRO groups, following Sohoni.

## 6. Radford et al. (2023): Whisper
**APA:** Radford, A., Kim, J. W., Xu, T., Brockman, G., McLeavey, C., & Sutskever, I. (2023). Robust speech recognition via large-scale weak supervision. *Proceedings of the 40th International Conference on Machine Learning (ICML 2023)*, PMLR 202, 28492–28518.

- **What it is:** an encoder–decoder Transformer trained on about 680k hours of weakly labeled audio. Input: a 30 s, 80-channel log-mel spectrogram. The decoder predicts text tokens after special tokens (language, transcribe/translate, timestamps).
- **Sizes:** tiny 39M, **base 74M** (ours), small 244M, medium 769M, large 1550M.
- **Key result:** zero-shot Whisper-large matches wav2vec 2.0 on LibriSpeech-clean (2.7 WER) but makes 55% fewer errors averaged across other datasets (e.g. Common Voice 29.9 → 9.0).
- No dedicated accent-fairness analysis in the main text.
- **Cross-refs:** ⚠ cites Koenecke 2020 (seen in an extracted reference list, not confirmed in the PDF). **Cited by** Swain.
- **Bearing on project:** the model we fine-tune. Its loss is token-level cross-entropy over 30 s-padded audio, which is why the CTC length issue that CTC-DRO fixes doesn't apply to us.

## 7. Swain et al. (2026): fairness-prompted fine-tuning for L2 speakers
**APA:** Swain, M., Maji, B., Mishra, J., Schedl, M., Søgaard, A., & Jensen, J. R. (2026). Towards fair ASR for second language speakers using fairness prompted finetuning. In *Proceedings of the 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP 2026)*. (arXiv:2510.18374 says "Accepted in ICASSP 2026"; ⚠ add the DOI/pages once on Xplore.)

- **Setup:** Whisper (tiny–large) and SeamlessM4T fine-tuned on **EdAcc** (26 accent groups) *with true group labels*. Compares ERM, Spectral Decoupling, Group DRO, IRM, and a weighted "Fusion" of all of them.
- **Results (macro-avg WER / max–min gap):**
  - Whisper-large: zero-shot 58.3 / 114.0 → ERM 26.7 / 30.1 → **DRO 33.3 / 37.6 (worse than ERM)** → Fusion 24.1 / 30.8
  - On Seamless, DRO *did* beat ERM (e.g. Seamless-medium 40.5 → 26.8)
- **Caveats:**
  - Fusion doesn't always shrink the gap.
  - No seeds or significance tests are reported.
  - Urdu and Indian English didn't improve.
- **Cross-refs:** cites Sagawa [18], Radford [14], Dheram [5], Koenecke [6].
- **Bearing on project:** the closest prior work, and the reference point for our "true labels" condition. Its finding that Group DRO hurt Whisper is a warning: regularize carefully (per Sagawa), and a negative result for condition 2 is plausible.
