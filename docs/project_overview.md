# Poster Project Overview: Fair Speech Recognition Without Accent Labels

CS 445 Machine Learning, JMU, Fall 2026
Partners: _[your name]_, _[partner name]_
Proposal due: 2026-10-29

---

## 1. The question in one sentence

> **Can Group DRO reduce accent disparities in Whisper using automatically discovered speaker clusters instead of accent labels? How close does it come to using the true labels?**

Project type (from the assignment): **Application**, which the spec describes as "apply a method to a new dataset or problem domain. How well does it transfer?" The method is cluster-then-Group-DRO (GEORGE, from computer vision). The new domain is accent fairness in speech recognition.

## 2. Why it matters

Speech-to-text runs underneath a lot of things that affect people:
- live captions in lectures and Zoom (deaf and hard-of-hearing students depend on them)
- doctors' dictation and AI medical scribes (one misheard word ends up in a patient's chart)
- automated job-interview screening, which transcribes answers and then scores them
- court, police and immigration transcripts
- phone menus and voice assistants

These systems make more mistakes for some groups. Koenecke et al. (2020) found that commercial systems made nearly twice as many errors for Black speakers as for white speakers. Most English speakers worldwide are non-native, so accent gaps affect a lot of people.

Methods that fix this, like Group DRO, need to know each person's group. **Companies often can't collect accent or background labels because of privacy**, so we test whether automatically discovered groups work as a substitute.

Framing for the poster: *"Universities use automatic captions in lectures. Do they work equally well for international students and faculty, and can we make them fairer without asking anyone where they're from?"*

## 3. Key terms (plain language)

**Whisper.** OpenAI's open-source speech-to-text model (Radford et al., 2023), trained on about 680,000 hours of internet audio. We use the small versions (tiny, base or small) so it fits on a free Colab GPU. Whisper is the model we experiment *on*, not the method we're testing.

**WER (word error rate).** The percent of words the model gets wrong. Lower is better.
- WER = (substitutions + deletions + insertions) ÷ number of words actually said
- Example: if someone said "the patient is allergic to penicillin" and the model wrote "the patient is a lurgic to penicillin", that's 2 errors in 6 words, so WER ≈ 33%.
- Limitation: every error counts the same, even though "penicillin" matters much more than "the".
- The Python library `jiwer` computes it.

**Standard fine-tuning (ERM).** Normal training. Each step, average the loss over all clips, with every clip counting equally, then backpropagate. The model never knows who is speaking. This is our baseline.

**Group DRO** (Sagawa et al., 2020). A fairness-focused version of training:
1. Compute the loss separately for each group in the batch.
2. Increase the weight of whichever group currently has the highest loss.
3. Train on the weighted sum of group losses.

The model ends up putting extra effort into whichever group it currently serves least accurately. It's only about 10–15 extra lines of code in the training loop. The original paper found it **needs strong regularization** (weight decay or early stopping). Otherwise the model memorizes the training set, every group's loss drops to about 0, and Group DRO behaves like normal training.

*Wording note:* we say "the group the model serves least accurately" or "the least-served group," never "the worst accent." Any failure belongs to the model, not to the accent.

**Embeddings and clustering.** A pretrained speaker-recognition model (for example ECAPA-TDNN from the SpeechBrain library) turns each voice into a vector, a kind of "voice fingerprint." Then k-means (from class) groups similar fingerprints together, with no labels needed.

## 4. The experiment

### Step 1: Discover groups
- Extract a voice embedding for each speaker. Main option: ECAPA-TDNN, the speaker-ID approach used by Meta. Stretch option: Whisper's own encoder features, the approach GEORGE uses.
- Run k-means with k = 6 (matching the 6 first-language groups), plus a few other values of k.

### Step 2: Check what the clusters actually found
Compare the clusters with the true first-language labels. Did they capture **accent**, or **gender, pitch or recording quality**? Speaker-ID models are built to tell individuals apart, so a gender split is a real possibility. L2-ARCTIC has 2 men and 2 women per accent group. Either answer is a finding, and this analysis explains the results in Step 3.

### Step 3: Fine-tune Whisper 4 ways

| # | Fine-tuned with | Role |
|---|---|---|
| 1 | Standard fine-tuning, no groups | Baseline |
| 2 | Group DRO + **true accent labels** | Best case ("what if we knew everyone's accent?") |
| 3 | Group DRO + **discovered clusters** | **Main experiment** |
| 4 | Group DRO + **random groups** | Sanity check: does the grouping matter at all? |

The groupings are used **only during training**. Every model is **tested the same way**: WER on held-out speakers, broken down by their *true* accent.

### Step 4: Compare

| Comparison | Answers |
|---|---|
| 2 vs. 1 | Does Group DRO improve accent fairness *at all* in our setup? |
| **3 vs. 2** | **How much do we lose by not knowing people's accents?** (main question) |
| 3 vs. 4 | Do the clusters carry real information, or is any gain just luck? |
| Step 2 analysis | *Why* the clusters did or didn't work |

**Metrics:** average WER, and the **gap** between the best- and least-served accent groups. Run several random seeds.

**How to read it:**
- Clusters ≈ true labels: automatic grouping is a practical, privacy-friendly substitute.
- Clusters between random and true labels: partial success. Measure how much of the gap they close.
- Clusters ≈ random or baseline: the clusters didn't capture what matters. Step 2 explains why.

## 5. Data

**L2-ARCTIC** (Zhao et al., 2018) contains:
- 24 non-native English speakers, 4 for each first language: Hindi, Mandarin, Korean, Spanish, Arabic and Vietnamese
- each speaker reading about an hour of the same CMU ARCTIC sentences
- **Access:** free, but requires a request form on the corpus page. **Submit it right away.**
- Because everyone reads the same sentences, sentence content and length can't explain differences between groups.

**Splits (important):** hold out **speakers** *and* **sentences**. Everyone reads the same prompts, so if a test sentence appears in training, the model can memorize its text and the results look better than they really are.

**Backup / extra speakers:** EdAcc (Edinburgh International Accents of English Corpus), free on Hugging Face. Mozilla Common Voice also has accent tags.

**Limitation to state on the poster:** 24 speakers is a small sample for clustering.

## 6. What's been done before (and where we fit)

| Paper | Clusters without labels? | Group DRO? | Speech-to-text for accents? | Whisper? |
|---|---|---|---|---|
| Sohoni et al., GEORGE (NeurIPS 2020) | ✅ | ✅ | ❌ images | ❌ |
| Veliche & Fung (ICASSP 2023, Meta) | ✅ | ❌ uses clusters as input features | ✅ | ❌ |
| Dheram et al. (Interspeech 2022, Amazon) | ✅ | ❌ oversampling | ✅ | ❌ |
| Kim et al. (2024, arXiv only) | ✅ | ✅ | ❌ accent *recognition*, not transcription | ❌ |
| Swain et al. (ICASSP 2026) | ❌ true labels | ✅ | ✅ | ✅ |
| **Us** | ✅ | ✅ | ✅ | ✅ |

We didn't find any paper combining all four (searched 2026-10-28), but we can't rule it out. We frame this as testing whether an existing method transfers to a new domain at small scale, not as a new research claim.

**Why speech might behave differently from images (why the answer isn't obvious):**
- Voice embeddings are built to tell *individuals* apart, so they may not capture accent.
- Accent is a spectrum, not a clean category.
- Speech-recognition losses are per word, so they're noisier than one label per image.

## 7. Compute and risks

**Compute:** embeddings and clustering take minutes. Each Whisper fine-tuning run (base or small) should take under an hour on a Colab T4. 4 conditions × 3 seeds is manageable.

**Risks and fallback:**
- Clusters may capture gender instead of accent. Still a reportable finding, explained in Step 2.
- Group DRO may not help even with true labels (condition 2 ≈ 1). Still an honest result; check regularization first.
- **Fallback:** if clustering goes badly, conditions 1 and 2 alone answer "does Group DRO improve accent fairness for small Whisper on L2-ARCTIC?", which is a complete project.
- L2-ARCTIC access is delayed: switch to EdAcc.

## 8. References (APA)

✅ = details checked online on 2026-10-28. Still click through each one yourself before submitting.

1. ✅ Sohoni, N., Dunnmon, J., Angus, G., Gu, A., & Ré, C. (2020). No subclass left behind: Fine-grained robustness in coarse-grained classification problems. *Advances in Neural Information Processing Systems 33 (NeurIPS 2020)*.
2. ✅ Sagawa, S., Koh, P. W., Hashimoto, T. B., & Liang, P. (2020). Distributionally robust neural networks for group shifts: On the importance of regularization for worst-case generalization. *International Conference on Learning Representations (ICLR 2020)*.
3. ✅ Swain, M., Maji, B., Mishra, J., Schedl, M., Søgaard, A., & Jensen, J. R. (2026). Towards fair ASR for second language speakers using fairness prompted finetuning. In *Proceedings of the 51st IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP 2026)*. *(Not yet on IEEE Xplore as of 2026-10-28; add pages/DOI if it appears.)*
4. ✅ Veliche, I.-E., & Fung, P. (2023). Improving fairness and robustness in end-to-end speech recognition through unsupervised clustering. *IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP 2023)*.
5. ✅ Dheram, P., Ramakrishnan, M., Raju, A., Chen, I.-F., King, B., Powell, K., Saboowala, M., Shetty, K., & Stolcke, A. (2022). Toward fairness in speech recognition: Discovery and mitigation of performance disparities. *Proc. Interspeech 2022*, 1268–1272.
6. ✅ Zhao, G., Sonsaat, S., Silpachai, A., Lucic, I., Chukharev-Hudilainen, E., Levis, J., & Gutierrez-Osuna, R. (2018). L2-ARCTIC: A non-native English speech corpus. *Proc. Interspeech 2018*, 2783–2787. https://doi.org/10.21437/Interspeech.2018-1110
7. ✅ Radford, A., Kim, J. W., Xu, T., Brockman, G., McLeavey, C., & Sutskever, I. (2023). Robust speech recognition via large-scale weak supervision. *Proceedings of the 40th International Conference on Machine Learning (ICML 2023)*, PMLR 202, 28492–28518.
8. ✅ Koenecke, A., Nam, A., Lake, E., Nudell, J., Quartey, M., Mengesha, Z., Toups, C., Rickford, J. R., Jurafsky, D., & Goel, S. (2020). Racial disparities in automated speech recognition. *Proceedings of the National Academy of Sciences, 117*(14), 7684–7689. https://doi.org/10.1073/pnas.1915768117
9. ✅ Hashimoto, T., Srivastava, M., Namkoong, H., & Liang, P. (2018). Fairness without demographics in repeated loss minimization. *Proceedings of the 35th International Conference on Machine Learning (ICML 2018)*, PMLR 80, 1929–1938.
10. ✅ van Zee, A. K., van Zee, M., & Søgaard, A. (2024). Group fairness in multilingual speech recognition models. *Findings of the Association for Computational Linguistics: NAACL 2024*, 2213–2226. https://doi.org/10.18653/v1/2024.findings-naacl.143

**For the proposal (≥3 refs):** 1 (GEORGE), 2 (Group DRO), 3 (Swain), 4 (Veliche & Fung), plus 6 (dataset).

**Background reading only:** Kim et al. (2024), "Clustering and mining accented speech for inclusive and fair speech recognition" (arXiv:2408.02582). It isn't peer-reviewed, so it doesn't count for the bibliography.

## 9. Deadline checklist

- [ ] **Deadline 1 (proposal):** 1–2 paragraphs (question + data), ≥3 full references (authors, year, venue), both names, submitted as a PDF
- [ ] Submit the L2-ARCTIC access request
- [ ] **Deadline 2:** annotated bibliography (≥5 peer-reviewed papers, ≥3 from 2022 or later, ≥3 that cite another paper in the list; **summaries written by us, not by AI**, which would be an honor-code violation), a one-page project plan, and a GitHub repo with a README and code that loads the data
- [ ] Deadline 3: progress report with preliminary figures
- [ ] Deadline 4: poster draft
- [ ] Deadline 5: poster session (36"×24" landscape, with a QR code to the repo)

## 10. Ideas we considered and dropped

- **Reproducing Swain et al. at smaller scale:** now folded in as condition 2, rather than being a separate question.
- **Testing CTC-DRO's fixes** (Bartelds et al., arXiv:2502.01777, reportedly ICLR 2026): CTC-DRO fixes a problem where loss grows with audio length in CTC-based speech models. Whisper pads all audio to 30 seconds and computes its loss on text tokens, and in L2-ARCTIC everyone reads the same sentences. So the problem it fixes probably doesn't exist in our setup, and testing the fix wouldn't make sense.
- **Fairness on held-out accents** (train on 5 accent groups, test on the 6th): a good alternative if the clustering direction stalls.
- **Model size vs. fairness gap** (van Zee et al., 2024: bigger models had bigger worst-case gaps): an easy extra analysis, since it only needs evaluation, no training.
