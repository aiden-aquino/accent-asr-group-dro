# Project Plan: Fair Speech Recognition Without Accent Labels

**[Your Name] and [Partner Name]**, CS 445: Machine Learning, Fall 2026

**Goal.** Can Group DRO make Whisper fairer across accents using automatically discovered speaker groups instead of accent labels, and how close does it come to using the true labels? We will fine-tune Whisper-base four ways: (1) standard fine-tuning (ERM), (2) Group DRO with true first-language (L1) labels, (3) Group DRO with k-means clusters of speaker embeddings, and (4) Group DRO with random groups. We expect (3) to land between (1) and (2). This is not guaranteed: in Swain et al. (2026), Group DRO with true labels did *worse* than ERM for Whisper on EdAcc. In that case our first result is whether (2) beats (1) at all.

**Connection to the literature.** We transfer the "cluster, then Group DRO" recipe of Sohoni et al. (2020) from image classification to accented speech. Group DRO and its need for strong regularization come from Sagawa et al. (2020). Swain et al. (2026) is the closest prior work: true-label Group DRO on Whisper for L2 speakers. Veliche & Fung (2023) and Dheram et al. (2022) show that k-means clusters of speaker embeddings carry fairness-relevant information in ASR, but they used the clusters as input features or for oversampling, not for Group DRO. Koenecke et al. (2020) motivate the problem, and Radford et al. (2023) describe the model.

**Data.** L2-ARCTIC v5.0 (Zhao et al., 2018): 24 non-native English speakers, 4 per L1 (Arabic, Mandarin, Hindi, Korean, Spanish, Vietnamese; 2 male and 2 female each), about 1 hour of read CMU ARCTIC prompts per speaker. **Split by speaker and by sentence:** per L1, 2 speakers for training (1 male, 1 female), 1 for validation and 1 for testing, alternating sex across L1s so that the test set is 3 male and 3 female. Prompts are split 80/10/10 by sentence ID, so no test sentence is ever heard in training. Backup dataset: EdAcc (conversational, 40 h, L1 labels), which Swain et al. used.

**Methods.** Group discovery: ECAPA-TDNN speaker embeddings (SpeechBrain), computed per utterance, reduced with PCA and clustered with k-means at k = 6, plus a sweep over k ∈ {3, 6, 12}. Training: Hugging Face Whisper-base (74M parameters), same hyperparameters for all four conditions. Group DRO is the online algorithm of Sagawa et al. (exponentiated-gradient group weights) with strong weight decay and early stopping on validation worst-group WER. **Baselines:** ERM (condition 1) and zero-shot Whisper-base.

**Evaluation.** WER (jiwer) on the held-out test speakers and sentences, reported per *true* L1: average WER, worst-group WER, and the gap between the best and worst L1. Mean ± std over 3 seeds. Cluster analysis: adjusted Rand index and NMI of clusters against L1 and against sex, to explain *why* condition 3 does or doesn't work. Sanity checks: zero-shot WER should be nonzero but reasonable, ERM should beat zero-shot, and we will listen to a sample of the worst transcripts. Answering the question means comparing (3) against (2) and (4); "closes X% of the ERM→true-label gap" is the headline number.

**Compute.** Embeddings and clustering take minutes on a laptop CPU. Each fine-tuning run should take under 1 hour on a free Colab T4: 4 conditions × 3 seeds = 12 runs, about 10–12 GPU-hours split between two accounts.

**Timeline.**
- *Checkpoint (Dec 3):* data pipeline and splits finished; zero-shot and ERM results; clusters computed with ARI/NMI analysis; first Group DRO (true labels) run.
- *Poster draft (Dec 10):* all 4 conditions × 3 seeds; per-L1 WER figure and gap table; cluster-vs-L1/sex figure.
- *Poster session (Dec 17):* final figures, error examples, write-up.

**Fallback.** If clustering is uninformative (it captures sex rather than accent), we report that as a finding and the project reduces to conditions 1 vs. 2: "Does Group DRO improve accent fairness for small Whisper on L2-ARCTIC?" If L2-ARCTIC access is delayed, we switch to EdAcc. If Group DRO is unstable, we fall back to the simpler reweighting baseline (upweighting groups by inverse frequency or by loss).
