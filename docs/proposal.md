# Fair Speech Recognition Without Accent Labels

**[Your Name] and [Partner Name]**
CS 445: Machine Learning, Fall 2026

Speech-to-text systems make more mistakes for some speakers than others, including people with non-native accents. One training method that helps is Group DRO (Sagawa et al., 2020). Instead of treating all training examples equally, it focuses extra effort on whichever group the model is currently handling worst. Swain et al. (2026) used Group DRO to make OpenAI's Whisper model fairer across accent groups. However, their method requires knowing every speaker's accent, which real companies often can't collect for privacy reasons. In image classification, Sohoni et al. (2020) got around this by clustering the data to discover groups automatically and then using those clusters in Group DRO. In speech, Veliche and Fung (2023) also grouped speakers automatically by their voices, but they gave each speaker's group to the model as an extra input rather than using it in Group DRO. This still improved fairness, especially for accented speakers, which suggests that automatically discovered voice groups carry useful information about accent. We want to find out whether Sohoni et al.'s approach of combining discovered groups with Group DRO also works for accents in speech.

**Our question:** can Group DRO make Whisper fairer across accents using automatically discovered speaker groups instead of accent labels, and how close does it get to using the real labels? We'll use L2-ARCTIC (Zhao et al., 2018), a free dataset of 24 non-native English speakers whose first languages are Hindi, Mandarin, Korean, Spanish, Arabic and Vietnamese. All of them read the same sentences. We'll group the speakers with k-means on voice embeddings from a pretrained speaker-recognition model. Then we'll fine-tune a small Whisper model four ways: normal fine-tuning, Group DRO with the real accent labels, Group DRO with our discovered groups, and Group DRO with random groups as a sanity check. We'll measure word error rate for each accent group on speakers the model hasn't seen, and compare how big the gap between groups is under each approach.

## References

Sagawa, S., Koh, P. W., Hashimoto, T. B., & Liang, P. (2020). Distributionally robust neural networks for group shifts: On the importance of regularization for worst-case generalization. *International Conference on Learning Representations (ICLR 2020)*. https://arxiv.org/abs/1911.08731

Sohoni, N., Dunnmon, J., Angus, G., Gu, A., & Ré, C. (2020). No subclass left behind: Fine-grained robustness in coarse-grained classification problems. *Advances in Neural Information Processing Systems 33 (NeurIPS 2020)*. https://papers.nips.cc/paper_files/paper/2020/file/e0688d13958a19e087e123148555e4b4-Paper.pdf

Swain, M., Maji, B., Mishra, J., Schedl, M., Søgaard, A., & Jensen, J. R. (2026). Towards fair ASR for second language speakers using fairness prompted finetuning. In *Proceedings of the 51st IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP 2026)*. https://arxiv.org/abs/2510.18374

Veliche, I.-E., & Fung, P. (2023). Improving fairness and robustness in end-to-end speech recognition through unsupervised clustering. In *IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP 2023)*. https://arxiv.org/abs/2306.06083

Zhao, G., Sonsaat, S., Silpachai, A., Lucic, I., Chukharev-Hudilainen, E., Levis, J., & Gutierrez-Osuna, R. (2018). L2-ARCTIC: A non-native English speech corpus. In *Proc. Interspeech 2018* (pp. 2783–2787). https://doi.org/10.21437/Interspeech.2018-1110
