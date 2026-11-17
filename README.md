# Fair Speech Recognition Without Accent Labels

CS 445: Machine Learning poster project, James Madison University, Fall 2026
**Team:** [Your Name], [Partner Name]

Speech-to-text systems make more mistakes for some speakers than others, including non-native English speakers. **Group DRO** (Sagawa et al., 2020) trains a model to focus on whichever group it currently serves worst, but it needs every speaker's group label, and accent labels are often unavailable for privacy reasons. In computer vision, Sohoni et al. (2020) replaced true labels with *automatically discovered* clusters.

**Our question:** can Group DRO make OpenAI's Whisper fairer across accents using k-means clusters of speaker embeddings instead of accent labels, and how close does that get to using the true labels?

We fine-tune Whisper-base on [L2-ARCTIC](https://psi.engr.tamu.edu/l2-arctic-corpus/) (24 speakers, 6 first languages) four ways:

| # | Training | Role |
|---|---|---|
| 1 | Standard fine-tuning (ERM) | baseline |
| 2 | Group DRO + true first-language labels | upper reference |
| 3 | Group DRO + discovered speaker clusters | **main experiment** |
| 4 | Group DRO + random groups | sanity check |

Every model is evaluated the same way: word error rate per *true* first language, on held-out speakers and held-out sentences.

See [docs/project_plan.md](docs/project_plan.md) for the full plan and [docs/proposal.md](docs/proposal.md) for the proposal.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Data

**L2-ARCTIC (main dataset).** Free under CC BY-NC 4.0 but requires a request form: fill it in at https://psi.engr.tamu.edu/l2-arctic-corpus/ and a download link is emailed to you. Unzip it so the folder looks like:

```
data/l2arctic/ABA/wav/arctic_a0001.wav
data/l2arctic/ABA/transcript/arctic_a0001.txt
...
```

**EdAcc (backup dataset).** Streamed automatically from Hugging Face (`edinburghcstr/edacc`); no download step is needed.

## Running

Print summary statistics (speakers, first languages, utterances, hours, shared sentences):

```bash
python src/data_stats.py --source l2arctic --data-dir data/l2arctic
python src/data_stats.py --source edacc --max-rows 2000   # works without the L2-ARCTIC download
```

Later stages (embedding extraction, clustering, fine-tuning and evaluation) will be added under `src/` as the project progresses.

## Repository layout

```
src/data_stats.py      dataset loading + summary statistics
docs/                  proposal, project plan, bibliography notes
requirements.txt
```

## How we used AI tools

We used Claude Code (Anthropic) as a coding and research assistant. It helped draft the data-loading script, the README and an early version of the project plan, and it helped us look up and cross-check paper references. We reviewed and edited everything it produced, ran the code ourselves, and made all decisions about the experimental design. Per course policy, the annotated-bibliography summaries are written entirely by us. *(Update this section as the project goes on.)*

## References

- Sagawa, S., Koh, P. W., Hashimoto, T. B., & Liang, P. (2020). Distributionally robust neural networks for group shifts. *ICLR 2020*.
- Sohoni, N., Dunnmon, J., Angus, G., Gu, A., & Ré, C. (2020). No subclass left behind: Fine-grained robustness in coarse-grained classification problems. *NeurIPS 2020*.
- Swain, M., et al. (2026). Towards fair ASR for second language speakers using fairness prompted finetuning. *ICASSP 2026*.
- Veliche, I.-E., & Fung, P. (2023). Improving fairness and robustness in end-to-end speech recognition through unsupervised clustering. *ICASSP 2023*.
- Radford, A., et al. (2023). Robust speech recognition via large-scale weak supervision. *ICML 2023*.
- Zhao, G., et al. (2018). L2-ARCTIC: A non-native English speech corpus. *Interspeech 2018*.
