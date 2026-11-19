# Fair Speech Recognition Without Accent Labels

CS 445: Machine Learning poster project, James Madison University, Fall 2026
**Team:** Aiden Aquino, Josh Fuentes

Speech-to-text systems make more mistakes for some speakers than others, including non-native English speakers. **Group DRO** (Sagawa et al., 2020) trains a model to focus on whichever group it currently serves worst, but it needs every speaker's group label, and accent labels are often unavailable for privacy reasons. In computer vision, Sohoni et al. (2020) replaced true labels with *automatically discovered* clusters.

**Our question:** can Group DRO make OpenAI's Whisper fairer across accents using k-means clusters of speaker embeddings instead of accent labels, and how close does that get to using the true labels?

We fine-tune Whisper-base on [EdAcc](https://huggingface.co/datasets/edinburghcstr/edacc) (122 speakers, 40+ first languages, conversational English) four ways:

| # | Training | Role |
|---|---|---|
| 1 | Standard fine-tuning (ERM) | baseline |
| 2 | Group DRO + true first-language labels | upper reference |
| 3 | Group DRO + discovered speaker clusters | **main experiment** |
| 4 | Group DRO + random groups | sanity check |

Every model is evaluated the same way: word error rate per *true* accent group, on held-out speakers.

Project documents:
- [docs/project_plan.md](docs/project_plan.md): the full project plan
- [docs/proposal.md](docs/proposal.md): the topic proposal
- [docs/project_context.md](docs/project_context.md): deadlines, decisions, status, and key takeaways
- [docs/annote.bib](docs/annote.bib): bibliography citations (our annotations are written in Overleaf)
- [docs/bibliography_notes.md](docs/bibliography_notes.md): AI-generated study notes on the papers, used for understanding only, not our annotations

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Data

**EdAcc (main dataset).** On Hugging Face (`edinburghcstr/edacc`), about 7 GB. `data_stats.py` streams it, so nothing is stored locally. For training, download it on the GPU machine (non-streaming `load_dataset`); set `HF_HOME` to put the cache on another drive.

**L2-ARCTIC (backup dataset).** Free under CC BY-NC 4.0 but requires a request form: fill it in at https://psi.engr.tamu.edu/l2-arctic-corpus/ and a download link is emailed to you. Unzip it so the folder looks like:

```
data/l2arctic/ABA/wav/arctic_a0001.wav
data/l2arctic/ABA/transcript/arctic_a0001.txt
...
```

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
docs/                  proposal, plan, context log, citations, study notes
requirements.txt
CLAUDE.md              instructions for AI coding assistants
```

## How we used AI tools

We used Claude Code (Anthropic) as a coding and research assistant. It helped draft the data-loading script, the README and an early version of the project plan, and it helped us look up and cross-check paper references. It also produced study notes that helped us understand the papers ([docs/bibliography_notes.md](docs/bibliography_notes.md)). We reviewed and edited everything it produced, ran the code ourselves, and made all decisions about the experimental design. Per course policy, the annotated-bibliography summaries are written entirely by us. *(Update this section as the project goes on.)*

## References

- Sagawa, S., Koh, P. W., Hashimoto, T. B., & Liang, P. (2020). Distributionally robust neural networks for group shifts. *ICLR 2020*.
- Sohoni, N., Dunnmon, J., Angus, G., Gu, A., & Ré, C. (2020). No subclass left behind: Fine-grained robustness in coarse-grained classification problems. *NeurIPS 2020*.
- Swain, M., et al. (2026). Towards fair ASR for second language speakers using fairness prompted finetuning. *ICASSP 2026*.
- Veliche, I.-E., & Fung, P. (2023). Improving fairness and robustness in end-to-end speech recognition through unsupervised clustering. *ICASSP 2023*.
- Radford, A., et al. (2023). Robust speech recognition via large-scale weak supervision. *ICML 2023*.
- Sanabria, R., et al. (2023). The Edinburgh International Accents of English Corpus: Towards the democratization of English ASR. *ICASSP 2023*.
- Zhao, G., et al. (2018). L2-ARCTIC: A non-native English speech corpus. *Interspeech 2018*.
