# Project Context Log

A running record of decisions, status, and key takeaways for the team (and for AI assistants helping us).
Last updated: 2026-11-18.

## Deadlines (from the CS 445 course schedule)
| Date | Deadline |
|---|---|
| 10/29 | Topic proposal + preliminary bibliography ✅ |
| **11/19** | **Annotated bibliography + project plan + GitHub repo** |
| 12/03 | Project checkpoint (≤ 2-page progress report with figures) |
| 12/10 | Poster draft (in-class peer review) |
| 12/17 | Poster session (36"×24" landscape, QR code to this repo) |

## Assignment rules worth remembering
- **Annotated bibliography:**
  - ≥ 5 peer-reviewed papers
  - ≥ 3 that cite another paper in the list
  - ≥ 3 from the last 4 years
  - each summary covers key results, cross-references, what we didn't understand, and 1–2 sentences on how it bears on the project
  - **Using AI to generate the summary text is an honor-code violation.** AI may only help us understand the papers and check facts.
- **Project plan:** about 1 page covering Goal, Literature, Data, Methods (+ baseline), Evaluation, Compute, Timeline, Fallback.
- **Repo:** README (describe the project, how to run it, how we used AI tools) plus code that runs.
- arXiv-only papers count only if peer-reviewed or clearly seminal.

## Bibliography status
- **Citations:** [annote.bib](annote.bib) (7 papers; van Zee 2024 optional 8th, commented out).
- **Summaries:** written by us in Overleaf.
  - Drafted: Sagawa, Sohoni, Koenecke, Dheram.
  - To do: Veliche & Fung, Radford, Swain.
- **Study notes** (AI-generated, for understanding only): [bibliography_notes.md](bibliography_notes.md).
- **Still to verify:**
  - Sohoni page numbers
  - DOIs/pages for Veliche & Fung and Swain
  - Radford → Koenecke citation
  - full author first names in annote.bib
- **LaTeX pitfalls:**
  - `%` → `\%` (otherwise the rest of the line vanishes)
  - `&` → `\&` (otherwise compile error)
  - `---` for dashes
  - ``` ``quotes'' ``` and a straight `'` apostrophe
  - `et al.` with a period; "et al." takes a plural verb
- **Entry types:** `@InProceedings` for conference papers, `@Article` for journals (Koenecke / PNAS).

## Key takeaways from reading the papers
- **Why Group DRO needs regularization (Sagawa):**
  - big networks memorize the training set, so every group's training loss is about 0
  - Group DRO's weights have nothing to react to, so it behaves like ERM
  - early stopping or strong L2 keeps the rare groups' loss visible
  - for BERT (a transformer) only early stopping helped, so it's our primary regularizer for Whisper
- **How Group DRO finds the worst group:**
  - it needs group labels in advance
  - it keeps a weight per group and multiplies the weight up when that group's training loss is high
  - it shifts gradually and never "picks" one group
  - "worst" is judged by *training loss*, not WER
- **Sagawa did not invent Group DRO** (Hu et al. 2018; Oren et al. 2019). It made Group DRO work on large networks.
- **The paper never names the ERM worst group on Waterbirds.** The 60% is just "worst-group accuracy."
- **k-means vs. finding the gap:** k-means only groups similar voices. Finding the worst cluster is a separate step: Dheram ranked clusters by ASR confidence. In our project, Group DRO plays that role during training.
- **We adapt GEORGE; we don't use it.** See the comparison table in [bibliography_notes.md](bibliography_notes.md#george-vs-our-project).
- **Dheram's mitigation was on ZIP-code cohorts only**, never on the clusters, so discovery and mitigation were separate studies.
- **Koenecke is about race/dialect, not accents.** Its main comparison is matched on age, gender, and duration, *not* identical sentences. Only its 206-phrase cause test used identical phrases.
- **Swain:** Group DRO with true labels was *worse* than ERM for Whisper-large on EdAcc, so a negative result for our condition 2 is plausible.

## Decisions
- **Repo:** standalone public repo (separate from the CS445 coursework repo).
- **Primary dataset:** L2-ARCTIC. **Backup:** EdAcc (streamed from Hugging Face; works now).
- **Split:**
  - per L1: 2 train speakers (1M/1F), 1 validation, 1 test, alternating sex so the test set is 3M/3F
  - sentences split 80/10/10 by ID, so no test sentence is heard in training
- **Model:** Whisper-base (74M).
- **Clustering:** ECAPA-TDNN embeddings + k-means (k = 6, plus a sweep); optional UMAP + silhouette-chosen k.
- **Conditions:**
  1. ERM
  2. Group DRO + true L1
  3. Group DRO + clusters
  4. Group DRO + random groups
- **Evaluation:** per-L1 WER on held-out speakers and sentences; 3 seeds.

## Open to-dos
- [ ] Submit the L2-ARCTIC request form (https://psi.engr.tamu.edu/l2-arctic-corpus/)
- [ ] Fill in names (README, project plan)
- [ ] Add partner as a GitHub collaborator
- [ ] Finish the Veliche & Fung, Radford, Swain summaries
- [ ] Decide on van Zee as an 8th paper

## Environment notes
- Local Python: conda env `env445` (`C:\Users\aaqui\miniconda3\envs\env445\python.exe`).
- Installing `datasets` upgraded `huggingface-hub` to 2.x, which broke `transformers` 5.x. Fixed by pinning `huggingface-hub<2.0` (now in requirements.txt).
