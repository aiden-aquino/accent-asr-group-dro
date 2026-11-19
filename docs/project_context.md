# Project Context Log

A running record of decisions, status, and key takeaways for the team (and for AI assistants helping us).
Last updated: 2026-11-19.

## Deadlines (from the CS 445 course schedule)
| Date | Deadline |
|---|---|
| 10/29 | Topic proposal + preliminary bibliography ✅ |
| **11/19** | **Annotated bibliography + project plan + GitHub repo** |
| 12/03 | Project checkpoint (≤ 2-page progress report with figures) |
| *12/07* | *Internal: all 4 conditions × 3 seeds finished* |
| 12/10 | Poster draft (in-class peer review) |
| *12/13–14* | *Internal: final figures, error examples, peer-review fixes; poster sent to print* |
| 12/17 | Poster session (36"×24" landscape, QR code to this repo) |

## Assignment rules worth remembering
- Assignment page: https://w3.cs.jmu.edu/spragunr/CS445_F26/hw/poster.shtml (local PDF copy in docs/, not committed)
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
  - Dataset papers are cited in the plan and README only, **not** in the annotated bib: **EdAcc** (Sanabria et al., 2023; ⚠ details unverified) and **L2-ARCTIC** (Zhao et al., 2018). Keep Sanabria23 out of the Overleaf bib, or `\nocite{*}` prints it as an unannotated entry.
- **Submission format:** one Overleaf `main.tex` with the project plan first, then `\newpage` and the annotated bibliography (heading renamed with `\renewcommand{\refname}{Annotated Bibliography}`), `geometry` with 1in margins. [project_plan.md](project_plan.md) is a markdown copy of the submitted plan.
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
- **Primary dataset: EdAcc** (switched from L2-ARCTIC on 2026-11-18). **Backup: L2-ARCTIC** (controlled read-speech test of conditions 1 vs. 2).
  - Why: L2-ARCTIC leaves only 12 training speakers to cluster (~2 per cluster at k = 6). EdAcc gives ~80–90 after holding out val/test, needs no request form, and matches Swain.
  - EdAcc check (full stream of both HF splits, 2026-11-18):
    - 122 speakers, 19,137 utterances; 59 M / 62 F / 1 other
    - ~34 h estimated from word counts (published ~40 h)
    - minutes per speaker (estimated): median 12.4, min 2.7, max 69.7; 118 speakers have ≥ 5 min
    - no speaker appears in both HF splits (validation 9,848 rows, test 9,289)
    - 41 L1 labels, 17 with ≥ 3 speakers; 32 accent labels, 17 with ≥ 3 speakers
    - labels need cleanup: "Spanish" vs. "Spanish (Mexican)", typos ("Portoguese", "Ghanain"), a "Don't know" accent group
  - Size: ~7 GB (validation 3.3 GB, test 3.6 GB).
- **Split:** pool both HF splits, re-split by speaker (~70/15/15), stratified by group and balanced by sex where possible. Save the speaker lists as a small JSON in the repo so every machine uses the same split.
- **True-label groups:** clean the labels and merge the long tail into broader groups; check Swain's 26-group scheme first. *(Not decided yet.)*
- **Model:** Whisper-base (74M).
- **Clustering:** ECAPA-TDNN embeddings averaged per speaker + k-means (k = number of true groups, plus a sweep); optional UMAP + silhouette-chosen k.
- **Compute:** train on the MSI GL66 (RTX 3070 laptop, 8 GB VRAM); Colab T4 as backup or a second machine. This laptop has no CUDA GPU, so it only streams data, runs stats and does quick CPU tests.
- **Conditions:**
  1. ERM
  2. Group DRO + true L1
  3. Group DRO + clusters
  4. Group DRO + random groups
- **Evaluation:** per-group WER on held-out speakers; 3 seeds; bootstrap CIs (many groups have only 1–2 test speakers).
  - Headline number: the % of the ERM → true-label improvement in worst-group WER that the clusters recover (0% = no better than ERM, 100% = matches true labels). Meaningless if condition 2 doesn't beat ERM; then compare conditions 3 and 4 with ERM directly.

## Open to-dos
- [ ] Verify the EdAcc citation details (Sanabria et al., ICASSP 2023: authors, pages, DOI) for the plan/poster
- [ ] Decide the true-label grouping for EdAcc (check Swain's grouping); save the label → group mapping in the repo
- [ ] Set up the GL66 (see Environment notes)
- [ ] Submit the L2-ARCTIC request form anyway, as the backup (https://psi.engr.tamu.edu/l2-arctic-corpus/)
- [ ] Check what format the poster draft should be (printed, on a laptop, or PDF) and JMU's print turnaround
- [x] Mirror the EdAcc switch in the Overleaf version of the plan
- [x] Fill in names (README, project plan)
- [ ] Add partner as a GitHub collaborator
- [ ] Finish the Veliche & Fung, Radford, Swain summaries
- [ ] Decide on van Zee as an 8th paper

## Environment notes
- Local Python: conda env `env445` (`C:\Users\aaqui\miniconda3\envs\env445\python.exe`).
- Installing `datasets` upgraded `huggingface-hub` to 2.x, which broke `transformers` 5.x. Fixed by pinning `huggingface-hub<2.0` (now in requirements.txt).
- This laptop: Intel UHD graphics only, CPU-only PyTorch (`2.12.0+cpu`), ~9 GB free disk. Stream EdAcc here; don't download it.
- **GL66 setup checklist** (training machine):
  1. `nvidia-smi` to confirm the RTX 3070 and driver; check ≥ 20 GB free disk
  2. Miniconda, `conda create -n env445 python=3.11`
  3. CUDA build of PyTorch (exact command from pytorch.org), then `pip install -r requirements.txt` (keeps `huggingface-hub<2.0`)
  4. Confirm `python -c "import torch; print(torch.cuda.get_device_name(0))"`
  5. If the SSD is short on space, put the HF cache on the external hard drive: `setx HF_HOME "E:\hf_cache"`, then open a new terminal (not the thumb drive)
  6. Download EdAcc (non-streaming) and do a short ERM test run to check speed and temperature
- Colab: browser only, nothing to install. Its disk resets each session, so re-download EdAcc and save checkpoints to Google Drive.
