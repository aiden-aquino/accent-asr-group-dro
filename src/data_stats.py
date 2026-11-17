"""Print summary statistics for the project's speech datasets.

Two sources are supported:

  l2arctic  Local copy of L2-ARCTIC v5.0 (request it at
            https://psi.engr.tamu.edu/l2-arctic-corpus/ and unzip it).
            Expected layout: <data-dir>/<SPEAKER>/wav/*.wav and
            <data-dir>/<SPEAKER>/transcript/*.txt
  edacc     EdAcc (Edinburgh International Accents of English Corpus),
            streamed from Hugging Face. Only the first --max-rows rows
            are read, so nothing large gets downloaded.

Examples:
  python src/data_stats.py --source l2arctic --data-dir data/l2arctic
  python src/data_stats.py --source edacc --max-rows 2000
"""

import argparse
import sys
import wave
from collections import Counter, defaultdict
from pathlib import Path

# Speaker -> (first language, gender), from the L2-ARCTIC README.
# 4 speakers per L1, 2 male + 2 female.
L2ARCTIC_SPEAKERS = {
    "ABA": ("Arabic", "M"), "SKA": ("Arabic", "F"),
    "YBAA": ("Arabic", "M"), "ZHAA": ("Arabic", "F"),
    "BWC": ("Mandarin", "M"), "LXC": ("Mandarin", "F"),
    "NCC": ("Mandarin", "F"), "TXHC": ("Mandarin", "M"),
    "ASI": ("Hindi", "M"), "RRBI": ("Hindi", "M"),
    "SVBI": ("Hindi", "F"), "TNI": ("Hindi", "F"),
    "HJK": ("Korean", "F"), "HKK": ("Korean", "M"),
    "YDCK": ("Korean", "F"), "YKWK": ("Korean", "M"),
    "EBVS": ("Spanish", "M"), "ERMS": ("Spanish", "M"),
    "MBMPS": ("Spanish", "F"), "NJS": ("Spanish", "F"),
    "HQTV": ("Vietnamese", "M"), "PNV": ("Vietnamese", "F"),
    "THV": ("Vietnamese", "F"), "TLV": ("Vietnamese", "M"),
}


def wav_seconds(path):
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / w.getframerate()


def l2arctic_stats(data_dir):
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        sys.exit(f"Data directory not found: {data_dir}\n"
                 "Download L2-ARCTIC (see README) or try --source edacc.")

    rows = []  # (speaker, l1, gender, n_utts, seconds, sentence_ids)
    for spk_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        spk = spk_dir.name
        if spk not in L2ARCTIC_SPEAKERS:
            continue
        wavs = sorted((spk_dir / "wav").glob("*.wav"))
        secs = sum(wav_seconds(w) for w in wavs)
        rows.append((spk, *L2ARCTIC_SPEAKERS[spk], len(wavs), secs,
                     {w.stem for w in wavs}))

    if not rows:
        sys.exit(f"No L2-ARCTIC speaker folders found in {data_dir}")

    print(f"\nL2-ARCTIC at {data_dir}")
    print(f"{'speaker':<8}{'L1':<12}{'sex':<5}{'utts':>6}{'minutes':>10}")
    for spk, l1, g, n, secs, _ in sorted(rows, key=lambda r: (r[1], r[0])):
        print(f"{spk:<8}{l1:<12}{g:<5}{n:>6}{secs / 60:>10.1f}")

    by_l1 = defaultdict(lambda: [0, 0, 0.0])
    for _, l1, _, n, secs, _ in rows:
        by_l1[l1][0] += 1
        by_l1[l1][1] += n
        by_l1[l1][2] += secs
    print(f"\n{'L1':<12}{'speakers':>9}{'utts':>8}{'hours':>8}")
    for l1, (s, n, secs) in sorted(by_l1.items()):
        print(f"{l1:<12}{s:>9}{n:>8}{secs / 3600:>8.2f}")

    # Everyone reads the same prompts, so the train/test split must hold out
    # sentences as well as speakers. Count how many prompts are shared.
    sentence_sets = [r[5] for r in rows]
    shared = set.intersection(*sentence_sets)
    total_sents = set.union(*sentence_sets)
    print(f"\nSpeakers: {len(rows)}  |  utterances: {sum(r[3] for r in rows)}"
          f"  |  hours: {sum(r[4] for r in rows) / 3600:.1f}")
    print(f"Distinct sentence IDs: {len(total_sents)}  |  read by all "
          f"speakers: {len(shared)}")


def edacc_stats(max_rows):
    try:
        from datasets import Audio, load_dataset
    except ImportError:
        sys.exit("EdAcc needs the Hugging Face datasets library: "
                 "pip install -r requirements.txt")

    ds = load_dataset("edinburghcstr/edacc", split="validation",
                      streaming=True)
    # Don't decode the audio; we only need lengths from the metadata.
    ds = ds.cast_column("audio", Audio(decode=False))

    speakers = {}
    utts = Counter()
    words = Counter()
    for i, ex in enumerate(ds):
        if i >= max_rows:
            break
        spk = ex["speaker"]
        speakers[spk] = (ex.get("l1") or "?", ex.get("accent") or "?",
                         ex.get("gender") or "?")
        utts[spk] += 1
        words[spk] += len((ex.get("text") or "").split())

    n = sum(utts.values())
    print(f"\nEdAcc validation split, first {n} utterances "
          f"({len(speakers)} speakers)")
    by_l1 = defaultdict(lambda: [set(), 0, 0])
    for spk, (l1, _, _) in speakers.items():
        by_l1[l1][0].add(spk)
        by_l1[l1][1] += utts[spk]
        by_l1[l1][2] += words[spk]
    print(f"{'L1':<28}{'speakers':>9}{'utts':>7}{'words':>8}")
    for l1, (spks, u, w) in sorted(by_l1.items(), key=lambda kv: -kv[1][1]):
        print(f"{l1[:27]:<28}{len(spks):>9}{u:>7}{w:>8}")
    genders = Counter(g for _, _, g in speakers.values())
    print(f"\nGender (speakers): {dict(genders)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["l2arctic", "edacc"],
                    default="l2arctic")
    ap.add_argument("--data-dir", default="data/l2arctic",
                    help="unzipped L2-ARCTIC folder (l2arctic only)")
    ap.add_argument("--max-rows", type=int, default=2000,
                    help="rows to stream (edacc only)")
    args = ap.parse_args()

    if args.source == "l2arctic":
        l2arctic_stats(args.data_dir)
    else:
        edacc_stats(args.max_rows)


if __name__ == "__main__":
    main()
