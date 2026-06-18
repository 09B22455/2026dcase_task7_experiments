import shutil
import pandas as pd

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent

DOWNLOAD_DIR = (
    ROOT_DIR
    / "data"
    / "downloaded_corpus"
)

D2_DIR = DOWNLOAD_DIR / "DIL-DCASE26-Dev-D2"
D3_DIR = DOWNLOAD_DIR / "DIL-DCASE26-Dev-D3"

TASK7_DIR = (
    ROOT_DIR
    / "data"
    / "task7_data"
)

AUDIO_DIR = TASK7_DIR / "audio"
EVAL_DIR = TASK7_DIR / "evaluation_setup"


LABEL_MAP = {
    "alarm": 0,
    "baby": 1,
    "dog": 2,
    "engine": 3,
    "fire": 4,
    "footsteps": 5,
    "knock": 6,
    "phone": 7,
    "piano": 8,
    "speech": 9,
}

def normalize_label(label):

    mapping = {
        "baby_cry": "baby",
        "dog_bark": "dog",
        "knocking": "knock",
        "telephone_ringing": "phone",
    }

    return mapping.get(label, label)

def copy_split(domain, split, source_dir):

    print(f"copying {domain} {split}")

    for wav_path in source_dir.glob("*.wav"):

        dst_name = (
            f"{domain}_"
            f"{split}_"
            f"{wav_path.name}"
        )

        dst = AUDIO_DIR / dst_name

        if not dst.exists():
            shutil.copy2(wav_path, dst)

def build_metadata(csv_path, domain, split):

    df = pd.read_csv(csv_path)

    rows = []

    for _, row in df.iterrows():

        filename = (
            "audio/"
            f"{domain}_{split}_{row['filename']}"
        )

        label_name = normalize_label(row["class"])

        rows.append([
            filename,
            label_name,
            domain,
            LABEL_MAP[label_name]
        ])

    return pd.DataFrame(
        rows,
        columns=[
            "filename",
            "target",
            "domain",
            "new_target"
        ]
    )

def main():

    AUDIO_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    EVAL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ====================
    # copy wav
    # ====================

    copy_split(
        "D2",
        "train",
        D2_DIR / "d2-dev-train"
    )

    copy_split(
        "D2",
        "test",
        D2_DIR / "d2-dev-test"
    )

    copy_split(
        "D3",
        "train",
        D3_DIR / "d3-dev-train"
    )

    copy_split(
        "D3",
        "test",
        D3_DIR / "d3-dev-test"
    )

    # ====================
    # metadata
    # ====================

    train_df = pd.concat([
        build_metadata(
            D2_DIR / "metadata" / "d2-dev-train.csv",
            "D2",
            "train"
        ),
        build_metadata(
            D3_DIR / "metadata" / "d3-dev-train.csv",
            "D3",
            "train"
        )
    ])

    test_df = pd.concat([
        build_metadata(
            D2_DIR / "metadata" / "d2-dev-test.csv",
            "D2",
            "test"
        ),
        build_metadata(
            D3_DIR / "metadata" / "d3-dev-test.csv",
            "D3",
            "test"
        )
    ])

    train_df.to_csv(
        EVAL_DIR / "development_train.txt",
        sep="\t",
        index=False,
        header=False
    )

    test_df.to_csv(
        EVAL_DIR / "development_test.txt",
        sep="\t",
        index=False,
        header=False
    )

    print()
    print("Dataset organized successfully")
    print(
        "audio files:",
        len(list(AUDIO_DIR.glob("*.wav")))
    )

if __name__ == "__main__":
    main()