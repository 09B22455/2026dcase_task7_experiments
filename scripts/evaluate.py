import pandas as pd
import torch
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from config import (
    TEST_CSV,
    AUDIO_DIR,
    DEVICE,
    CLASS_NAMES,
    TASK1_MODEL_PATH,
    TASK2_MODEL_PATH,
)

from datasets.datasetfactory_task7 import DILDatasetInc
from models.task7_models import load_model

def evaluate_domain(model, test_df, audio_dir, domain):

    df = test_df
    df = df[df["domain"] == domain]

    dataset = DILDatasetInc(df, audio_dir)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    print()
    print(f"[{domain}]")
    print("num samples =", len(dataset))

    correct = 0
    total = 0

    num_classes = len(CLASS_NAMES)

    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    with torch.no_grad():

        for inputs, targets, audio_file in loader:
            inputs = inputs.float().to(DEVICE)
            logits = model(inputs)
            pred = torch.argmax(
                logits,
                dim=1
            )
            gt = torch.argmax(
                targets,
                dim=1
            )
            pred_idx = pred.item()
            gt_idx = gt.item()

            total += 1

            class_total[gt_idx] += 1

            if pred_idx == gt_idx:
                correct += 1
                class_correct[gt_idx] += 1

    acc = 100.0 * correct / total

    print(f"{domain} Accuracy = " f"{acc:.2f}")

    print()
    print("Class-wise Accuracy")

    for i, class_name in enumerate(CLASS_NAMES):

        if class_total[i] > 0:
            class_acc = (100.0 * class_correct[i] / class_total[i])
        else:
            class_acc = 0.0

        print(f"{class_name:20s}" f"{class_acc:6.2f}")

    return acc



def evaluate_task1(test_df):

    model = load_model(model_path=TASK1_MODEL_PATH, task=1)
    model.to(DEVICE)
    model.eval()

    print()
    print("========== Task1 ==========")

    evaluate_domain(model, test_df, AUDIO_DIR, "D2")

def evaluate_task2(test_df):

    model = load_model(model_path=TASK2_MODEL_PATH, task=2)
    model.to(DEVICE)
    model.eval()

    print()
    print("========== Task2 ==========")

    acc_d2 = evaluate_domain(model, test_df, AUDIO_DIR, "D2")
    acc_d3 = evaluate_domain(model, test_df, AUDIO_DIR, "D3")

    print()
    print(f"Average = " f"{(acc_d2 + acc_d3) / 2:.2f}")


def main():

    test_df = pd.read_csv(
        TEST_CSV,
        sep="\t",
        names=[
            "filename",
            "target",
            "domain",
            "new_target"
        ]
    )

    evaluate_task1(test_df)
    evaluate_task2(test_df)

if __name__ == "__main__":
    main()