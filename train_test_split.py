import os

import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

load_dotenv(verbose=True)
# 데이터 경로 설정

DATA_ROOT = os.getenv("DATA_PATH")
OUTPUT_CSV_ROOT = os.getenv("TRAIN_CSV_DIR")
os.makedirs(OUTPUT_CSV_ROOT, exist_ok=True)


with open(os.getenv("LABEL_DIR"), "r", encoding="utf-8") as f:
    lines = f.readlines()
label_names = [line.strip() for line in lines]

train_csv = "train.csv"
valid_csv = "val.csv"
test_csv = "test.csv"

random_state = 42

Data = {
    "video_path": [],
    "total_frames": [],
    "label": [],
    "video_name": [],
    "time": [],
    "season": [],
    "strat_col": [],
}

Min_Frame = 64
Max_Frame = 100

save_col = ["label", "season", "time", "strat_col", "video_name"]


# 데이터를 train/valid/test로 분할
def split_data(dataframe, train_ratio=0.7, valid_ratio=0.2):
    train_data, temp_data = train_test_split(
        dataframe,
        test_size=(1 - train_ratio),
        random_state=random_state,
        stratify=dataframe["strat_col"],
    )
    valid_data, test_data = train_test_split(
        temp_data,
        test_size=(1 - valid_ratio / (1 - train_ratio)),
        random_state=random_state,
        stratify=temp_data["strat_col"],
    )
    return (
        train_data,
        valid_data,
        test_data,
    )


# 스페이스바로 구분된 파일 생성
def write_space_separated(file_path, data):
    data.to_csv(
        file_path,
        columns=["video_path", "total_frames", "label"],
        index=False,
        header=False,
        sep=" ",
    )


def save_plot(file_path, train_data, valid_data, test_data, attribute):
    import matplotlib.pyplot as plt

    train_counts = train_data[attribute].value_counts(normalize=True)
    valid_counts = valid_data[attribute].value_counts(normalize=True)
    test_counts = test_data[attribute].value_counts(normalize=True)

    # 예: 막대 그래프 여러 개로 비교
    fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharey=True)

    train_counts.plot(kind="bar", ax=axes[0], title="Train attribute dist")
    valid_counts.plot(kind="bar", ax=axes[1], title="Valid attribute dist")
    test_counts.plot(kind="bar", ax=axes[2], title="Test attribute dist")

    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_CSV_ROOT, f"train_test_split_{attribute}.png"),
        bbox_inches="tight",
    )


# 실행
if __name__ == "__main__":
    # 데이터를 train, valid, test로 분할
    for root, dirs, files in os.walk(DATA_ROOT):
        video_path = root.split("/")
        if (
            len(os.listdir(root)) < Min_Frame
            or len(os.listdir(root)) > Max_Frame
        ):
            continue
        video_event = video_path[-1].split("_")[0]
        label = label_names.index(video_event)
        Data["label"].append(label)
        Data["video_path"].append(root)
        Data["season"].append(video_path[-2].split("_")[-1])
        Data["time"].append(video_path[-2].split("_")[-2])
        Data["video_name"].append(video_path[-2].split("-")[0])
        Data["total_frames"].append(len(os.listdir(root)))

        Data["strat_col"].append(
            "_".join(
                [
                    video_path[-2].split("_")[4],
                    video_path[-2].split("_")[5],
                ]
            )
            + "_"
            + str(label)
        )

    data_frame = pd.DataFrame(Data)
    train_data, valid_data, test_data = split_data(data_frame)

    # 각각의 데이터를 스페이스로 구분된 파일로 저장
    for csv, split_data in zip(
        [train_csv, valid_csv, test_csv], [train_data, valid_data, test_data]
    ):
        write_space_separated(os.path.join(OUTPUT_CSV_ROOT, csv), split_data)

    print(f"파일 생성 완료:\n- {train_csv}\n- {valid_csv}\n- {test_csv}")
    data_frame.to_csv(
        os.path.join(OUTPUT_CSV_ROOT, "all_data.csv"), index=False
    )
    for attribute in save_col:
        save_plot(
            OUTPUT_CSV_ROOT, train_data, valid_data, test_data, attribute
        )
