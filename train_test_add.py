import os
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
import csv
load_dotenv(verbose=True)

# 데이터 경로 설정
DATA_ROOT = os.getenv("DATA_PATH")
OUTPUT_CSV_ROOT = os.getenv("TRAIN_CSV_DIR")
os.makedirs(OUTPUT_CSV_ROOT, exist_ok=True)

# 기존 CSV 파일 경로 설정
train_csv = os.path.join(OUTPUT_CSV_ROOT, "train.csv")
valid_csv = os.path.join(OUTPUT_CSV_ROOT, "val.csv")
test_csv = os.path.join(OUTPUT_CSV_ROOT, "test.csv")
all_data_csv = os.path.join(OUTPUT_CSV_ROOT, "all_data.csv")

random_state = 42

# 기존 데이터 불러오기
def load_existing_data(csv_file):
    """기존 CSV 파일을 불러오거나 빈 DataFrame을 반환"""
    return pd.read_csv(csv_file, sep=" ", header=None, names=["video_path", "total_frames", "label"]) if os.path.exists(csv_file) else pd.DataFrame(columns=["video_path", "total_frames", "label"])

# 기존 데이터를 불러옴
existing_train = load_existing_data(train_csv)
existing_valid = load_existing_data(valid_csv)
existing_test = load_existing_data(test_csv)

with open(os.getenv("LABEL_DIR"), "r", encoding="utf-8") as f:
    lines = f.readlines()
label_names = [line.strip() for line in lines]

Min_Frame = 64
Max_Frame = 100

save_col = ["label"]

# 새로운 데이터 수집
Data = {
    "video_path": [],
    "total_frames": [],
    "label": [],
}

def save_plot(file_path, train_data, valid_data, test_data, attribute, save_arg = None):
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
    if save_arg:
        save_name = f"train_test_split_{attribute+save_arg}.png"
    else:
        save_name = f"train_test_split_{attribute}.png"
    plt.savefig(
        os.path.join(OUTPUT_CSV_ROOT, save_name),
        bbox_inches="tight",
    )

# 데이터를 train/valid/test로 분할
def split_data(dataframe, train_ratio=0.7, valid_ratio=0.2):
    train_data, temp_data = train_test_split(
        dataframe,
        test_size=(1 - train_ratio),
        random_state=random_state,
        stratify=dataframe["label"],
    )
    valid_data, test_data = train_test_split(
        temp_data,
        test_size=(1 - valid_ratio / (1 - train_ratio)),
        random_state=random_state,
        stratify=temp_data["label"],
    )
    return train_data, valid_data, test_data

# 스페이스바로 구분된 파일 추가 저장
def append_space_separated(file_path, new_data):
    """기존 파일이 있으면 데이터를 추가 저장"""
    # if os.path.exists(file_path):
    #     new_data.to_csv(file_path, mode="a", header=False, index=False, sep=" ")
    # else:
    #     new_data.to_csv(file_path, header=False, index=False, sep=" ")
    # new_data_frame = pd.read_csv(file_path)
    shuffled_df = new_data.sample(frac=1, random_state=random_state).reset_index(drop=True)
    shuffled_df.to_csv(file_path, header=False, index=False, sep=" ",quoting=csv.QUOTE_NONE)

    
    

# 실행
if __name__ == "__main__":
    # 새로운 데이터를 수집
    for root, dirs, files in os.walk(DATA_ROOT):
        video_path = root.split("/")
        if (
            len(os.listdir(root)) < Min_Frame
            or len(os.listdir(root)) > Max_Frame
        ):
            continue
        video_event = video_path[-1].split("_")[0]
        if (video_event) not in label_names:
            continue
        label = label_names.index(video_event)
        Data["label"].append(label)
        Data["video_path"].append(root)
        Data["total_frames"].append(len(os.listdir(root)))
    # 새로운 데이터프레임 생성
    new_data_frame = pd.DataFrame(Data)

    # train, valid, test 데이터로 분할
    train_data, valid_data, test_data = split_data(new_data_frame)

    # 기존 데이터와 합침
    train_data = pd.concat([existing_train, train_data], ignore_index=True)
    valid_data = pd.concat([existing_valid, valid_data], ignore_index=True)
    test_data = pd.concat([existing_test, test_data], ignore_index=True)

    # 각각의 데이터를 기존 파일에 추가 저장
    append_space_separated(train_csv, train_data)
    append_space_separated(valid_csv, valid_data)
    append_space_separated(test_csv, test_data)

    print(f"파일 업데이트 완료:\n- {train_csv}\n- {valid_csv}\n- {test_csv}")

    # 전체 데이터 저장
    if os.path.exists(all_data_csv):
        all_data_frame = pd.read_csv(all_data_csv)
        all_data_frame = pd.concat([all_data_frame, new_data_frame], ignore_index=True)
    else:
        all_data_frame = new_data_frame

    all_data_frame.to_csv(all_data_csv, index=False)

    print(f"전체 데이터 저장 완료: {all_data_csv}")
    for attribute in save_col:
        save_plot(
            OUTPUT_CSV_ROOT, train_data, valid_data, test_data, attribute
        )