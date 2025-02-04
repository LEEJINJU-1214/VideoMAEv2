import os
from concurrent.futures import ThreadPoolExecutor

import cv2
from sklearn.model_selection import train_test_split

# 데이터 경로 설정
data_root = "/media/bom/8tb_SSD/fight"
output_image_root = "/media/bom/8tb_SSD/fight_images/"
output_csv_root = "data/fight/"
os.makedirs(output_image_root, exist_ok=True)
os.makedirs(output_csv_root, exist_ok=True)

train_csv = "train.csv"
valid_csv = "val.csv"
test_csv = "test.csv"
random_state = 42

save_frame = 64


# 비디오 파일에서 이미지를 추출하여 저장
def extract_images_from_video(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    count = 0
    folder_count = 0
    extracted_folders = []

    save_cnt = frame_count // save_frame

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        os.makedirs(output_dir, exist_ok=True)

        if count % save_frame == 0:
            if folder_count == save_cnt:
                break

            folder_name = f"folder_{folder_count:03d}"
            current_folder = os.path.join(output_dir, folder_name)
            if current_folder not in extracted_folders:
                os.makedirs(current_folder, exist_ok=True)
                extracted_folders.append(current_folder)
            folder_count += 1
            count = 0

        # 이미지 저장
        image_name = f"img_{count + 1:05d}.jpg"
        image_path = os.path.join(current_folder, image_name)
        cv2.imwrite(image_path, frame)

        count += 1

    cap.release()
    return extracted_folders


# 비디오 파일 수집 및 이미지 저장, 클래스 번호 매기기
def collect_video_files_and_save_images(data_root, output_image_root):
    video_files = []

    def process_video(file_path, label):
        # 이미지 저장 경로 생성
        output_dir = os.path.join(
            output_image_root, os.path.splitext(os.path.basename(file_path))[0]
        )

        # 이미지 추출 및 저장
        folder_paths = extract_images_from_video(file_path, output_dir)
        return [
            (folder_path, save_frame, label) for folder_path in folder_paths
        ]

    tasks = []
    with ThreadPoolExecutor() as executor:
        for root, _, files in os.walk(data_root):
            for file in files:
                if file.endswith(".mp4"):  # MP4 파일만 포함
                    file_path = os.path.join(root, file)
                    label = 0 if "nomal" in file else 1
                    tasks.append(
                        executor.submit(process_video, file_path, label)
                    )

        for task in tasks:
            video_files.extend(task.result())

    return video_files


# 데이터를 train/valid/test로 분할
def split_data(video_files, train_ratio=0.7, valid_ratio=0.2):
    train_data, temp_data = train_test_split(
        video_files,
        test_size=(1 - train_ratio),
        random_state=random_state,
    )
    valid_data, test_data = train_test_split(
        temp_data,
        test_size=(1 - valid_ratio / (1 - train_ratio)),
        random_state=random_state,
    )
    return train_data, valid_data, test_data


# 스페이스바로 구분된 파일 생성
def write_space_separated(file_path, data):
    with open(file_path, mode="w", encoding="utf-8") as file:
        for row in data:
            file.write(f"{row[0]} {row[1]} {row[2]}\n")


# 실행
if __name__ == "__main__":
    # 비디오 파일 수집 및 이미지 저장
    video_files = collect_video_files_and_save_images(
        data_root, output_image_root
    )

    # 데이터를 train, valid, test로 분할
    train_data, valid_data, test_data = split_data(video_files)

    # 각각의 데이터를 스페이스로 구분된 파일로 저장
    write_space_separated(os.path.join(output_csv_root, train_csv), train_data)
    write_space_separated(os.path.join(output_csv_root, valid_csv), valid_data)
    write_space_separated(os.path.join(output_csv_root, test_csv), test_data)

    print(f"파일 생성 완료:\n- {train_csv}\n- {valid_csv}\n- {test_csv}")
