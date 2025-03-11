import os
import re
import shutil
import asyncio
import subprocess
from typing import Union
from xml.etree import ElementTree as ET

import numpy as np
from dotenv import load_dotenv

load_dotenv(verbose=True)

# ----------------------------
# GLOBAL CONSTANTS & CONFIG
# ----------------------------
INTERESTING_ACTIONS = [
    "pulling",
    "pushing",
    "punching",
    "kicking",
    "throwing",
    "piercing",
]

CHUNK_SIZE = 64
FPS = 30
NORMAL_STEP = 60


# ----------------------------
# 1. XML PARSING FUNCTIONS
# ----------------------------
def parse_annotation(xml_path: str) -> dict:
    """
    주어진 XML 파일에서 다음 정보를 추출한다:
      - filename
      - header/frames(전체 프레임 수)
      - event(eventname, starttime, duration)
      - object/action/frame (actionname과 start/end 프레임 구간)

    Returns:
        dict: {
            "filename": str or None,
            "eventname": str or None,
            "starttime": str or None,
            "duration": str or None,
            "actions": [
                {
                    "actionname": str or None,
                    "frames": [
                        {"start": int, "end": int},
                        ...
                    ]
                },
                ...
            ],
            "frame_cnt": str or None (전체 프레임 수)
        }
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # 1) filename, frame_cnt
    filename = None
    frame_cnt = None
    filename_elem = root.find("filename")
    if filename_elem is not None:
        filename = filename_elem.text

    header = root.find("header")
    if header is not None:
        frames_elem = header.find("frames")
        if frames_elem is not None:
            frame_cnt = frames_elem.text

    # 2) eventname, starttime, duration
    event_elem = root.find("event")
    eventname, starttime, duration = None, None, None
    if event_elem is not None:
        ename = event_elem.find("eventname")
        stime = event_elem.find("starttime")
        dur = event_elem.find("duration")
        eventname = ename.text if ename is not None else None
        starttime = stime.text if stime is not None else None
        duration = dur.text if dur is not None else None

    # 3) actions
    #    (XML 예시에선 <object>가 1개이지만, 필요하다면 여러 object 지원 가능)
    actions_list = []
    obj = root.find("object")
    if obj is not None:
        for action_elem in obj.findall("action"):
            actionname = None
            action_name_elem = action_elem.find("actionname")
            if action_name_elem is not None:
                actionname = action_name_elem.text

            frames = []
            for frame_elem in action_elem.findall("frame"):
                start_text = frame_elem.findtext("start")
                end_text = frame_elem.findtext("end")

                try:
                    start_val = int(start_text)
                except (TypeError, ValueError):
                    start_val = None

                try:
                    end_val = int(end_text)
                except (TypeError, ValueError):
                    end_val = None

                frames.append({"start": start_val, "end": end_val})

            actions_list.append({"actionname": actionname, "frames": frames})

    return {
        "filename": filename,
        "eventname": eventname,
        "starttime": starttime,
        "duration": duration,
        "actions": actions_list,
        "frame_cnt": frame_cnt,
    }


# ----------------------------
# 2. INTERVAL/FRAME UTILS
# ----------------------------
def merge_intervals(intervals: list) -> tuple[list, list]:
    """
    겹치는 구간들을 병합하여 새 리스트를 만든다.
    인접한 구간(끝지점+16 프레임 이내)도 겹치는 것으로 간주.

    Args:
        intervals (list): [{'start': x1, 'end': y1}, ...]

    Returns:
        (merged, merged_with_overlap):
            merged: 병합된 구간 [{'start': a, 'end': b}, ...]
            merged_with_overlap: 병합 전 개별 구간 리스트(겹치기 후보들)
    """
    if not intervals:
        return [], []

    # 시작점을 기준으로 정렬
    sorted_intervals = sorted(intervals, key=lambda x: x["start"])
    merged = [sorted_intervals[0]]
    merged_with_overlap = sorted_intervals[:]

    for current in sorted_intervals[1:]:
        last = merged[-1]

        # 만약 current.start가 last.end + 16 이내이면 병합
        if current["start"] <= last["end"] + 16:
            last["end"] = max(last["end"], current["end"])
            # 중복 제거
            if last in merged_with_overlap:
                merged_with_overlap.remove(last)
            if current in merged_with_overlap:
                merged_with_overlap.remove(current)
        else:
            merged.append(current)

    return merged, merged_with_overlap


def get_min_start_max_end(
    intervals: list,
) -> tuple[Union[int, None], Union[int, None]]:
    """
    병합된 구간 리스트에서 가장 작은 start 프레임과 가장 큰 end 프레임을 반환.
    Args:
        intervals (list): [{'start': x1, 'end': y1}, ...]

    Returns:
        (int | None, int | None): (min_start, max_end)
            빈 리스트일 경우 (None, None) 반환
    """
    if not intervals:
        return None, None

    min_start = min(i["start"] for i in intervals)
    max_end = max(i["end"] for i in intervals)
    return min_start, max_end


# ----------------------------
# 3. ASYNC FRAME EXPORT UTILS
# ----------------------------
async def save_frames_as_images(
    input_file: str,
    output_folder: str,
    start_frame: int,
    end_frame: int,
    frame_rate: int = FPS,
):
    """
    ffmpeg를 사용하여 (start_frame ~ end_frame) 구간을 이미지로 추출한다.
    비동기(subprocess) 방식으로 실행.

    Args:
        input_file (str): 비디오 파일 경로
        output_folder (str): 결과 이미지 폴더 경로
        start_frame (int): 시작 프레임 번호
        end_frame (int): 끝 프레임 번호
        frame_rate (int): 비디오 프레임레이트 (기본값 30)
    """
    os.makedirs(output_folder, exist_ok=True)

    # ffmpeg 필터링
    filter_str = f"select='between(n,{start_frame},{end_frame})',setpts=N/{frame_rate}/TB"
    output_pattern = os.path.join(output_folder, "img_%05d.jpg")

    command = [
        "ffmpeg",
        "-y",
        "-i",
        input_file,
        "-vf",
        filter_str,
        "-vsync",
        "vfr",
        "-q:v",
        "2",
        output_pattern,
    ]

    print(" ".join(command))  # 디버그용 출력
    process = await asyncio.create_subprocess_exec(*command)
    await process.wait()


def save_frames_by_chunks(
    input_file: str,
    output_base_folder: str,
    start_frame: int,
    end_frame: int,
    chunk_size: int = 64,
    prefix: str = "fight",
    step: int = 0,
    tasks: list = None,
):
    """
    (start_frame ~ end_frame) 구간을 'chunk_size' 단위로 잘라서,
    각 청크별로 save_frames_as_images()를 호출하는 비동기 tasks를 생성.

    Args:
        input_file (str): 비디오 파일 경로
        output_base_folder (str): 결과 이미지를 저장할 폴더(청크별 하위 폴더 생성)
        start_frame (int): 시작 프레임
        end_frame (int): 끝 프레임
        chunk_size (int): 한 번에 추출할 프레임 개수
        prefix (str): 폴더명 접두사
        step (int): 청크 간 간격(= 건너뛸 프레임 수), 기본 0
        tasks (list): asyncio Task를 저장할 리스트 (참조로 전달)
    """
    if tasks is None:
        tasks = []

    os.makedirs(output_base_folder, exist_ok=True)

    current = start_frame

    while current + chunk_size - 1 <= end_frame:
        chunk_start = current
        chunk_end = current + chunk_size - 1
        chunk_folder = os.path.join(
            output_base_folder,
            f"{prefix}_{(current // chunk_size) + 1}",
        )

        task = save_frames_as_images(
            input_file, chunk_folder, chunk_start, chunk_end
        )
        tasks.append(task)

        current += chunk_size + step


# ----------------------------
# 4. MAIN PROCESS LOGIC
# ----------------------------
async def process_videos(data_path: str, save_path: str):
    """
    data_path 하위 모든 디렉터리/파일에서 .xml 파일을 찾아:
    1) XML 파싱
    2) 대응되는 .mp4를 열어서
    3) 흥미로운 액션 프레임(병합)과 일반 프레임을 분리 추출
    4) 비동기적으로 이미지 출력 (ffmpeg)

    Args:
        data_path (str): 원본 동영상 + XML이 위치한 상위 경로
        save_path (str): 결과 이미지를 저장할 경로
    """
    # 디렉터리 순회
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if not file.endswith(".xml"):
                continue

            xml_path = os.path.join(root, file)
            xml_info = parse_annotation(xml_path)

            # 동일 이름의 mp4가 있다고 가정 (file[:-4].mp4)
            input_mp4 = os.path.join(root, file[:-3] + "mp4")
            output_folder = os.path.join(save_path, file[:-4])

            # 이미 결과 폴더가 있다면 스킵
            if os.path.exists(output_folder):
                continue

            # 전체 프레임 수
            total_frames = xml_info["frame_cnt"]
            if total_frames is None:
                continue
            total_frames = int(total_frames)

            # 액션 프레임 수집
            action_frames = []
            for action_dict in xml_info["actions"]:
                if action_dict["actionname"] in INTERESTING_ACTIONS:
                    action_frames.extend(action_dict["frames"])

            # 액션 구간 병합
            merged_actions, removed_frames = merge_intervals(action_frames)
            event_start, event_end = get_min_start_max_end(merged_actions)

            tasks = []

            # 1) Fight(Action) 프레임 청크 추출
            for interval in merged_actions:
                if interval in removed_frames:
                    continue
                save_frames_by_chunks(
                    input_mp4,
                    output_folder,
                    interval["start"],
                    interval["end"],
                    CHUNK_SIZE,
                    "fight",
                    0,
                    tasks,
                )

            # 2) Normal 프레임 청크 추출
            #    액션 구간 전/후로 NORMAL_STEP(초 단위) * FPS만큼 건너뛰며 추출
            if event_start is None or event_end is None:
                # 액션이 전혀 없으면 전체를 normal로
                save_frames_by_chunks(
                    input_mp4,
                    output_folder,
                    0,
                    total_frames,
                    CHUNK_SIZE,
                    "normal",
                    FPS * NORMAL_STEP,
                    tasks,
                )
            else:
                # 앞쪽 normal
                if event_start > 0:
                    front_end = (event_start - 1) - (FPS * 5)
                    if front_end > 0:
                        save_frames_by_chunks(
                            input_mp4,
                            output_folder,
                            0,
                            front_end,
                            CHUNK_SIZE,
                            "normal_front",
                            FPS * NORMAL_STEP,
                            tasks,
                        )
                # 뒤쪽 normal
                if event_end < total_frames:
                    back_start = (event_end + 1) + (FPS * 5)
                    if back_start < total_frames:
                        save_frames_by_chunks(
                            input_mp4,
                            output_folder,
                            back_start,
                            total_frames,
                            CHUNK_SIZE,
                            "normal_back",
                            FPS * NORMAL_STEP,
                            tasks,
                        )

            # 비동기 작업들 병렬 실행
            await asyncio.gather(*tasks)


# ----------------------------
# 5. MAIN ENTRY POINT
# ----------------------------
async def main():
    """
    메인 엔트리 포인트.
    .env 파일에서 VIDEO_DIR, DATA_PATH 읽어와 process_videos 실행.
    """
    data_path = os.getenv("VIDEO_DIR")
    save_path = os.getenv("DATA_PATH")

    if not data_path or not save_path:
        print("VIDEO_DIR, DATA_PATH 환경 변수를 설정하세요.")
        return

    os.makedirs(save_path, exist_ok=True)
    await process_videos(data_path, save_path)


if __name__ == "__main__":
    asyncio.run(main())
