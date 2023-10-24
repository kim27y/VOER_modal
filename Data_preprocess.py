import os
from moviepy.editor import *

# 비디오 파일이 있는 폴더 경로
video_folder = './datas/'

# 오디오를 저장할 폴더 경로
output_audio_folder = './Audio/data/wav'
os.makedirs(output_audio_folder, exist_ok=True) 

# 폴더 내의 모든 비디오 파일에 대해 반복
for root, dirs, files in os.walk(video_folder):
    for file in files:
        if file.endswith(".mp4"):  # 비디오 파일 형식에 따라 수정
            video_path = os.path.join(root, file)
            audio_output_path = os.path.join(output_audio_folder, f"{file.split('.')[0]}.wav")
            
            # 비디오 파일에서 오디오 추출
            video = VideoFileClip(video_path)
            audio = video.audio.to_audiofile(audio_output_path)
            
            print(f"Extracted audio from {file} and saved as {audio_output_path}")