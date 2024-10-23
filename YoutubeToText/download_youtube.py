import os
from pytubefix import YouTube

def save_video_from_link(video_link, save_path):
    yt = YouTube(video_link)

    # Get all streams and filter for mp4 files
    mp4_streams = yt.streams.filter(file_extension='mp4').all()

    # Get the video with the highest resolution
    d_video = mp4_streams[-1]

    # Download the video
    video_path = d_video.download(output_path=save_path)

    # Count existing mp4 files in the save directory
    existing_videos = [f for f in os.listdir(save_path) if f.endswith('.mp4')]
    index = len(existing_videos)

    # Generate new filename
    new_filename = f"video_{index}.mp4"
    new_file_path = os.path.join(save_path, new_filename)

    # Rename the downloaded video
    os.rename(video_path, new_file_path)

    return new_file_path