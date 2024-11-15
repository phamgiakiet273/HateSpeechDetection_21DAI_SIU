from youtube_comment_downloader import YoutubeCommentDownloader, SORT_BY_POPULAR
import pandas as pd
import ujson
from itertools import islice

def save_comment_from_video_link(video_link, output_raw_comment_path, k=50):

    video_link = video_link
    downloader = YoutubeCommentDownloader()


    comments = downloader.get_comments_from_url(video_link, sort_by=SORT_BY_POPULAR, language='en')

    save_json = []

    for comment in islice(comments, k):
        save_json.append([comment['text'], comment['time']])

    df = pd.DataFrame(save_json)
    df.to_csv(output_raw_comment_path, index=False, encoding='utf-8-sig', header=False)
    
    print("Get Comments Completed")
