import sys
import os
import shutil
import pickle
from flask import Flask, render_template, send_file, request, redirect, url_for

from DataPreprocessing.hateSD_preprocess import preprocess
from YoutubeToText.download_youtube import save_video_from_link
from YoutubeToText.extract_comment import save_comment_from_video_link
from YoutubeToText.extract_transcription import save_transcript_from_video_path

class HateSpeechDetection:
    def __init__(self):
        #pass
        print("Init Completed")
    
    
    def video_to_comment_and_transcript(self, video_link, wav_path, video_save_path, output_raw_comment_path, output_raw_transcript_path):
        video_path = save_video_from_link(video_link, video_save_path)
        save_comment_from_video_link(video_link, output_raw_comment_path)
        save_transcript_from_video_path(video_path, wav_path, output_raw_transcript_path)
        for item in os.listdir(video_save_path):
            file_path = os.path.join(video_save_path, item)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # Remove file or symbolic link
    
    def preprocess_comment_and_transcript(self, comment_path, transcript_path, 
                                          output_comment_path, output_transcript_path, 
                                          output_NER_comment_path, output_NER_transcript_path):
        
        
        preprocess(comment_path, transcript_path, 
           output_comment_path, output_transcript_path, 
           output_NER_comment_path, output_NER_transcript_path)
        
        print("Preprocess Completed")
        
        
# video_link = "https://www.youtube.com/watch?v=hytigOSjJxc"
video_save_path="E:/HateSpeechDetection_21DAI_SIU/samples/videos"
wav_path="E:/HateSpeechDetection_21DAI_SIU/samples/videos/wav_path"

output_raw_comment_path = "E:/HateSpeechDetection_21DAI_SIU/samples/raw_data/raw_comment.csv"
output_raw_transcript_path = "E:/HateSpeechDetection_21DAI_SIU/samples/raw_data/raw_transcript.csv"
output_cleaned_comment_path = "E:/HateSpeechDetection_21DAI_SIU/samples/clean_data/cleaned_comments.csv"
output_cleaned_transcript_path = "E:/HateSpeechDetection_21DAI_SIU/samples/clean_data/cleaned_transcripts.csv"
output_cleaned_NER_comment_path = "E:/HateSpeechDetection_21DAI_SIU/samples/clean_data/cleaned_comments_NER.csv"
output_cleaned_NER_transcript_path = "E:/HateSpeechDetection_21DAI_SIU/samples/clean_data/cleaned_transcripts_NER.csv"


hateSpeechDetection = HateSpeechDetection()


app = Flask(__name__, template_folder='E:/HateSpeechDetection_21DAI_SIU/WebUI/templates', static_folder='E:/HateSpeechDetection_21DAI_SIU/WebUI/static')
path_parent = os.getcwd()


@app.route('/')
def main():
    return render_template('home.html')

@app.route('/download')
def download_file():

    path = path_parent + '/result.csv'
    return send_file(path, as_attachment=True)

@app.route('/process', methods=['POST'])
def video_processing():
    if request.method == 'POST':
        video_path = request.form['youtube-url']
        hateSpeechDetection.video_to_comment_and_transcript(video_link=video_path,
                                                            video_save_path=video_save_path,
                                                            wav_path=wav_path,
                                                            output_raw_comment_path=output_raw_comment_path,
                                                            output_raw_transcript_path=output_raw_transcript_path)
        
        print('Done')


@app.route('/predict', methods=['POST'])
def custom():
    
    path_csv = request.form['task']    
    # Classify toxic text
    if request.method == 'POST':
        hateSpeechDetection.preprocess_comment_and_transcript(output_raw_comment_path, output_raw_transcript_path, 
                output_cleaned_comment_path, output_cleaned_transcript_path, 
                output_cleaned_NER_comment_path, output_cleaned_NER_transcript_path)
        
    return render_template('home.html', message="Processing complete. You can download the results.")

if __name__ == "__main__":
    app.run(port=8502, debug=False)