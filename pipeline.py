import sys
import os
import shutil
import pickle
from flask import Flask, render_template, send_file, request, redirect, url_for, jsonify
import zipfile

from DataPreprocessing.hateSD_preprocess import preprocess
from YoutubeToText.download_youtube import save_video_from_link
from YoutubeToText.extract_comment import save_comment_from_video_link
from YoutubeToText.extract_transcription import save_transcript_from_video_path
from ToxicDetection.detection import process_csv

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
                                          output_NER_comment_path, output_NER_transcript_path,
                                          output_detection_cleaned_comment_path,
                                          output_detection_cleaned_transcript_path,
                                          output_detection_cleaned_NER_comment_path,
                                          output_detection_cleaned_NER_transcript_path):
        
        
        preprocess(comment_path, transcript_path, 
           output_comment_path, output_transcript_path, 
           output_NER_comment_path, output_NER_transcript_path)
        
        process_csv(output_comment_path, output_detection_cleaned_comment_path)
        process_csv(output_transcript_path, output_detection_cleaned_transcript_path)
        process_csv(output_NER_comment_path, output_detection_cleaned_NER_comment_path)
        process_csv(output_NER_transcript_path, output_detection_cleaned_NER_transcript_path)
        
        print("Preprocess Completed")
        
        
# video_link = "https://www.youtube.com/watch?v=hytigOSjJxc"
video_save_path = "samples/videos"
wav_path = "samples/videos/wav_path"

output_raw_comment_path = "samples/raw_data/raw_comment.csv"
output_raw_transcript_path = "samples/raw_data/raw_transcript.csv"
output_cleaned_comment_path = "samples/clean_data/cleaned_comments.csv"
output_cleaned_transcript_path = "samples/clean_data/cleaned_transcripts.csv"
output_cleaned_NER_comment_path = "samples/clean_data/cleaned_comments_NER.csv"
output_cleaned_NER_transcript_path = "samples/clean_data/cleaned_transcripts_NER.csv"

output_detection_cleaned_comment_path = "samples/output/cleaned_comments.csv"
output_detection_cleaned_transcript_path = "samples/output/cleaned_transcripts.csv"
output_detection_cleaned_NER_comment_path = "samples/output/cleaned_comments_NER.csv"
output_detection_cleaned_NER_transcript_path = "samples/output/cleaned_transcripts_NER.csv"

base_path = "E:/HateSpeechDetection_21DAI_SIU/"

hateSpeechDetection = HateSpeechDetection()


app = Flask(__name__, template_folder='WebUI/templates', static_folder='WebUI/static')


@app.route('/')
def main():
    return render_template('home.html')

@app.route('/download')
def download_file():
    # Create a zip file name
    zip_filename = "clean_data_files.zip"
    zip_filepath = os.path.join("output", zip_filename)

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(zip_filepath), exist_ok=True)

    # Create a zip file
    try:
        with zipfile.ZipFile(zip_filepath, 'w') as zipf:
            zipf.write(output_cleaned_comment_path, arcname=os.path.basename(output_cleaned_comment_path))
            zipf.write(output_cleaned_transcript_path, arcname=os.path.basename(output_cleaned_transcript_path))
            zipf.write(output_cleaned_NER_comment_path, arcname=os.path.basename(output_cleaned_NER_comment_path))
            zipf.write(output_cleaned_NER_transcript_path, arcname=os.path.basename(output_cleaned_NER_transcript_path))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # Send the zip file for download
    if os.path.exists(zip_filepath):
        return send_file(zip_filepath, as_attachment=True)
    else:
        return jsonify({"error": "File not found"}), 404
    
@app.route('/csv/<path:filename>', methods=['GET'])
def serve_csv(filename):
    filepath = os.path.join("output", filename)
    print(filepath)  # This will help you debug if the path is correct
    if os.path.exists(filepath):
        return send_file(filepath, as_attachment=True)  # Use as_attachment=True to prompt download
    else:
        return jsonify({"error": "File not found"}), 404

@app.route('/process', methods=['POST'])
def video_processing():
    if request.method == 'POST':
        video_path = request.form.get('youtube_url')
        if video_path:
            hateSpeechDetection.video_to_comment_and_transcript(
                video_link=video_path,
                video_save_path=video_save_path,
                wav_path=wav_path,
                output_raw_comment_path=output_raw_comment_path,
                output_raw_transcript_path=output_raw_transcript_path
                
            )
            hateSpeechDetection.preprocess_comment_and_transcript(output_raw_comment_path, output_raw_transcript_path, 
                output_cleaned_comment_path, output_cleaned_transcript_path, 
                output_cleaned_NER_comment_path, output_cleaned_NER_transcript_path,
                                          output_detection_cleaned_comment_path,
                                          output_detection_cleaned_transcript_path,
                                          output_detection_cleaned_NER_comment_path,
                                          output_detection_cleaned_NER_transcript_path)
            
            return jsonify({
                    "output_cleaned_comment_path": base_path + output_detection_cleaned_comment_path,
                    "output_cleaned_transcript_path": base_path + output_detection_cleaned_transcript_path,
                    "output_cleaned_NER_comment_path": base_path + output_detection_cleaned_NER_comment_path,
                    "output_cleaned_NER_transcript_path": base_path + output_detection_cleaned_NER_transcript_path
                    })
        return jsonify({"error": "Missing URL"}), 400

if __name__ == "__main__":
    app.run(port=8502, debug=False)