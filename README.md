# Hate Speech Detection on Youtube based on Speech Transcription and Comments

An academic-based project for The Saigon University NLP course. The aim is to detect hate / offensive content from the speech transcription of video and comment on the Youtube platform. 

## Description

The Hate Speech Detection project aims to leverage Natural Language Processing (NLP) techniques to identify and categorize hate speech and offensive content on the YouTube platform. By analyzing both speech transcriptions of videos and user comments, this project seeks to develop a robust model capable of distinguishing between acceptable discourse and harmful rhetoric. The project employs advanced machine learning algorithms to process and analyze textual data extracted from video content and user interactions. By providing an efficient detection system, this initiative not only contributes to maintaining a safer online community but also enhances the understanding of hate speech dynamics in digital communications. The outcomes of this research can serve as valuable resources for content moderation teams, researchers, and educators, paving the way for more effective interventions against hate speech on social media platforms.

## Getting Started

### Dependencies


* Windows / Linux
* Python 3.9.20
* ffmpeg
  
### Installing for Windows

* Put [ffmpeg.exe](https://www.gyan.dev/ffmpeg/builds/ffmpeg-git-full.7z) from the bin folder to the same path as pipeline.py
* Put [models file](https://drive.google.com/drive/folders/1VbTjNVxTeUODK4A7Fkh2puvJ9tKbfHvc?usp=sharing) in corresponding folder (read detailed instruction inside ToxicDetection folder)
```
git clone https://github.com/phamgiakiet273/HateSpeechDetection_21DAI_SIU/
pip install -r requirements.txt
pip install --upgrade --force-reinstall "numpy<1.24"
```
* If have NVIDIA GPU
```
pip install --upgrade --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### Installing Linux

* Install [ffmpege](https://itsfoss.com/ffmpeg/) for Linux if not available
* Put [models file](https://drive.google.com/drive/folders/1VbTjNVxTeUODK4A7Fkh2puvJ9tKbfHvc?usp=sharing) in corresponding folder (read detailed instruction inside ToxicDetection folder)
```
git clone https://github.com/phamgiakiet273/HateSpeechDetection_21DAI_SIU/
pip install -r requirements.txt
pip install --upgrade --force-reinstall "numpy<1.24"
```

### Executing program

Adjust **base_path** in pipeline.py according to the installation location then:

```
python pipeline.py
```

## Version History
* 1.1
    * Use CUDA if available, or else CPU
    * Add instruction and adaptability for Linux installation
    * Fix preprocessing bug (floating character)
    * Update requirements.txt to use Keras 2
    * Disable printing in Detection model to speed up inference
    * Create needed folders
* 1.0
    * Merge and run
* 0.2
    * Bug fixs for modules
    * Web UI
* 0.1
    * Initial release of each module

## Acknowledgments

Please refer the the corresponding readme from each subdirectories of this git.
