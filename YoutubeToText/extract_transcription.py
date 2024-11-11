from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection
from transformers import  Wav2Vec2Processor, Wav2Vec2ForCTC
import torch
import subprocess
import pydub
import numpy as np
import os
import math
from pydub import AudioSegment
from collections import defaultdict, OrderedDict
import ujson
import pandas as pd
import shutil

class SpeechToText():
    
    #read wav file and return frame_rate, vector
    def read(self, f, normalized=True):
        a = pydub.AudioSegment.from_mp3(f)
        y = np.array(a.get_array_of_samples())
        if a.channels == 2:
            y = y.reshape((-1, 2))
        if normalized:
            return a.frame_rate, np.float32(y) / 2**15
        else:
            return a.frame_rate, y
    
    #convert audio from audio path to vector (normalized)
    def audio_to_vector(self, audio_path):
        sr, x = self.read(audio_path)
        return x
    
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        #load processor and model for speech to text
        self.processor =  Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english", use_auth_token="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")
        self.model = Wav2Vec2ForCTC.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english", use_auth_token="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").to(self.device)

        #init audio model and pipeline for voice activity detection
        self.audio_model = Model.from_pretrained("pyannote/segmentation", 
                              use_auth_token="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").to(self.device)
        self.audio_pipeline = VoiceActivityDetection(segmentation=self.audio_model, device=torch.device(self.device))
        HYPER_PARAMETERS = {
            # onset/offset activation thresholds
            "onset": 0.5, "offset": 0.5,
            # remove speech regions shorter than that many seconds.
            "min_duration_on": 0.5,
            # fill non-speech regions shorter than that many seconds.
            "min_duration_off": 0.0
        }
        self.audio_pipeline.instantiate(HYPER_PARAMETERS)

    #return transcript from audio file with beam search decode
    #audio file must be .wav file and 16k sampling rate
    def speech_to_text(self, audio_path : str):
        audio_vector = self.audio_to_vector(audio_path)
        inputs = self.processor(audio_vector, sampling_rate=16_000, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs).logits
        # transcription = self.ngram_lm_model.decode(outputs.cpu().detach().numpy()[0], beam_width=500)
        predicted_ids = torch.argmax(outputs, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)
        return transcription

    #return transcript from video file
    #the function will save splitted .wav files in wav_path
    #the return is a dict : { tuple(start,end) : transcript }
    def video_to_text(self, video_path : str, wav_path : str):
        video_name = os.path.basename(video_path)
        print(video_name)
        total_wav_path = wav_path + str(video_name).replace(".mp4","") + ".wav"
        if not (os.path.exists(total_wav_path)):
            try:
                command = "ffmpeg.exe -y -i "+ video_path + " -ac 1 -ar 16000 " + total_wav_path
                subprocess.call(command, shell=True)
            except:
                command = "ffmpeg -y -i "+ video_path + " -ac 1 -ar 16000 " + total_wav_path
                subprocess.call(command, shell=True)
        video_speech_region = self.audio_pipeline(total_wav_path)
        video_audio = AudioSegment.from_wav(total_wav_path)
        video_transcript = {}
        for speech in video_speech_region.get_timeline().support():
            # active speech between speech.start and speech.end
            start = math.floor(speech.start)
            end = math.ceil(speech.end)
            print("speech start: ", start, " ---- ", "speech end: ", end)
            cur_audio = video_audio[start*1000:end*1000+1]
            cur_audo_path = wav_path + "speech"+str(start)+"-"+str(end)+".wav"
            cur_audio.export(cur_audo_path, format="wav")
            transcription = self.speech_to_text(cur_audo_path)
            video_transcript[str(str(start) + "_" + str(end))] = transcription
        #text = self.speech_to_text(wav_path)
        return OrderedDict(video_transcript)
        


def save_transcript_from_video_path(video_path, wav_path, save_raw_transcript_path):

    s2t = SpeechToText()
    s2t.video_to_text(video_path=video_path, wav_path=wav_path+'/')
    
    s2t_list = []
    for item in os.listdir(wav_path):
        if str(item).startswith("speech"):
            s2t_list.append(s2t.speech_to_text(audio_path=wav_path+'/'+str(item))[0])
        
    with open(save_raw_transcript_path, 'w') as outfile:
        ujson.dump(s2t_list, outfile, indent=4)

    df = pd.DataFrame(s2t_list)
    df.to_csv(save_raw_transcript_path, index=False, header=False, encoding='utf-8-sig')
    
    for item in os.listdir(wav_path):
        file_path = os.path.join(wav_path, item)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)  # Remove file or symbolic link