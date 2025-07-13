from torch.utils.data import Dataset,DataLoader
import pandas as pd
from transformers import AutoTokenizer
import os
import sys
from pathlib import Path
import numpy as np
import subprocess
import cv2
import torch


os.environ["TOKENIZERS_PARALLELISM"] = "false"


class MELDDataset(Dataset):

    def __init__(self,csv_path,video_dir) -> None:
        self.data = pd.read_csv(csv_path)
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.video_dir = video_dir
        self.emotion_map = {
            'anger':0,
            'disgust':1,
            'fear':2,
            'joy':3,
            'neutral':4,
            'sadness':5,
            'surprise':6,
        }
        self.sentiment_map={
            'negative':0,
            'neutral':1,
            'positive':2,
        }
        super().__init__()

    def _load_video_frames(self,path):
        cap = cv2.VideoCapture(path)
        frames = []

        try:
            if not cap.isOpened():
                raise ValueError(f"Video not found: {path}")

            # Try and read first frame to validate video
            ret,frame = cap.read()
            if not ret or frame is None:
                raise ValueError(f"Video not found: {path}")
            
            # Reset index to not skip first frame
            cap.set(cv2.CAP_PROP_POS_FRAMES,0)

            while len(frames) < 30 and cap.isOpened():
                ret,frame = cap.read()
                if not ret:
                    break

                frame = cv2.resize(frame,(224,224))
                frame = frame / 255.0
                frames.append(frame)
            
        except Exception as e:
            raise ValueError(f"Video error: {e}")
        finally:
            cap.release()
        
        if (len(frames) == 0):
            raise ValueError(f"No frames could be extracted from video: {path}")

        # Pad or truncate frames to 30
        if(len(frames) < 30):
            frames += [np.zeros_like(frames[0])] * (30 - len(frames))
        else:
            frames = frames[:30]
        
        # Before permute: [frames,height,width,channels]
        # After permute: [frames,channels,height,width]
        return torch.FloatTensor(np.array(frames)).permute(0,3,1,2)

    def _extract_audio_features(self,video_path):
        audio_path = video_path.replace('.mp4','.wav')

        try:
            subprocess.run([
                        'ffmpeg',
                        '-i',video_path,
                        '-vn',
                        '-acodec','pcm_s16le',
                        '-ar','16000',
                        '-ac','1',
                        audio_path
                    ],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
            )

            waveform,sample_rate = torchaudio.load(audio_path)

            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate,16000)
                waveform = resampler(waveform)
            
            mel_spectrogram = torchaudio.transforms.MelSpectrogram(
                sample_rate=16000,
                n_mels=64,
                n_fft=1024,
                hop_length=512,
            )

            mel_spec = mel_spectrogram(waveform)
            
            # Normalize
            mel_spec = (mel_spec - mel_spec.mean()) / mel_spec.std()

            if mel_spec.size(2) < 300:
                padding = 300  - mel_spec.size(2)
                mel_spec = torch.nn.functional.pad(mel_spec,(0,padding))
            else:
                mel_spec = mel_spec[:,:,:300]
            
            return mel_spec
        except subprocess.CalledProcessError as e:
            raise ValueError(f"Audio extraction failed: {e}")
        except Exception as e:
            raise ValueError(f"Audio error: {e}")
        finally:
            if os.path.exists(audio_path):
                os.remove(audio_path)


    def __len__(self):
        return len(self.data)
    

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        video_filename = f"""dia_{row['Dialogue_ID']}_utt{row['Utterance_ID']}.mp4"""

        path = os.path.join(self.video_dir,video_filename)
        video_path = os.path.exists(path)
        if not video_path:
            raise FileNotFoundError(f"Video not found for filename: {path}")
        
        text_inputs = self.tokenizer(row['Utterance'],
                padding='max_length',
                truncation=True,
                max_length=512,
                return_tensors='pt'
          )

        video_frames = self._load_video_frames(path)
        audio_features = self._extract_audio_features(path)
        print("file found")
        

def collate_fn(batch):
    # Filter out any None values
    batch = list(filter(None,batch))
    return torch.utils.data.dataloader.default_collate(batch)

def prepare_data_loaders(
    train_csv,train_video_dir,
    dev_csv,dev_video_dir,
    test_csv, test_video_dir, batch_size=32):

    train_dataset = MELDDataset(train_csv,train_video_dir)
    dev_dataset = MELDDataset(dev_csv,dev_video_dir)
    test_dataset = MELDDataset(test_csv,test_video_dir)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn
    )

    return train_loader,dev_loader,test_loader

if __name__ == "__main__":
    # Get the absolute path to the project root
    project_root = Path(os.path.abspath(__file__)).parent.parent
    
    # Create paths relative to the project root
    csv_path = os.path.join(project_root, 'dataset', 'dev', 'dev_sent_emo.csv')
    video_dir = os.path.join(project_root, 'dataset', 'dev', 'dev_splits_complete')
    
    # Verify paths exist before attempting to create dataset
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        sys.exit(1)
    if not os.path.exists(video_dir):
        print(f"Error: Video directory not found at {video_dir}")
        sys.exit(1)
        
    print(f"Using CSV file at: {csv_path}")
    print(f"Using video directory at: {video_dir}")
    
    meld = MELDDataset(csv_path=csv_path, video_dir=video_dir)
    print(f"Dataset length: {meld.__len__()}")

