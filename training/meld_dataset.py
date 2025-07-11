from torch.utils.data import Dataset,DataLoader
import pandas as pd
from transformers import AutoTokenizer

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

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        video_path = os.path.join(self.video_dir,row['split_id'],row['video_id'])
        
        

if __name__ == "__main__":
    meld = MELDDataset(csv_path='../dataset/dev/dev_sent_emo.csv',video_dir='../dataset/dev/dev_splits_complete')
    print(meld.__len__())
    print(meld.__getitem__(0))
    