import os
import re
import pandas as pd
from torch.utils.data import Dataset

class TextDataset(Dataset):
    '''
        Expecting csv files with columns ['sent1', 'sent2']

        Args:
                        data_path: Root folder Containing all the data
                        dataset: Specific Folder==> data_path/dataset/    (Should contain train.csv and dev.csv)
                        max_length: Self Explanatory
                        is_debug: Load a subset of data for faster testing
                        is_train: 

    '''

    def __init__(self, data_path='data', dataset='mawps', datatype='train', max_length=30, is_debug=False, is_train=False):
        if datatype=='train':
            file_path = os.path.join(data_path, dataset, 'train.json')
        elif datatype=='dev':
            file_path = os.path.join(data_path, dataset, 'dev.json')
        else:
            file_path = os.path.join(data_path, dataset, 'dev.json')

        self.challenge_info = False
    
        file_df= pd.read_json(file_path, orient='index') # TODO add parser for pd.read_csv  

        self.ques= file_df['Question'].values
        self.eqn= file_df['Equation'].values
        self.nums= file_df['Numbers'].values
        self.ans= file_df['Answer'].values

        if is_debug:
            self.ques= self.ques[:5000:500]
            self.eqn= self.eqn[:5000:500]

        self.max_length= max_length
        all_sents = zip(self.ques, self.eqn, self.nums, self.ans)

        if is_train:
            all_sents = sorted(all_sents, key = lambda x : len(x[0].split()))

    
        else:
            self.ques, self.eqn, self.nums, self.ans = zip(*all_sents)

    def __len__(self):
        return len(self.ques)

    def __getitem__(self, idx):
        ques = self.process_string(str(self.ques[idx]))
        eqn = self.process_string(str(self.eqn[idx]))
        nums = self.nums[idx]
        ans = self.ans[idx]

    
        return {'ques': self.curb_to_length(ques), 'eqn': self.curb_to_length(eqn), 'nums': nums, 'ans': ans}
        # return {'ques': self.curb_to_length(ques)}

    def curb_to_length(self, string):
        return ' '.join(string.strip().split()[:self.max_length])

    def process_string(self, string):
        #string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " 's", string)
        string = re.sub(r"\'ve", " 've", string)
        string = re.sub(r"n\'t", " n't", string)
        string = re.sub(r"\'re", " 're", string)
        string = re.sub(r"\'d", " 'd", string)
        string = re.sub(r"\'ll", " 'll", string)
        #string = re.sub(r",", " , ", string)
        #string = re.sub(r"!", " ! ", string)
        #string = re.sub(r"\(", " ( ", string)
        #string = re.sub(r"\)", " ) ", string)
        #string = re.sub(r"\?", " ? ", string)
        #string = re.sub(r"\s{2,}", " ", string)
        return string

