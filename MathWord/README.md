# MathWords

모델의 문제를 읽고 수식을 이해하여 생성하는 모델에 대한 코드

## Datasets

- svamp 
  - dir: data/mawps-asdiv-a_svamp/
  - train, dev (csv, json format 둘 다 가능 - default format: json)
  - label.json: 정답지(gold answer)


  ### Examples

각 문제마다 질문("Question")과 계산에 사용되는 숫자("Numbers"), 질문에 해당하는 수식("Equation")을 입력으로 사용


  ![Screenshot from 2021-05-17 13-19-34](https://user-images.githubusercontent.com/44221520/118432097-bf32bd80-b712-11eb-8bb2-96d9aea200c7.png)

## Model
- Seq2Seq
  - RNN layer - choices = [RNN, LSTM, GRU]
  - attention - choices = [Luong, Bahdanau]
- Pretrained embedding
  - BERT
  - Roberta(Default)
  - Word2vec
  
  
  사전 학습 모델을 통해 얻은 임베딩으로 Encoder를 통과시킨 후 얻은 context vector를  Decoder layer에 전달하여 수식을 생성
  생성된 수식을 그대로 계산하여 얻은 답을 실제 ground truth와 비교

  

## Weights for inference(google cloud)
### components
#### 1. Vocabulary for each question and equation
- 질문을 파악하기 위한 vocabulary set과 수식을 이해하기 위한 vocabulary set
#### 2. Model weights
미리 학습시킨 가중치 link

https://drive.google.com/drive/folders/1U6N95jkWHjsZy-jlnaXGpLygxyssGAF7?usp=sharing

  
## Installation 

Install all the required packages:

```bash 
pip install -r requirements.txt

```

결과물 관련 폴더 생성
```bash 
sh setup.sh

components 
    - logs: train and inference
    - weights: weights for inference
    - data: data dir
    - runs: history of events(train or inference)
    - outputs
    - answer: folder contains answer.json(result of prediction)

```

## Usage - Run locally

```bash
sh main.sh

or

python3 main.py \
       -mode test \
       -gpu 0 \
       -embedding roberta \
       -emb_name roberta-base \
       -emb1_size 768 \
       -hidden_size 256 \
       -data_path data/ \
       -depth 2 \
       -lr 0.0002 \
       -emb_lr 8e-6 \
       -batch_size 8 \
       -epochs 50 \
       -dataset mawps-asdiv-a_svamp \
       -save_model

-mode를 train으로 변경하면 학습모드로 변경됨
```


## Results
answer 폴더에 json 형식으로 저장됨 

![Screenshot from 2021-05-17 13-28-56](https://user-images.githubusercontent.com/44221520/118432584-e8a01900-b713-11eb-8fe1-7d45a7672bd2.png)
