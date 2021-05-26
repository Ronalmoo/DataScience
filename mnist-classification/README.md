# MNIST-Classification

## 1. Getting started

### 1.1. Project structure
MNIST-Classification 프로젝트의 구조는 아래와 같습니다.

```bash
.
├── apply.sh
├── conf
├── docker
├── evaluate.py
├── __init__.py
├── LICENSE
├── main.py
├── Makefile
├── predict.py
├── pretrained_weights
├── README.md
├── requirements.txt
├── src
├── tests
├── train.py
└── utils
```

### 1.2. Installation
본 프로젝트의 설치는 1). Docker 기반 설치, 2). Local 환경 설치, 두 가지 방법이 있습니다.

#### 1.2.1. Docker-based
Docker-based로 설치하기 위해서는 아래의 순서대로 명령어를 입력

1. `make set-mnist-dataset`
2. `make build-docker`
3. `make run-docker`
4. `make exec-docker`
5. `python3 main.py evaluate`

#### 1.2.2. Local environments-based
**1.2.2.1. pre-requisites**  
- python3 >= 3.6.x

**1.2.2.2. install dependency package**  
python package manager인 `pip`를 활용하여 의존성 패키지들을 설치
```bash
$ python3 -m pip install -r requirements.txt
```


### 1.3. Usage
프로젝트에서 사용할 수 있는 커맨드는 `Makefile`에서 정의되어 `make` 명령어를 기반으로 실행되는 1).`build-docker`, 2).`run-docker`, 3).`exec-docker`, 4).`rm-docker`, 5).`clean`와 python package인 [`type-docopt`](https://github.com/dreamgonfly/type-docopt) 에서 정의되어 python script에서 실행되는 1). `train`, 2). `predict`, 3). `evaluate`, 4). `test`로 구성

1. **`Makefile`로 정의된 커맨드**
    `Makefile`로 정의된 커맨드는 다음과 같은 방식으로 실행
    ```bash
    $ make build-docker
    ```

    각 커맨드의 의미
    - `build-docker`: `docker/Dockerfile`을 빌드
    - `run-docker`: build한 docker image의 container를 실행
    - `exec-docker`: 실행한 docker container에 접속
    - `rm-docker`: 실행한 docker container를 중지하고 제거
    - `clean`: 작업을 하면서 부수적으로 생성된 python, test코드의 잔여물을(`cache파일`) 삭제

2. **`type-docopt`로 정의된 커맨드**
   `type-docopt`로 정의된 커맨드는 다음과 같은 방식으로 실행할 수 있습니다.
   ```bash
   $ python3 main.py evaluate
   ```

   각 커맨드의 의미
   - `train`: MNIST 데이터셋의 손글씨를 인식하는 인공지능 모델을 학습
   - `predict`: 손글씨 이미지가 어떤 손글씨인지 예측
   - `evaluate`: 손글씨 이미지를 인식하는 인공지능 모델의 성능을 평가


### 1.4. Commands you should use to test the AIGC
아래의 커맨드를 사용
```bash
$ python3 main.py evaluate
```

위 명령어는 MNIST 테스트 데이터셋을 모두 예측한 후에, 이를 `answer.json`에 기록하여 프로젝트의 root 디렉토리에 저장합니다.
