# Digit Detection :zero:~:nine: :computer: 
### using the TensorFlow2 object detection api 
![Footer](https://capsule-render.vercel.app/api?type=waving&color=auto&height=200&section=footer)

# :pushpin:Project Description
Anaconda 환경에서 TensorFlow 2 Object Detection API 를 사용하여 필기체 이미지를 95% 확률로 검출을 수행하는 프로그램이다.

# :pushpin:Project Purpose
- 전이학습된 모델 SSD ResNet50 V1 FPN 640x640을 숫자 검출을 위한 모델로 훈련시켜 숫자 검출률 95%이상 높이기
- tensorflow2 object detection api를 응용하여 숫자 검출 과제에 맞게 수정하여 사용해보기 위함
- epoch별 모델 성능을 비교해보고 모델 평가를 분석해보며 공부한 내용 복습

# :pushpin:Project Dataset
- 사용한 데이터셋 :  google 이미지 다운 (train데이터 : 0~9 xml파일 약 18개 , 예측 데이터 : 10개)
- 객체 클래스 수 : 0~9 총 10개

# :pushpin:Project Results
- 필기체 인식률 : 96%
- 컬러 , 흑백 모두 인식 가능
![7bdfe0a10dd2d5af57a75d6432d0da8ed54373e7_re_1674143258386](https://user-images.githubusercontent.com/105347300/214508661-f1239640-a307-4654-96f6-ad8dc645749a.png)
- 100개의 숫자중 4개 틀림
- epochs =25000
![73807c832a82f390d06462f9f68486f11ae2f9c6_re_1674143258386](https://user-images.githubusercontent.com/105347300/214508795-4ec91da3-99b8-43ee-ba4a-86b862a80622.png)

# :pushpin:Project Pre-Work
- 아나콘다 파이썬 3.8 설치
- TensorFlow 설치
- GPU 지원  -CUDA 툴킷 v11.2 , CuDNN 8.1.0 설치 , 환경변수 설정
- TensorFlow 객체 감지 API 설치
- Protobuf 설치/컴파일 , COCO API 설치 ,객체 감지 API 설치
- Resolving errors during installation : pip install --upgrade protobuf==3.20.0 로 변경

# :pushpin:Project Process (Training Custom Object Detector)
#### :dizzy: 객체 인식 정의
- 객체 인식 :  이미지나 영상 내에 있는 객체를 식별하는 컴퓨터 비전 기술
- 객체 인식 = 여러가지 객체에 대한 분류 + 객체의 위치 정보를 파악하는 위치 검출
- 딥러닝을 이용한 객체 인식 알고리즘은 1단계 객체 인식과 2단계 객체 인식으로 나눌 수 있음.
- 1단계 : 분류와 위치검출을 동시에 행하는 방법 =>객체인식 비교적 빠름, 정확도 낮음(yolo,ssd)
- 2단계 : 분류와 위치검출 순차적으로 행하는 방법 =>객체인식 비교적 느림, 정확도 높음(r-cnn계열)
----------------------------------------------------------------
## :loudspeaker: 데이터세트에 주석 달기 
- LabelImg  사용
![76d36c161bc6bfbed6b5216a47a0160afd69ad0c_re_1674143258383](https://user-images.githubusercontent.com/105347300/214510142-cd76c0b2-36d6-487b-816b-2f6cb15cdf91.png)
- 0~9 까지의 훈련 데이터 객체에 주석을 달아 xml파일 생성
----------------------------------------------------------------
## :loudspeaker: Create Label Map (레이블 맵 생성)
- TensorFlow에는 사용된 각 레이블을 정수 값에 매핑하는 레이블 맵이 필요함.
- label_map.pbtxt파일 사용
![889fb11ec81fa699e1f749f4b5989b3ba926ee55_re_1674143258384](https://user-images.githubusercontent.com/105347300/214510342-9c46f47b-59a7-4c59-a26c-9ed3c9679318.png)
- id는 1부터 시작
- 0~9 까지 id매칭
![edab9ba1d45e19d811502a773e142d77f7339b6e_re_1674143258383](https://user-images.githubusercontent.com/105347300/214510426-c3bf5d05-b577-4b49-ad2e-0250734f5f47.png)
----------------------------------------------------------------
## :loudspeaker: TensorFlow 레코드 만들기
- 주석을 TFRecord형식으로 변환
- 폴더 의 모든 *.xml파일 을 반복 하고 두 파일 각각에 대한 파일을 생성하는 간단한 스크립트 사용
- 업로드된 generate_tfrecord.py사용
![45bc57d1d6c77e84d5299448ef01e7a88d65659a_re_1674143258384](https://user-images.githubusercontent.com/105347300/214510691-72cf5338-8d6f-42b2-b998-3929e057239f.png)

- Create train data:
- python generate_tfrecord.py -x [PATH_TO_IMAGES_FOLDER]/train -l [PATH_TO_ANNOTATIONS_FOLDER]/label_map.pbtxt -o [PATH_TO_ANNOTATIONS_FOLDER]/train.record

- Create test data:
- python generate_tfrecord.py -x [PATH_TO_IMAGES_FOLDER]/test -l [PATH_TO_ANNOTATIONS_FOLDER]/label_map.pbtxt -o [PATH_TO_ANNOTATIONS_FOLDER]/test.record

- 경로 넣고 명령어 입력하면 레코드 파일 생성됨! 

![b663e70f36eba093d9f9cab3af65d12bdc75f1a2_re_1674143258384](https://user-images.githubusercontent.com/105347300/214510897-1b6e6cdd-32b2-4c91-b874-7aa0d0a3296f.png)
- test.xml파일 용량이 0KB이면 test.record는 빈 파일임
- 모델 평가를 하기위해서는 test.xml파일을 만들어야함.
- 저자는 추후 모델 평가시 test폴더에서 labelimg를 사용하여 xml파일 추가후 test.record파일을 다시 생성해주었음.
----------------------------------------------------------------
## :loudspeaker: 사전 학습된 모델 SSD ResNet50 V1 FPN 640x640  다운, 교육 파이이프라인 수정
- 다운로드 받은 모델 폴더 속의 pipeline.config 파일 수정!
- 위의 폴더에서 my_ssd_resnet50_v1_fpn_분할압축 폴더임. 분할압축 되어있음.
- [7dd78918967b0f23af4c599a607e76a902b31b96_re_1674143258384](https://user-images.githubusercontent.com/105347300/214511647-1c68d1c4-e770-43b1-b3b6-c648124a3a86.png)
- 클래스 10개 이므로 수정함

![eccc55683d056621e0e152f1b74fbeba24201125_re_1674143258384](https://user-images.githubusercontent.com/105347300/214511705-f8f4dcaa-d3aa-4ec3-b394-d421d305fd8b.png)
-내가 사용하는 gpu메모리 최대 6GB이므로 4로 수정함

![a24beca45a892cb1a4f2402342a9c5103f52c7a6_re_1674143258385](https://user-images.githubusercontent.com/105347300/214511741-3423cc78-d25e-41e2-88f0-5777c15db814.png)
- 파일 경로를 넣어줌 , num_steps 수정가능
----------------------------------------------------------------
## :loudspeaker:모델 훈련
- TensorBoard를 사용하여 교육 작업 진행률 모니터링
![31e6f58050a88d3c61431b312578e57063343958_re_1674143258383](https://user-images.githubusercontent.com/105347300/214511940-f77965ad-88ea-4ff1-ad39-6c64c4e6d71c.png)

#### tensor board 해석
- classification_loss (분류 손실):감지된 객체를 다양한 클래스로 분류하기 위한 손실(클래스를 얼마나 잘 예측했는지에 대한 loss)
- localization_loss:바운딩 박스를 얼마나 잘 예측했는지에 대한 손실(지역화 손실 또는 경계 상자 회귀자의 손실)
- regularization_loss(정규화 손실) :  정규화 손실은 신경망 의 가중치 에서 계산된 L2 손실과 같음. 이 손실을 최소화하면 가중치 값이 축소되는 경향이 있음. 과적합과 같은 문제를 해결하는 데 도움이 될 수 있는 정규화기술
- learning_rate: 학습률

### - :boom: 25000epochs 훈련에 걸린 시간
### - :boom: 2023/01/17 약 16시 ~  2023/01/19 02시 약 34 시간 19분 소요 ==> tensorboard 1.43day 소요
![cfed34d191623197d165aac2e594f85e8fcfe114](https://user-images.githubusercontent.com/105347300/214512276-ec1fffaa-0bdd-484b-bcc4-df30f682423c.png)
----------------------------------------------------------------
## :loudspeaker:모델 평가
- 기본적으로 교육 프로세스는 교육 성과의 몇 가지 기본 측정값을 기록
- 과정 :모델 학습 중 체크포인트 파일 세트가 생성되면 평가 프로세스는 이러한 파일을 사용하고 모델이 테스트 데이터 세트에서 개체를 얼마나 잘 감지하는지 평가
- ==> 이 평가의 결과는 시간이 지남에 따라 검사할 수 있는 몇 가지 메트릭 형식으로 요약됨
----------------------------------------------------------------
### 평가 완료
![fde66980e89fca3f9a14a0849a0f7ccbcbfcc878_re_1674143258385](https://user-images.githubusercontent.com/105347300/214512874-caf8b167-bfe2-40ad-b84b-7129b70e796e.png)
----------------------------------------------------------------
### 평가 해석
![7d8dabacd6d775fd74d721e897f6bfaab23084be_re_1674143258385](https://user-images.githubusercontent.com/105347300/214512975-6e838854-d94a-4118-b066-6ae940a58824.png)
- Precision (정밀도) :Precision은 모든 검출 결과 중 옳게 검출한 비율
- Recall (재현율) :Recall은 마땅히 검출해내야 하는 물체들 중에서 제대로 검출된 비율

- Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.677
- => 0.5 와 0.95사이의 IoU(교집합에 대한 교집합) 임계값에 대한 평균 정밀도 및 평균 재현율이며 계산되는 최대 감지 수는 100임.또한 다양한 영역(전체, 소형, 중형 및 대형)에 대한 평균 정밀도(AP) 및 AR을 계산했습니다.

![f994f8fe501db754bd0843e150aee4f86309e555_re_1674143258385](https://user-images.githubusercontent.com/105347300/214513073-c04d80bb-9772-43e3-a3be-b72893f6d2df.png)

- IoU = 교집합 영역 넓이 / 합집합 영역 넓이
- 'DetectionBoxes_Precision/mAP': .05 단위로 .5에서 .95 범위의 IOU 임계값에 대해 평균화된 클래스에 대한 평균 정밀도를 의미합니다.
- 'DetectionBoxes_Precision/mAP@.50IOU': 50% IOU에서 평균 평균 정밀도
- 'DetectionBoxes_Precision/mAP@.75IOU': 75% IOU에서 평균 평균 정밀도
- 'DetectionBoxes_Precision/mAP (소형)': 작은 물체에 대한 평균 정밀도를 의미합니다(면적 < 32^2 픽셀).
- 'DetectionBoxes_Precision/mAP (medium)': 중간 크기 개체의 평균 정밀도를 의미합니다(32^2 픽셀 < 영역 < 96^2 픽셀).
- 'DetectionBoxes_Precision/mAP (대형)': 큰 개체에 대한 평균 정밀도를 의미합니다(96^2 픽셀 < 영역 < 10000^2 픽셀).
- 'DetectionBoxes_Recall/AR@1': 1회 감지된 평균 회수.
- 'DetectionBoxes_Recall/AR@10': 10회 감지된 평균 회수.
- 'DetectionBoxes_Recall/AR@100': 100개 감지의 평균 회수.
- 'DetectionBoxes_Recall/AR@100 (소형)': 작은 물체에 대한 평균 회수율이 100입니다.
- 'DetectionBoxes_Recall/AR@100 (medium)': 100인 중간 물체에 대한 평균 회수.
- 'DetectionBoxes_Recall/AR@100 (large)': 100회 감지된 큰 물체에 대한 평균 회수.

![20bd5ee03d96f8d0b3a70f6b4fb77200055f1700_re_1674143258385](https://user-images.githubusercontent.com/105347300/214513648-7c9868b7-b13c-47a0-90ed-c3e5502e84d2.png)
- classification_loss (분류 손실):감지된 객체를 다양한 클래스로 분류하기 위한 손실(클래스를 얼마나 잘 예측했는지에 대한 loss)
- localization_loss:바운딩 박스를 얼마나 잘 예측했는지에 대한 손실(지역화 손실 또는 경계 상자 회귀자의 손실)
- regularization_loss(정규화 손실) :  정규화 손실은 신경망 의 가중치 에서 계산된 L2 손실과 같음. 이 손실을 최소화하면 가중치 값이 축소되는 경향이 있습니다. 과적합과 같은 문제를 해결하는 데 도움이 될 수 있는 정규화기술
- learning_rate: 학습률
----------------------------------------------------------------
#### Resolving errors during evaluation  :v:
![f3aefbabb5bd36364b57b56dd2e54fd9181ba7dd_re_1674143258385](https://user-images.githubusercontent.com/105347300/214513742-67393059-4f35-4cf9-a4e5-141396f42d41.png)
- =>해결 : test.record파일 생성


![25a72fdffb0cef87c7815ea1b6993abf3b00c5a5_re_1673935912240](https://user-images.githubusercontent.com/105347300/214513813-bce2161c-0a83-4f84-851f-bb76bdb36c08.png)
InvalidArgumentError: TypeError: 'numpy.float64' 개체를 정수로 해석할 수 없습니다. #2961
- ==> 해결 : 파일 수정


## :loudspeaker:모델 내보내기
-아나콘다 터미널
- python .\exporter_main_v2.py --input_type image_tensor --pipeline_config_path .\models\my_ssd_resnet50_v1_fpn\pipeline.config --trained_checkpoint_dir .\models\my_ssd_resnet50_v1_fpn\ --output_directory .\exported-models\my_model

![6df90c00670a10a34394437b6e46f94ffe785cd1_re_1674143258385](https://user-images.githubusercontent.com/105347300/214514161-bb753c6e-6cf5-4b1e-9a5e-9b50499c0156.png)

## :loudspeaker: Object Detection From TF2 Saved Model(TF2 저장된 모델에서 개체 감지) :boom:

- epochs당 다른 loss값을 가진 저장된 모델들을 같은 조건에서 비교해보며 공부
- my_model_ep25000 파일이 완성된 모델 ep300,2500은 비교해 공부하기위한 목적의 모델임

## :bulb: 모델 비교 (epochs =25000인 모델) 
- 10개의 테스트 이미지 사용
- 위에서 훈련한 모델 사용
- epochs =25000
- 가장 성능이 좋은 모델임

![73807c832a82f390d06462f9f68486f11ae2f9c6_re_1674143258386](https://user-images.githubusercontent.com/105347300/214515550-8aefc88c-9790-4c32-9902-f08143f864e0.png)

![7bdfe0a10dd2d5af57a75d6432d0da8ed54373e7_re_1674143258386](https://user-images.githubusercontent.com/105347300/214515553-3d2d3f91-08f6-445e-bec0-d131a18b17d5.png)

### :boom: ==>같은 조건에서 100개의 숫자중 4개 틀림 == > 검출률 96%

----------------------------------------------------------------------------------
## :bulb: 모델 비교 (epochs =2500인 모델) 
- 아래 그래프에서 손실이 높아지기 전의 2500epoch모델임
![3ed8dff84d20d506024939a7a06502eb96e940d3](https://user-images.githubusercontent.com/105347300/214515879-9ae65a7e-275a-4699-990a-004c8ad58a67.png)

- ==>모델파일에서 시간에따른 ckpt-num를 삭제하고 cketpoint파일을 수정한 뒤 원하는 시간대까지 저장된 모델을 내보내면 됨.
![9a2ce67d5fc889c0396a1f278df85bc0a2176194_re_1674121259216](https://user-images.githubusercontent.com/105347300/214515887-bfefe66d-02b8-440c-84e1-e81744bd882c.png)

- 검출 결과
![08e47c3daf3a95c20da757ac75bc59b70f9d4c3d_re_1674143258386](https://user-images.githubusercontent.com/105347300/214515966-709676d1-a5e3-4bdf-9826-8bf4fb5ad186.png)

#### :boom: ==>같은 조건에서 100개의 숫자중 13개 틀림 & 검출하지 못함 == > 검출률 87%
-------------------------------------------------------------------------------------
## :bulb: 모델 비교 (epochs =3300인 모델) 
- 훈련 중 특히 오차가 높아졌을때의 모델

![7bdc52ec04219fb4ab4358753471e4035c441bdf](https://user-images.githubusercontent.com/105347300/214516172-74f4b343-bcb8-4fbe-8982-7e9199783182.png)
- 오차가 증가했지만 25000epoch에서 봤을때 일반적인 현상 이었음.

![ad298f4dcc64fd1600b8efc0049abcdc885bfe22_re_1674143258386](https://user-images.githubusercontent.com/105347300/214516263-bd6eaf90-bfe5-42ff-abc3-2f2e247889f3.png)

### :boom: ==>같은 조건에서 100개의 숫자중 100개 틀림 & 검출하지 못함 == > 검출률 0%

- ==>이유
![2a6e47014146807e044a48cdc531057a9c16a028_re_1674143258386](https://user-images.githubusercontent.com/105347300/214516355-37ee9c33-fcef-4d8a-bc60-eafecc3f2718.png)
- 객체의 정확도가 30%를 넘지 못함
- ==>조건에서 50%이상 확률을 가지는 객체만 그리도록 설정했기 때문에 그려지지 않았음.

--------------------------------------------------------------------------------------------------
## :bulb: epochs당 다른 loss값을 가진 저장된 모델들을 같은 조건에서 비교 결과

- 25000epochs 의 가장 작은 loss값을 가진 모델의 성능이 확실히 좋음. 
- 그러나 시간이 많이 걸린 단점  약 34 시간 19분 소요 ==> tensorboard 1.43day 소요
- 배경이 없는 객체들만 학습 시키다보니 배경이 있으면 확실히 인식을 못함

---------------------------------------------------------------------------------------------------
###  :grey_question: 검출된 객체의 id와 확률 표시 중 문제점

![5f1ab1d2721a32d666801f8a9915bca1f39be40a_re_1674143258386](https://user-images.githubusercontent.com/105347300/214516696-f7e09562-8a0c-4f18-82ae-0fa8204c69b1.png)
![5a1d29a95993b45a0331e7418ebe232e698e4071_re_1674143258386](https://user-images.githubusercontent.com/105347300/214516706-65b2ebf2-a699-48ec-88d8-7b9ef6bb51b2.png)
- ==>한 객체의 바운딩 박스가 겹쳤을때(파란 선으로 그은 두 바운딩 박스 좌표 같음) 더 높은 확률의 숫자가 랜덤으로 안보이는 경향이 있음
- ==>출력 글자를 확인해야하는 번거로움 발생
- 해결 못함
------------------------------------------------------------------------------------------------------
### :v: Resolving Code Errors  :v:

- 1.vscode matplotlib사용시 imshow 오류
- ==>해결 : 파일 코드 주석 처리

![629462bd14cce31fd5d1b3e15d54b4c2b54634e0_re_1673935912240](https://user-images.githubusercontent.com/105347300/214517311-b992530b-e0f3-487d-9217-46543f403f66.png)
![Uploading 65ebe310a07650b61cee8135c57eda33ffb16903_re_1673935912240.png…]()


- 2.바운딩 박스 글자 크기 조절 문제 
- 바운딩 박스 크기가 너무 커서 다른 바운딩 박스를 가리는 문제점
- D:\jyn\ncslab\tensorflow\models\research\object_detection\utils
- 파일 font 크기 수정 

![47b9925ae87dd11305cbac7949dce6378a532935_re_1674121259217](https://user-images.githubusercontent.com/105347300/214517401-7ca0c708-5daa-4de7-8f3e-68d66366d10c.png)
- ==>해결 : 기본 24에서 유동성있게 하기위해 이미지 가로 크기에 따른 크기 조절로 변경함. 

![956efa8464fe98ad374a4c91d46d410adaf6f720_re_1674121259217](https://user-images.githubusercontent.com/105347300/214517458-064d2418-26d1-4a72-9bb8-9d2fbb91e146.png)


# :loudspeaker:Code Addition Description

### :bulb: 1)저장된 모델 불러오기 및 검출 기능

![28301ea01da40f2a7cbab1fe0d6ffb0999113001](https://user-images.githubusercontent.com/105347300/214517699-a3d26f5a-ae1d-4d1b-89c4-95ffe6887501.png)
![0b7a92a1531e22b1a0e584dc34ddf8f7132e2c7e](https://user-images.githubusercontent.com/105347300/214517717-d2bda4f5-b36e-4558-ac08-7be4acd25af4.png)
![47abfe908c2911fda71d84ff1ac08ebd28b84094](https://user-images.githubusercontent.com/105347300/214517729-be1756da-d1b4-4e5d-b1c0-4d72d8e6ed18.png)

- =>모델을 가져와 데이터를 입력하면(학습한 데이터의 형식이 tensor이므로 입력 형식을 tensor로 바꾸어줌) 추론한 값을 반환함.

-----------------------------------------------------------------------------------------------------

### :bulb: 2)반환된 detections 의 값 출력
- ==>'raw_detection_boxes' ,num_detections , 'raw_detection_scores' ,'detection_anchor_indices' , 'detection_boxes' ,'detection_scores' ,'detection_multiclass_scores' 의 정보를 확인 할 수 있음.

![0d71a96e5e0477d35e9bd93609b5b4c02616a72b](https://user-images.githubusercontent.com/105347300/214517910-2d03cb22-9e8d-4201-963d-efcce0e1edd2.png)

- ==>num_detections 의 값만 정수임을 확인 할 수 있음. 100임
- ==>num_detections 를 제거하여 배열인 값만 남기고  반환된 추론값의 정보들을 딕셔너리로 만든 후 다시 num_detections의 값을 넣어줌
- ==> 딕셔너리 형태로 추론 값 사용
![c6f8b4485cc209b05c1953a62c2441b9d60ceb99](https://user-images.githubusercontent.com/105347300/214518008-7d9110cc-1a66-4e32-9370-11803bbfaf32.png)

----------------------------------------------------------------------------------------------

### :bulb: 3)반환된 추론값의 딕셔너리 key 설명

![ef233e76211fe390c5b04cbf7b4f8032d429b5d9](https://user-images.githubusercontent.com/105347300/214518102-aa91a502-b727-4fe3-87f4-92b0598646ce.png)
- ==>이미지 추론 정보 출력 가능

![4ceab178f3ce2b3701a1caa1ce7fe6a9a80ddd10_re_1674143258387](https://user-images.githubusercontent.com/105347300/214518153-97737792-9ad7-480d-b626-780f8824d3d4.png)

![e8026ffe0173a86184aec695f6f94d8c8a29b277](https://user-images.githubusercontent.com/105347300/214518174-2f1ea17e-10ea-4e9d-b947-4fbae4f5004f.png)


-----------------------------------------------------------------------------------------
### :bulb: 4)검출 결과를 원본 영상에 그리기 (함수 이용)

    viz_utils.visualize_boxes_and_labels_on_image_array(

        image_np_with_detections,# 원본 이미지

        detections['detection_boxes'],# bounding box 좌표

        detections['detection_classes'],  # label 클래스라벨

        detections['detection_scores'],# confidence score 확률

        category_index,# label map # 모델들의 카테고리에 대한 정보

        use_normalized_coordinates=True,# bounding box 좌표 normalized 여부

        max_boxes_to_draw=11, # 이미지 위에 최대 몇 개의 bounding box 그릴 지 설정

        min_score_thresh=.50, # confidence score가 지정한 값 이상인 것만 표시

        agnostic_mode=False, #평가 여부를 제어(객체만 검출)

        line_thickness=1#바운딩 박스 두께 설정

        )

- 함수 파라미터 정보
![9d594380218ffe05443ec9d5be38e9faef39ea0e_re_1674121259217](https://user-images.githubusercontent.com/105347300/214518982-fc9f94dd-389d-452b-9c52-2e97b8298255.png)

-----------------------------------------------------------------------------------------------







