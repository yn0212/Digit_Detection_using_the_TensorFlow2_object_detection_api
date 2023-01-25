# Digit Detection :zero:~:nine: :computer:
## using the TensorFlow2 object detection api 
![Footer](https://capsule-render.vercel.app/api?type=waving&color=auto&height=200&section=footer)

# :pushpin:Project Description
Anaconda 환경에서 TensorFlow 2 Object Detection API 를 사용하여 필기체 이미지를 95% 확률로 검출을 수행하는 프로그램이다.

# :pushpin:Project Purpose
- 전이학습된 모델 SSD ResNet50 V1 FPN 640x640을 숫자 검출을 위한 모델로 훈련시켜 숫자 검출률 95%이상 높이기
- tensorflow2 object detection api를 응용하여 숫자 검출 과제에 맞게 수정하여 사용해보기 위함
- epoch별 모델 성능을 비교해보고 모델 평가를 분석해보며 공부한 내용 복습

# :pushpin:Device used for the project
- 학습에 사용한 GPU : GTX1660 Supper

# :pushpin:Project Dataset
- 사용한 데이터셋 :  google 이미지 다운 (train데이터 : 0~9 xml파일 약 18개 , 예측 데이터 : 10개)
- 객체 클래스 수 : 0~9 총 10개

# :pushpin:Project Results
- 필기체 인식률 : 96%
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

### :loudspeaker: 데이터세트에 주석 달기 
- LabelImg  사용
![76d36c161bc6bfbed6b5216a47a0160afd69ad0c_re_1674143258383](https://user-images.githubusercontent.com/105347300/214510142-cd76c0b2-36d6-487b-816b-2f6cb15cdf91.png)
- 0~9 까지의 훈련 데이터 객체에 주석을 달아 xml파일 생성

### :loudspeaker: Create Label Map (레이블 맵 생성)
- TensorFlow에는 사용된 각 레이블을 정수 값에 매핑하는 레이블 맵이 필요함.
- label_map.pbtxt파일 사용
![889fb11ec81fa699e1f749f4b5989b3ba926ee55_re_1674143258384](https://user-images.githubusercontent.com/105347300/214510342-9c46f47b-59a7-4c59-a26c-9ed3c9679318.png)
- id는 1부터 시작
- 0~9 까지 id매칭
![edab9ba1d45e19d811502a773e142d77f7339b6e_re_1674143258383](https://user-images.githubusercontent.com/105347300/214510426-c3bf5d05-b577-4b49-ad2e-0250734f5f47.png)

### :loudspeaker: TensorFlow 레코드 만들기
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

## :loudspeaker: 사전 학습된 모델 SSD ResNet50 V1 FPN 640x640  다운, 교육 파이이프라인 수정
- 다운로드 받은 모델 폴더 속의 pipeline.config 파일 수정!
- [7dd78918967b0f23af4c599a607e76a902b31b96_re_1674143258384](https://user-images.githubusercontent.com/105347300/214511647-1c68d1c4-e770-43b1-b3b6-c648124a3a86.png)
- 클래스 10개 이므로 수정함

![eccc55683d056621e0e152f1b74fbeba24201125_re_1674143258384](https://user-images.githubusercontent.com/105347300/214511705-f8f4dcaa-d3aa-4ec3-b394-d421d305fd8b.png)
-내가 사용하는 gpu메모리 최대 6GB이므로 4로 수정함

![a24beca45a892cb1a4f2402342a9c5103f52c7a6_re_1674143258385](https://user-images.githubusercontent.com/105347300/214511741-3423cc78-d25e-41e2-88f0-5777c15db814.png)
- 파일 경로를 넣어줌 , num_steps 수정가능

## :loudspeaker:모델 훈련
- TensorBoard를 사용하여 교육 작업 진행률 모니터링
![31e6f58050a88d3c61431b312578e57063343958_re_1674143258383](https://user-images.githubusercontent.com/105347300/214511940-f77965ad-88ea-4ff1-ad39-6c64c4e6d71c.png)

##### tensor board 해석
- classification_loss (분류 손실):감지된 객체를 다양한 클래스로 분류하기 위한 손실(클래스를 얼마나 잘 예측했는지에 대한 loss)
- localization_loss:바운딩 박스를 얼마나 잘 예측했는지에 대한 손실(지역화 손실 또는 경계 상자 회귀자의 손실)
- regularization_loss(정규화 손실) :  정규화 손실은 신경망 의 가중치 에서 계산된 L2 손실과 같음. 이 손실을 최소화하면 가중치 값이 축소되는 경향이 있음. 과적합과 같은 문제를 해결하는 데 도움이 될 수 있는 정규화기술
- learning_rate: 학습률

- :boom: 25000epochs 훈련에 걸린 시간
- :boom: 2023/01/17 약 16시 ~  2023/01/19 02시 약 34 시간 19분 소요 ==> tensorboard 1.43day 소요
![cfed34d191623197d165aac2e594f85e8fcfe114](https://user-images.githubusercontent.com/105347300/214512276-ec1fffaa-0bdd-484b-bcc4-df30f682423c.png)

## :loudspeaker:모델 평가
- 기본적으로 교육 프로세스는 교육 성과의 몇 가지 기본 측정값을 기록
- 과정 :모델 학습 중 체크포인트 파일 세트가 생성되면 평가 프로세스는 이러한 파일을 사용하고 모델이 테스트 데이터 세트에서 개체를 얼마나 잘 감지하는지 평가
- ==> 이 평가의 결과는 시간이 지남에 따라 검사할 수 있는 몇 가지 메트릭 형식으로 요약됨

#### 평가 완료
![fde66980e89fca3f9a14a0849a0f7ccbcbfcc878_re_1674143258385](https://user-images.githubusercontent.com/105347300/214512874-caf8b167-bfe2-40ad-b84b-7129b70e796e.png)

#### 평가 해석
![7d8dabacd6d775fd74d721e897f6bfaab23084be_re_1674143258385](https://user-images.githubusercontent.com/105347300/214512975-6e838854-d94a-4118-b066-6ae940a58824.png)
- Precision (정밀도) :Precision은 모든 검출 결과 중 옳게 검출한 비율
- Recall (재현율) :Recall은 마땅히 검출해내야 하는 물체들 중에서 제대로 검출된 비율

- Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.677
- => 0.5 와 0.95사이의 IoU(교집합에 대한 교집합) 임계값에 대한 평균 정밀도 및 평균 재현율이며 계산되는 최대 감지 수는 100임.또한 다양한 영역(전체, 소형, 중형 및 대형)에 대한 평균 정밀도(AP) 및 AR을 계산했습니다.

![f994f8fe501db754bd0843e150aee4f86309e555_re_1674143258385](https://user-images.githubusercontent.com/105347300/214513073-c04d80bb-9772-43e3-a3be-b72893f6d2df.png)

IoU = 교집합 영역 넓이 / 합집합 영역 넓이
'DetectionBoxes_Precision/mAP': .05 단위로 .5에서 .95 범위의 IOU 임계값에 대해 평균화된 클래스에 대한 평균 정밀도를 의미합니다.
'DetectionBoxes_Precision/mAP@.50IOU': 50% IOU에서 평균 평균 정밀도
'DetectionBoxes_Precision/mAP@.75IOU': 75% IOU에서 평균 평균 정밀도
'DetectionBoxes_Precision/mAP (소형)': 작은 물체에 대한 평균 정밀도를 의미합니다(면적 < 32^2 픽셀).
'DetectionBoxes_Precision/mAP (medium)': 중간 크기 개체의 평균 정밀도를 의미합니다(32^2 픽셀 < 영역 < 96^2 픽셀).
'DetectionBoxes_Precision/mAP (대형)': 큰 개체에 대한 평균 정밀도를 의미합니다(96^2 픽셀 < 영역 < 10000^2 픽셀).
'DetectionBoxes_Recall/AR@1': 1회 감지된 평균 회수.
'DetectionBoxes_Recall/AR@10': 10회 감지된 평균 회수.
'DetectionBoxes_Recall/AR@100': 100개 감지의 평균 회수.
'DetectionBoxes_Recall/AR@100 (소형)': 작은 물체에 대한 평균 회수율이 100입니다.
'DetectionBoxes_Recall/AR@100 (medium)': 100인 중간 물체에 대한 평균 회수.
'DetectionBoxes_Recall/AR@100 (large)': 100회 감지된 큰 물체에 대한 평균 회수.

![20bd5ee03d96f8d0b3a70f6b4fb77200055f1700_re_1674143258385](https://user-images.githubusercontent.com/105347300/214513648-7c9868b7-b13c-47a0-90ed-c3e5502e84d2.png)
- classification_loss (분류 손실):감지된 객체를 다양한 클래스로 분류하기 위한 손실(클래스를 얼마나 잘 예측했는지에 대한 loss)
- localization_loss:바운딩 박스를 얼마나 잘 예측했는지에 대한 손실(지역화 손실 또는 경계 상자 회귀자의 손실)
- regularization_loss(정규화 손실) :  정규화 손실은 신경망 의 가중치 에서 계산된 L2 손실과 같음. 이 손실을 최소화하면 가중치 값이 축소되는 경향이 있습니다. 과적합과 같은 문제를 해결하는 데 도움이 될 수 있는 정규화기술
- learning_rate: 학습률

##### Resolved during evaluation  :v:
![f3aefbabb5bd36364b57b56dd2e54fd9181ba7dd_re_1674143258385](https://user-images.githubusercontent.com/105347300/214513742-67393059-4f35-4cf9-a4e5-141396f42d41.png)
- =>해결 : test.record파일 생성


![25a72fdffb0cef87c7815ea1b6993abf3b00c5a5_re_1673935912240](https://user-images.githubusercontent.com/105347300/214513813-bce2161c-0a83-4f84-851f-bb76bdb36c08.png)
InvalidArgumentError: TypeError: 'numpy.float64' 개체를 정수로 해석할 수 없습니다. #2961

- ==> 해결 : 파일 수정





