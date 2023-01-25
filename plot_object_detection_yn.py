"""
Object Detection From TF2 Saved Model
=====================================
"""


#테스트 이미지 다운로드
#base_url에 테스트 이미지 경로를 넣어 사용
import os #운영체제에서 제공되는 여러 기능을 파이썬에서 수행시켜주는 파이썬 라이브러리
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    #파이썬 텐서플로에서 INFO, WARNING 로그를 출력 억제
import tensorflow as tf

tf.get_logger().setLevel('ERROR')           #python tensorflow 로그 수준을 "ERROR"로 설정

# Enable GPU dynamic memory allocation
#GPU 동적 메모리 할당 활성화
gpus = tf.config.experimental.list_physical_devices('GPU') # 이 아래 모든 부분에 대해 GPU로 실행하도록
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True) # 메모리 증가를 활성화해야 하는지 여부를 설정

def download_images(): # test 이미지 불러오기 함수
    base_url = 'D:/jyn/ncslab/tensorflow/workspace/training_demo/images/test/'#테스트 이미지 경로 설정
    filenames = [] #테스트 이미지 파일 이름을 넣을 리스트 생성
    for i in range(1,11): 
        val=base_url+'num'+str(i)+'.bmp'#테스트 파일 이름, 확장자 설정
        filenames.append(val) #파일 이름 리스트에 넣기

    return filenames #리스트 반환

IMAGE_PATHS = download_images() #image path설정
"""
====================================
"""
# Download the model
# 저장된 모델 경로 설정 함수
#사용할 객체 검출 알고리즘 모델은 my_ssd_resnet50_v1_fpn
# Download and extract model``
def download_model(model_name): #모델 불러오기
    base_url = 'D:/jyn/ncslab/tensorflow/workspace/training_demo/exported-models/'#모델 경로
    model_dir =base_url + model_name
 
    return str(model_dir)

#MODEL_DATE = '20200711'
MODEL_NAME = 'my_model' #모델 이름
PATH_TO_MODEL_DIR = download_model(MODEL_NAME) #모델 path 설정

"""
====================================
"""
# Download the labels
# 레이블 파일 다운로드 함수
# Download labels file
def download_labels(filename): #레이블 파일 경로 설정, 불러오기
    base_url = 'D:/jyn/ncslab/tensorflow/workspace/training_demo/annotations/'

    label_dir = base_url+filename
    return str(label_dir)

LABEL_FILENAME = 'label_map.pbtxt' #레이블 파일 이름
PATH_TO_LABELS = download_labels(LABEL_FILENAME) #레이블 파일 path설정
"""
====================================
"""

# 모델 불러오기
# Next we load the downloaded model``
import time
PATH_TO_SAVED_MODEL = PATH_TO_MODEL_DIR + "/saved_model"

print('Loading model...', end='')
start_time = time.time() #모델 불러오기 시간 측정

# Load saved model and build the detection function
#저장된 모델 불러오기 및 검출 기능 구축 !!
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))#모델 불러오기 시간 측정
"""
====================================
"""
# 레이블 맵 데이터 로드
# Load label map data (for plotting)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from object_detection.utils import label_map_util
# 모델들의 카테고리에 대한 정보입니다.
#detection_classes의 값이 무엇인지 알려주기 위해 꼭 필요한 작업 
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)
"""
====================================
"""

# Putting everything together
# `detections['detection_boxes']`를 출력하고 상자 위치를 이미지의 상자와 일치시키십시오.
#  좌표는 정규화된 형식(즉, 간격 [0, 1])으로 제공됩니다.
# # min_score_thresh``를 다른 값(0에서 1 사이)으로 설정하여 더 많은 탐지를 허용하거나 더 많은 탐지를 걸러냅니다.
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import warnings
from object_detection.utils import visualization_utils as viz_utils
warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings

def load_image_into_numpy_array(path):
    """파일에서 numpy 배열로 이미지를 로드

    이미지를 numpy 배열에 넣어 tensorflow 그래프에 공급합니다.
    관례에 따라 모양이 있는 numpy 배열에 넣습니다.
    (높이, 너비, 채널), 여기서 채널은 RGB의 경우 3입니다.
    Args:
      path: 이미지의 파일 경로`
,
    Returns:
      모양이 있는 uint8 numpy 배열(img_height, img_width, 3)
    """

    return np.array(Image.open(path)) #이미지 파일에서 numpy 배열로 이미지를 로드

img_list=[]
for index , image_path in enumerate(IMAGE_PATHS) : 
    #이미지 정보 출력
    print('Running inference for {}... '.format(image_path), end='')
    image_np = load_image_into_numpy_array(image_path) #파일에서 numpy 배열로 이미지를 로드
    print('\n데이터형:'+str(image_np.dtype))
    print('행,열 ,채널 :'+str(image_np.shape))
    print('전체 원소 수 :'+str(image_np.size))

    # Things to try:
    # Flip horizontally
    # image_np = np.fliplr(image_np).copy()

    # Convert image to grayscale
    # image_np = np.tile(
    #     np.mean(image_np, 2, keepdims=True), (1, 1, 3)).astype(np.uint8)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    # 학습한 데이터 형식이 tensor이기 떄문에 형식을 바꿔줍니다.
    input_tensor = tf.convert_to_tensor(image_np)
    
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    #모델은 이미지 배치를 예상하므로 `tf.newaxis`로 축을 추가합니다.
    input_tensor = input_tensor[tf.newaxis, ...] #차원 추가하기

    # input_tensor = np.expand_dims(image_np, 0)

    # 모델을 가져와 예측 데이터를 입력해 이미지 추론!!
    detections = detect_fn(input_tensor)
    #print('추론값 원본',detections)
    # 결과로 나온 값들을 딕셔너리 형태로 바꿔줍니다.
    #num_detections 값만 정수이기 때문에, 
    # num_detections를 제거하지 않으면 이후 반복문 작업에서 오류가 발생 (나머지 값: 배열)
    num_detections = int(detections.pop('num_detections'))
    
    # num_detections은 100임
    # items()함수를 사용해 딕셔너리에 있는 키와 값들의 쌍을 key,value에 반환함
    #100(num_detections)개 box 정보만 추출한후에 tensor를 ndarray로 변환
    detections = {key: value[0, :num_detections].numpy()
                for key, value in detections.items()}
                
    # 앞서 제거했던 num_detections 값을 다시 넣어줌
    detections['num_detections'] = num_detections

    # 클래스들의 타입을 숫자형식으로 바꿔줌(추론 라벨 값 형변환 (실수 -> 정수))
    # detections['detection_classes'] 값들은 라벨인데, 라벨들이 실수로 되어 있기 때문에 정수형으로 바꿔줌 
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    
    #각 박스 안에 물체가 있을 확률 =>정렬됨
    #print('각 박스 안에 물체가 있을 확률: ',detections['detection_scores'][:10])
    #bounding box의 좌표
    #print(' bounding box의 좌표: ',detections['detection_boxes'])
    #print('확률 높은 순서의 결과 클래스 정렬:',detections['detection_classes'])

    # 테스트 이미지 배열의 텐서값
    image_np_with_detections = image_np.copy()
    
    #검출 결과를 원본 영상에 그리기 (함수 이용)
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

    img_list.append(image_np_with_detections) #객체 검출 그린 결과 리스트에 저장
    print('Done')
    for j in range(20):
        print(index,'번째 이미지 :',' 클래스 이름: ',detections['detection_classes'][j],' bounding box의 좌표: ',detections['detection_boxes'][j])
        print('확률:',float(detections['detection_scores'][j]))

fig = plt.figure(figsize=(20,10)) #출력창 크기 조절
#서브 플롯간의 간격 조절
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
 
for i in range(10):
    #서브 플롯 추가 x행x열
    ax = fig.add_subplot(2, 5, i+1, xticks=[], yticks=[])
    ax.imshow(img_list[i]) #서브 플롯에 이미지 불러오기
plt.show()


    


# sphinx_gallery_thumbnail_number = 2