import os  # 운영체제(파일, 폴더 등) 기능을 위한 라이브러리
import random  # 무작위 선택을 위한 라이브러리
import subprocess  # 다른 프로그램을 실행하기 위한 라이브러리
import platform  # 운영체제 정보를 알아내기 위한 라이브러리
from ultralytics import YOLO  # 인공지능 모델인 YOLO를 사용하기 위한 라이브러리

# 'val' 폴더에서 무작위 이미지를 찾는 함수
def get_random_test_image_path(base_dir='./val'):
    """
    지정된 폴더(base_dir) 안에서 무작위로 하나의 이미지 파일을 찾아 그 경로를 반환합니다.
    예시: './val/rock/rock_123.jpg'
    """
    # 1. base_dir 안의 모든 폴더 목록을 가져옵니다. (rock, paper, scissors)
    class_folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
    if not class_folders:
        # 폴더가 없으면 함수를 종료하고 None을 반환합니다.
        return None
    
    # 2. 클래스 폴더(예: rock, paper, scissors) 중 하나를 무작위로 선택합니다.
    random_class = random.choice(class_folders)
    random_class_path = os.path.join(base_dir, random_class)

    # 3. 선택된 폴더 안의 모든 이미지 파일 목록을 가져옵니다.
    images = [f for f in os.listdir(random_class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not images:
        # 이미지가 없으면 None을 반환합니다.
        return None
    
    # 4. 이미지 파일 중 하나를 무작위로 선택합니다.
    random_image = random.choice(images)
    
    # 5. 최종 이미지의 전체 경로를 만들어 반환합니다.
    return os.path.join(random_class_path, random_image)

# 이미지를 자동으로 여는 함수
def open_image(path):
    """
    운영체제에 따라 시스템의 기본 이미지 뷰어로 이미지를 엽니다.
    """
    # 현재 운영체제가 무엇인지 확인합니다.
    if platform.system() == 'Darwin':       # macOS일 경우
        subprocess.run(['open', path])
    elif platform.system() == 'Windows':    # Windows일 경우
        os.startfile(path)
    else:                                   # 리눅스 등 다른 운영체제일 경우
        subprocess.run(['xdg-open', path])

#==============================================================
#  메인 프로그램 시작
#==============================================================

# 1. 예측
# 학습된 모델을 불러옵니다. 이 경로는 'runs' 폴더 안의 'best.pt' 파일입니다.
# 님 컴퓨터의 정확한 경로에 맞게 이 부분을 수정해야 합니다.
trained_model = YOLO('./runs/classify/train/weights/best.pt')

# 2. 무작위 이미지 경로 가져오기
test_image_path = get_random_test_image_path()

# 3. 만약 테스트할 이미지를 찾았다면...
if test_image_path:
    # 어떤 이미지를 테스트할지 터미널에 출력합니다.
    print(f"테스트할 무작위 이미지: {test_image_path}")
    # 그리고 그 이미지를 자동으로 엽니다.
    open_image(test_image_path)
    
    # 4. 예측 수행
    # 불러온 모델을 사용해 이미지에 대한 예측을 수행합니다.
    results = trained_model(test_image_path)

    # 5. 결과 분석 및 출력
    # 예측 결과(results)에서 가장 확률이 높은 값들을 가져옵니다.
    for result in results:
        probs = result.probs  # 예측 확률 정보
        class_id = probs.top1  # 가장 높은 확률을 가진 클래스의 ID (0, 1, 2 중 하나)
        confidence = probs.top1conf  # 가장 높은 확률 값

        # 클래스 ID에 해당하는 이름을 가져옵니다.
        class_name = trained_model.names[class_id]

        # 최종 예측 결과를 터미널에 출력합니다.
        print(f"예측 결과: {class_name} (확률: {confidence:.2f})")
else:
    # 테스트할 이미지를 찾지 못했을 경우 메시지 출력
    print("테스트할 이미지를 찾을 수 없습니다. 'val' 폴더와 하위 폴더의 구조를 확인해주세요.")