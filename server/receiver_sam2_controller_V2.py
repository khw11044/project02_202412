import socket
import time
import torch
import numpy as np
import cv2
import requests
import threading
from ultralytics import YOLO
from sam2.build_sam import build_sam2_camera_predictor

# 라즈베리파이 서버의 스트리밍 URL
url = "http://192.168.0.127:5000/video_feed"  # Flask 서버의 /video_feed URL

# 모바일 로봇의 IP 주소와 포트 설정
ROBOT_IP = "192.168.0.123"  # 모바일 로봇의 IP 주소 (변경 필요)
ROBOT_PORT = 5000  # 모바일 로봇의 수신 포트

# 초기화
x, y = 127, 127
blue, green, red, yellow, left_top, right_top = 0, 0, 0, 0, 0, 0
select_key, start_key = 0, 0

last_data = None
last_sent_time = 0
send_interval = 0.05  # 최소 전송 간격 (초)
distance = 0  # 중심 간 거리
changed=True
stay=False

# 로봇 제어 함수
def control_robot(dist, sock):
    global x, y
    global blue, green, red, yellow, left_top, right_top, select_key, start_key
    global last_data, last_sent_time

    prev_x = x  # 이전 x 값 저장
    if dist < -140:
        print('좌회전')
        x = -1
    elif dist > 140:
        print('우회전')
        x = 1
    else:
        print('그대로')
        x = 127

    # 변경된 값이 있거나 송신 주기가 지난 경우에만 데이터 전송
    if x != prev_x or time.time() - last_sent_time >= send_interval:
        data = f"{x},{y},{blue},{green},{red},{yellow},{left_top},{right_top},{select_key},{start_key},{dist}"

        # 동일 데이터 중복 전송 방지
        if data != last_data:
            sock.sendto(data.encode(), (ROBOT_IP, ROBOT_PORT))
            last_data = data
            last_sent_time = time.time()

            # 디버깅 출력
            print("전송 데이터:", data)

# 로봇 제어를 별도 스레드에서 실행
def robot_control_thread(sock):
    global distance
    while True:
        if largest_bbox:  # 바운딩 박스가 있을 때만 제어
            control_robot(distance, sock)


# 스트리밍 연결
stream = requests.get(url, stream=True, timeout=10)
if stream.status_code != 200:
    print(f"Streaming 연결 실패: 상태 코드 {stream.status_code}")
    exit()

# YOLO 모델 로드
yolo_model = YOLO("./models/yolo11n.pt")  # YOLO 모델 경로

# SAM2 모델 로드
sam2_checkpoint = "./models/sam2_hiera_small.pt"
model_cfg = "sam2_hiera_s.yaml"
predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint)

# PyTorch 설정
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# 소켓 생성
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 65535)  # 송신 버퍼 크기 설정

# 로봇 제어 스레드 시작
robot_thread = threading.Thread(target=robot_control_thread, args=(sock,))
robot_thread.daemon = True  # 메인 스레드 종료 시 함께 종료
robot_thread.start()

# 초기화 변수
if_init = False
largest_bbox = None  # 가장 큰 바운딩 박스를 저장
byte_data = b""  # 스트리밍 데이터를 저장할 바이트 버퍼
frame_counter = 0  # 프레임 카운터
previous_mask = None  # 이전 마스크 저장
previous_bbox = []  # 이전 바운딩 박스 저장

color = (0, 0, 150)  # 빨간색 (B, G, R)
thickness = 2  # 선 두께
frame_cnt = 0
for chunk in stream.iter_content(chunk_size=1024):  # 1KB 단위로 데이터 읽기
    byte_data += chunk
    a = byte_data.find(b'\xff\xd8')  # JPEG 시작 부분
    b = byte_data.find(b'\xff\xd9', a)  # JPEG 끝 부분
    if a != -1 and b != -1:  # JPEG 이미지의 시작과 끝이 존재할 때
        jpg = byte_data[a:b+2]  # JPEG 이미지 추출
        byte_data = byte_data[b+2:]  # 읽은 데이터 버퍼에서 제거

        # JPEG 데이터를 OpenCV 이미지로 디코딩
        frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
        frame = cv2.flip(frame, 1)
        copy_frame = frame.copy()
        width, height = frame.shape[:2][::-1]
        # 중심점 계산
        center_x, center_y = width // 2, height // 2

        # 빨간 십자가 그리기
        cv2.line(frame, (center_x, 0), (center_x, height), color, thickness)  # 세로선
        cv2.line(frame, (0, center_y), (width, center_y), color, thickness)  # 가로선
        

        if not largest_bbox:
            # YOLO로 사람 탐지
            results = yolo_model.predict(frame, conf=0.5, classes=[0])
            
            largest_area = 0
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # 바운딩 박스 좌표
                    area = (x2 - x1) * (y2 - y1)  # 바운딩 박스 면적 계산
                    if area > largest_area:  # 가장 큰 영역 선택
                        largest_area = area
                        largest_bbox = (x1, y1, x2, y2)

            # 가장 큰 사람 바운딩 박스를 그리기
            if largest_bbox:
                cv2.rectangle(frame, (largest_bbox[0], largest_bbox[1]), 
                              (largest_bbox[2], largest_bbox[3]), (0, 255, 0), 2)

        # SAM2를 사용하여 세그멘테이션 및 트래킹
        if largest_bbox and not if_init:
            predictor.load_first_frame(frame)
            bbox = np.array([[largest_bbox[0], largest_bbox[1]],
                             [largest_bbox[2], largest_bbox[3]]], dtype=np.float32)
            _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(
                frame_idx=0, obj_id=1, bbox=bbox
            )
            if_init = True

            # 소켓 생성 및 버퍼 크기 설정
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 65535)  # 송신 버퍼 크기 설정

            
        elif if_init:
            # 2프레임마다 트래킹 실행
            if frame_counter % 5 == 0:
                out_obj_ids, out_mask_logits = predictor.track(frame)
                all_mask = np.zeros((height, width, 1), dtype=np.uint8)
                current_bbox = []  # 현재 프레임의 바운딩 박스

                for i in range(len(out_obj_ids)):
                    out_mask = (out_mask_logits[i] > 0.0).permute(1, 2, 0).byte().cuda()
                    all_mask = cv2.bitwise_or(all_mask, out_mask.cpu().numpy() * 255)

                    # 마스크에서 바운딩 박스 추출
                    mask_binary = (out_mask.cpu().numpy() > 0).astype(np.uint8)
                    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for contour in contours:
                        x, y, w, h = cv2.boundingRect(contour)
                        current_bbox.append((x, y, x + w, y + h))  # 바운딩 박스 저장

                previous_mask = all_mask  # 현재 마스크 저장
                previous_bbox = current_bbox  # 현재 바운딩 박스 저장
            else:
                all_mask = previous_mask  # 이전 마스크 사용
                current_bbox = previous_bbox  # 이전 바운딩 박스 사용

            # 마스크 적용
            if all_mask is not None:
                all_mask = cv2.cvtColor(all_mask, cv2.COLOR_GRAY2RGB)
                frame = cv2.addWeighted(frame, 1, all_mask, 0.5, 0)

            # 바운딩 박스 그리기
            for (x1, y1, x2, y2) in current_bbox:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # 바운딩 박스 중심 계산
            box_center_x = (x1 + x2) // 2
            box_center_y = (y1 + y2) // 2

            # 화면 중심과 바운딩 박스 중심을 잇는 선 그리기
            cv2.line(frame, (center_x, center_y), (box_center_x, box_center_y), (255, 0, 0), 2)
            distance = ((center_x - box_center_x) / 100)**2
            print(distance)     # 만약 150 기준 

            control_robot(distance, sock)

            frame_counter += 1  # 프레임 카운터 증가

        # OpenCV로 이미지 표시
        cv2.imshow("Camera", frame)

        cv2.imwrite(f'frames/{frame_cnt:03d}.jpg', copy_frame)
        frame_cnt += 1
        # 'q'를 누르면 종료
        if cv2.waitKey(10) & 0xFF == ord("q"):  # 10ms 대기
            break

# 리소스 해제
cv2.destroyAllWindows()