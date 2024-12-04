import socket
import time
from inputs import get_gamepad

# 모바일 로봇의 IP 주소와 포트 설정
ROBOT_IP = "192.168.0.123"  # 모바일 로봇의 IP 주소 (변경 필요)
ROBOT_PORT = 5000  # 모바일 로봇의 수신 포트

# 초기화
x, y = 127, 127
blue, green, red, yellow, left_top, right_top = 0, 0, 0, 0, 0, 0
select_key, start_key = 0, 0
dist = 0.5

last_data = None
last_sent_time = 0
send_interval = 0.05  # 최소 전송 간격 (초)

def process_event(event, sock):
    """
    게임패드 이벤트를 처리하여 데이터를 갱신합니다.
    """
    global x, y
    global blue, green, red, yellow, left_top, right_top, select_key, start_key
    global last_data, last_sent_time
    global dist

    changed = False

    if event.ev_type == 'Absolute':
        if event.code == 'ABS_X':  # X축
            x = int(event.state)
            changed = True
        elif event.code == 'ABS_Y':  # Y축
            y = int(event.state)
            changed = True

        if changed:
            if x == 255:
                x = 1
            elif x == 0:
                x = -1

            if y == 0:
                y = 1
            elif y == 255:
                y = -1

    elif event.ev_type == 'Key':
        if event.code == 'BTN_TRIGGER':  # X 버튼
            blue = int(event.state)
            changed = True
        elif event.code == 'BTN_TOP':  # Y 버튼
            green = int(event.state)
            changed = True
        elif event.code == 'BTN_THUMB':  # A 버튼
            red = int(event.state)
            changed = True
        elif event.code == 'BTN_THUMB2':  # B 버튼
            yellow = int(event.state)
            changed = True
        elif event.code == 'BTN_TOP2':  # Left Top
            left_top = int(event.state)
            changed = True
        elif event.code == 'BTN_PINKIE':  # Right Top
            right_top = int(event.state)
            changed = True
        elif event.code == 'BTN_BASE3':  # Select
            select_key = int(event.state)
            changed = True
        elif event.code == 'BTN_BASE4':  # Start
            start_key = int(event.state)
            changed = True

    if changed:
        data = f"{x},{y},{blue},{green},{red},{yellow},{left_top},{right_top},{select_key},{start_key},{dist}"

        # 동일 데이터 반복 전송 방지
        if data != last_data and time.time() - last_sent_time >= send_interval:
            sock.sendto(data.encode(), (ROBOT_IP, ROBOT_PORT))
            last_data = data
            last_sent_time = time.time()

            # 디버깅 출력
            print(f"Processed event: {event.ev_type}, {event.code}, {event.state}")
            print("전송 데이터:", data)


def main():
    global x, y, blue, green, red, yellow, left_top, right_top, select_key, start_key

    # 소켓 생성 및 버퍼 크기 설정
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 65535)  # 송신 버퍼 크기 설정

    print("게임패드 입력을 모바일 로봇으로 전송합니다.")
    try:
        while True:
            events = get_gamepad()
            for event in events:
                try:
                    # 이벤트 처리
                    process_event(event, sock)
                except socket.error as e:
                    print(f"소켓 전송 오류: {e}")
    except KeyboardInterrupt:
        print("\n프로그램 종료")
    finally:
        sock.close()

if __name__ == "__main__":
    main()
