import socket
import rclpy
from geometry_msgs.msg import Twist
import time
import atexit  # 프로그램 종료 시 실행할 정리 함수 등록
import signal  # 종료 신호 처리

# 로봇의 수신 설정
HOST = "0.0.0.0"
PORT = 5000
MAX_IDLE_TIME = 120  # 최대 유휴 시간 (초)

# 소켓 및 ROS2 퍼블리셔 변수
sock = None
node = None
pub = None

def cleanup():
    """
    프로그램 종료 시 리소스를 정리합니다.
    """
    global sock, node, pub

    # ROS2 퍼블리셔로 정지 메시지 발행
    if pub is not None:
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        pub.publish(twist)
        print("ROS2: 정지 메시지 발행")

    # ROS2 종료
    if node is not None:
        rclpy.shutdown()
        print("ROS2: 노드 종료")

    # 소켓 닫기
    if sock is not None:
        sock.close()
        print("소켓: 닫힘")

    print("프로그램 종료 및 리소스 정리 완료")


def signal_handler(sig, frame):
    """
    종료 신호 처리 핸들러.
    """
    print(f"종료 신호 {sig} 수신. 프로그램 종료 중...")
    cleanup()
    exit(0)


def main():
    global sock, node, pub

    # 종료 시 정리 함수 등록
    atexit.register(cleanup)
    signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C 신호 처리
    signal.signal(signal.SIGTERM, signal_handler)  # 종료 신호 처리

    # ROS 2 초기화
    rclpy.init()
    node = rclpy.create_node('joystick_receiver')

    # Twist 메시지 퍼블리셔 생성
    pub = node.create_publisher(Twist, 'cmd_vel', 10)

    # 소켓 설정
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((HOST, PORT))
    node.get_logger().info(f"조이스틱 데이터 수신 대기 중... (포트 {PORT})")

    # Twist 메시지 초기화
    twist = Twist()
    speed = 0.5
    turn = 0.2

    # 로봇 상태 유지용 변수
    linear_x = 0.0
    angular_z = 0.0

    # 마지막 데이터 수신 시간 기록
    last_received_time = time.time()

    try:
        while rclpy.ok():
            # 새로운 데이터 수신
            sock.settimeout(0.1)
            try:
                data, addr = sock.recvfrom(1024)
                last_received_time = time.time()
                decoded_data = data.decode()
                print(f"수신한 데이터: {decoded_data} (보낸 주소: {addr})")

                x, y, _, _, _, _, _, _, _, _, dist = map(float, decoded_data.split(","))

                # 방향 제어
                if y == 1:  # 앞으로
                    linear_x = speed
                elif y == -1:  # 뒤로
                    linear_x = -speed
                else:  # 중립
                    linear_x = 0.0

                if x == 1:  # 오른쪽 회전
                    angular_z = -abs(dist / 1000 * 2.5)
                elif x == -1:  # 왼쪽 회전
                    angular_z = abs(dist / 1000 * 2.5)
                else:  # 중립
                    angular_z = 0.0

            except socket.timeout:
                pass  # 새로운 데이터가 없으면 유지

            # 현재 시간과 마지막 수신 시간 비교
            current_time = time.time()
            if current_time - last_received_time > MAX_IDLE_TIME:
                node.get_logger().info("1분 이상 데이터 수신 없음. 프로그램 종료.")
                break

            # Twist 메시지 업데이트
            twist.linear.x = linear_x
            twist.angular.z = angular_z
            pub.publish(twist)

            # 10Hz로 루프 실행
            rclpy.spin_once(node, timeout_sec=0.1)

    except KeyboardInterrupt:
        node.get_logger().info("종료 요청을 받았습니다.")

    finally:
        cleanup()


if __name__ == "__main__":
    main()
