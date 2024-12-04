import socket
import rclpy
from geometry_msgs.msg import Twist

# 로봇의 수신 설정
HOST = "0.0.0.0"  # 모든 네트워크 인터페이스에서 수신
PORT = 5000  # 컴퓨터에서 전송한 포트와 동일하게 설정

def main():
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
    turn = 1.0

    # 로봇 상태 유지용 변수
    linear_x = 0.0
    angular_z = 0.0

    try:
        while rclpy.ok():
            # 새로운 데이터 수신
            sock.settimeout(0.1)  # 0.1초 동안 대기 (네트워크 비용 절감)
            try:
                data, addr = sock.recvfrom(1024)
                data = data.decode()
                x, y, blue, green, red, yellow, left_top, right_top, select_key, start_key = map(int, data.split(","))

                # 새로운 입력 값 처리
                if y == 1:  # 앞으로
                    linear_x = speed
                elif y == -1:  # 뒤로
                    linear_x = -speed
                else:  # 중립
                    linear_x = 0.0

                if x == 1:  # 오른쪽 회전
                    angular_z = -turn
                elif x == -1:  # 왼쪽 회전
                    angular_z = turn
                else:  # 중립
                    angular_z = 0.0

                 # Green (Y) 버튼 처리 (O 버튼 동작과 동일) 왼앞
                if green == 1:
                    linear_x = speed
                    angular_z = turn
                    
                    node.get_logger().info(f"그린키")
                 # Blue 버튼 (X) 버튼 처리 (U 버튼 동작과 동일) 오른 앞
                if blue == 1:
                    linear_x = speed
                    angular_z = -turn
                    node.get_logger().info(f"블루키")
                 # Red 버튼 (A) 버튼 처리 (M 버튼 동작과 동일) 오른 뒤
                if red == 1:
                    linear_x = -speed
                    angular_z = turn
                    node.get_logger().info(f"레드키")
                 # Yellow 버튼 (B) 버튼 동작 처리 (. 버튼 동작과 동일) 왼 뒤
                if yellow == 1:
                    linear_x = -speed
                    angular_z = -turn
                    node.get_logger().info(f"옐로키")
                # 속도 조정 버튼 처리
                if right_top == 1:
                    speed *= 1.1
                    turn *= 1.1
                    node.get_logger().info(f"속도 증가: speed={speed}, turn={turn}")
                if left_top == 1:
                    speed *= 0.9
                    turn *= 0.9
                    node.get_logger().info(f"속도 감소: speed={speed}, turn={turn}")

                # 초기화 버튼 처리
                if select_key == 1 or start_key == 1:
                    speed = 0.5
                    turn = 1.0
                    linear_x = 0.0
                    angular_z = 0.0
                    node.get_logger().info("초기화 완료")

            except socket.timeout:
                pass  # 새로운 데이터가 없으면 유지

            # Twist 메시지 업데이트
            twist.linear.x = linear_x
            twist.angular.z = angular_z
            pub.publish(twist)

            # 10Hz로 루프 실행
            rclpy.spin_once(node, timeout_sec=0.1)

    except KeyboardInterrupt:
        node.get_logger().info("종료 요청을 받았습니다.")

    finally:
        # 정지 메시지 발행
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        pub.publish(twist)

        # ROS 종료
        rclpy.shutdown()


if __name__ == "__main__":
    main()
