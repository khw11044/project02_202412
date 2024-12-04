import socket

# 로봇의 IP 주소와 포트 설정
HOST = "0.0.0.0"  # 모든 네트워크 인터페이스에서 수신
PORT = 5000       # 위에서 설정한 포트와 동일하게 설정

def main():
    # 소켓 생성
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((HOST, PORT))

    print(f"모바일 로봇 데이터 수신 대기 중... (포트: {PORT})")
    try:
        while True:
            # 데이터 수신
            data, addr = sock.recvfrom(1024)  # 버퍼 크기 설정
            decoded_data = data.decode()
            print(f"수신한 데이터: {decoded_data} (보낸 주소: {addr})")
    except KeyboardInterrupt:
        print("\n프로그램 종료")
    finally:
        sock.close()

if __name__ == "__main__":
    main()
