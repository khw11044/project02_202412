import cv2
from flask import Flask, Response
import time

app = Flask(__name__)

# OpenCV로 웹캠 캡처 객체 생성
camera = cv2.VideoCapture(0)

def generate_frames():
    """웹캠에서 프레임을 가져와 스트리밍"""
    while True:
        success, frame = camera.read()  # 웹캠에서 프레임 읽기
        if not success:
            break
        else:
            # 프레임 크기 조정 (640x480)
            frame = cv2.resize(frame, (640, 480))

            # JPEG 인코딩 품질 조정
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            frame = buffer.tobytes()

            # 프레임 데이터를 전송
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        # FPS 제한 (30 FPS)
        time.sleep(1 / 30)

@app.route('/video_feed')
def video_feed():
    """비디오 스트리밍"""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    # Flask 앱 실행
    try:
        app.run(host='0.0.0.0', port=5000, threaded=True)
    finally:
        # 리소스 해제
        if camera.isOpened():
            camera.release()
