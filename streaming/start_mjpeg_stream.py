import cv2
from flask import Flask, Response
import time
import argparse

app = Flask(__name__)

def generate_frames(video_path, target_fps=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Unable to open video file")
        return
    if target_fps is None:
        target_fps = cap.get(cv2.CAP_PROP_FPS)

    # Calculate the delay between frames (in seconds)
    frame_delay = 1.0 / target_fps

    while cap.isOpened():
        start_time = time.time()

        success, frame = cap.read()
        if not success:
            break

        # Encode the frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # Yield the frame as a multipart response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        # Ensure the loop respects the target FPS
        elapsed_time = time.time() - start_time
        time.sleep(max(0, frame_delay - elapsed_time))

    cap.release()

@app.route('/stream')
def stream():
    print(video_file_path)
    if video_file_path is None:
        return "No video file specified", 400
    return Response(generate_frames(video_file_path),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('video_file', type=str, help='path to mp4 video file to stream')
    args = parser.parse_args()
    
    # Store video path globally
    global video_file_path
    video_file_path = args.video_file
    
    app.run(host="0.0.0.0", port=8080, debug=True)