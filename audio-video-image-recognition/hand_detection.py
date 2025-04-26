import cv2
import mediapipe as mp
import os
from tqdm import tqdm

# Mediapipe documentation: https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker?hl=pt-br

def detect_hands(video_path, output_path):
    mo_hands = mp.solutions.hands
    hands = mo_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for _ in tqdm(range(total_frames), desc="Processing video..."):
        ret, frame = cap.read()

        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mo_hands.HAND_CONNECTIONS
                )

        out.write(frame)

        cv2.imshow('Hand Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_video_path = os.path.join(script_dir, 'data/input_hands_movements.mp4')
    output_video_path = os.path.join(script_dir, 'data/output_video_hands.mp4')

    detect_hands(input_video_path, output_video_path)