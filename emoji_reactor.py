#!/usr/bin/env python3
"""
Real-time emoji display based on camera pose and facial expression detection,
with added LAUGH and ANGRY expressions.
"""

import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh

# Configuration (sesuaikan jika perlu)
SMILE_THRESHOLD = 0.18
LAUGH_THRESHOLD = 0.32
ANGRY_BROW_THRESHOLD = 0.30    # jarak antar alis (lebih kecil -> alis mendekat)
ANGRY_MOUTH_THRESHOLD = 0.50   # MAR lebih kecil dari ini cenderung mulut menutup (angry)
WINDOW_WIDTH = 720
WINDOW_HEIGHT = 450
EMOJI_WINDOW_SIZE = (WINDOW_WIDTH, WINDOW_HEIGHT)

# Frame persistence (hysteresis)
LAUGH_FRAMES_REQUIRED = 3
ANGRY_FRAMES_REQUIRED = 5
SMILE_FRAMES_REQUIRED = 1

# Load emoji images (pastikan file ada di folder yang sesuai)
try:
    smiling_emoji = cv2.imread("smile.png")
    straight_face_emoji = cv2.imread("plain.png")
    hands_up_emoji = cv2.imread("air.png")
    laugh_emoji = cv2.imread("laugh.png")
    angry_emoji = cv2.imread("angry.png") 

    if smiling_emoji is None:
        raise FileNotFoundError("smile.png not found")
    if straight_face_emoji is None:
        raise FileNotFoundError("plain.png not found")
    if hands_up_emoji is None:
        raise FileNotFoundError("air.png not found")
    if laugh_emoji is None:
        raise FileNotFoundError("laugh.png not found")
    if angry_emoji is None:
        raise FileNotFoundError("angry.png not found")

    # Resize emojis
    smiling_emoji = cv2.resize(smiling_emoji, EMOJI_WINDOW_SIZE)
    straight_face_emoji = cv2.resize(straight_face_emoji, EMOJI_WINDOW_SIZE)
    hands_up_emoji = cv2.resize(hands_up_emoji, EMOJI_WINDOW_SIZE)
    laugh_emoji = cv2.resize(laugh_emoji, EMOJI_WINDOW_SIZE)
    angry_emoji = cv2.resize(angry_emoji, EMOJI_WINDOW_SIZE)

except Exception as e:
    print("Error loading emoji images!")
    print(f"Details: {e}")
    print("\nExpected files:")
    print("- smile.png (smiling face)")
    print("- plain.png (straight face)")
    print("- air.png (hands up)")
    print("- laugh.png (laugh)")
    print("- angry.png (angry)")
    exit()

blank_emoji = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8)

# Start webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

cv2.namedWindow('Emoji Output', cv2.WINDOW_NORMAL)
cv2.namedWindow('Camera Feed', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Camera Feed', WINDOW_WIDTH, WINDOW_HEIGHT)
cv2.resizeWindow('Emoji Output', WINDOW_WIDTH, WINDOW_HEIGHT)
cv2.moveWindow('Camera Feed', 100, 100)
cv2.moveWindow('Emoji Output', WINDOW_WIDTH + 150, 100)

print("Controls:")
print("  Press 'q' to quit")
print("  Smile for smiling emoji")
print("  Laugh (open mouth wide) for laugh emoji")
print("  Make angry face (furrow brows + mouth closed) for angry emoji")

# Counters for hysteresis
laugh_counter = 0
smile_counter = 0
angry_counter = 0

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose, \
     mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5) as face_mesh:

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False

        current_state = "STRAIGHT_FACE"

        # HANDS UP detection (keberadaan, tetap optional)
        results_pose = pose.process(image_rgb)
        if results_pose.pose_landmarks:
            lm = results_pose.pose_landmarks.landmark
            left_shoulder = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            left_wrist = lm[mp_pose.PoseLandmark.LEFT_WRIST]
            right_wrist = lm[mp_pose.PoseLandmark.RIGHT_WRIST]

            if (left_wrist.y < left_shoulder.y) or (right_wrist.y < right_shoulder.y):
                current_state = "HANDS_UP"

        # FACE detection & expression analysis (jika bukan hands up)
        if current_state != "HANDS_UP":
            results_face = face_mesh.process(image_rgb)
            if results_face.multi_face_landmarks:
                face_landmarks = results_face.multi_face_landmarks[0]

                # Landmark yang kita pakai:
                # - mouth corners: 61 (right), 291 (left)
                # - upper lip: 13, lower lip: 14
                # - inner brow-ish points: 70 (left inner brow area), 300 (right inner brow area)
                # Catatan: indices MediaPipe 0..467 valid; pemilihan ini adalah heuristik umum.

                # mouth landmarks
                left_corner = face_landmarks.landmark[291]
                right_corner = face_landmarks.landmark[61]
                upper_lip = face_landmarks.landmark[13]
                lower_lip = face_landmarks.landmark[14]

                # brow landmarks (inner brow-ish)
                brow_left = face_landmarks.landmark[70]
                brow_right = face_landmarks.landmark[200]

                # compute normalized distances (using normalized coordinates)
                mouth_width = np.linalg.norm(np.array([right_corner.x, right_corner.y]) -
                                             np.array([left_corner.x, left_corner.y]))
                mouth_height = np.linalg.norm(np.array([lower_lip.x, lower_lip.y]) -
                                              np.array([upper_lip.x, upper_lip.y]))
                # inter-brow distance (smaller => alis mendekat)
                brow_distance = np.linalg.norm(np.array([brow_left.x, brow_left.y]) -
                                               np.array([brow_right.x, brow_right.y]))

                # mouth aspect ratio (normalized)
                mar = 0.0
                if mouth_width > 1e-6:
                    mar = mouth_height / mouth_width

                # --- Debug overlay (nilai MAR dan brow_distance) ---
                # Tampilkan nilai di layar untuk kalibrasi
                cv2.putText(frame, f"MAR:{mar:.3f} BROW_DIST:{brow_distance:.3f}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # ============================
                # Update counters (hysteresis)
                # Prioritas: ANGRY > LAUGH > SMILE > STRAIGHT_FACE
                # ============================
                # angry: alis mendekat (brow_distance kecil) AND mulut relatif sempit (mar kecil)
                if (brow_distance < ANGRY_BROW_THRESHOLD * 0.8) and (mar < ANGRY_MOUTH_THRESHOLD * 0.7):
                    angry_counter += 1
                else:
                    angry_counter = 0


                # laugh: mulut terbuka sangat lebar
                if mar > LAUGH_THRESHOLD:
                    laugh_counter += 1
                else:
                    laugh_counter = 0

                # smile: sedikit terbuka (lebih sensitif)
                if mar > SMILE_THRESHOLD:
                    smile_counter += 1
                else:
                    smile_counter = 0

                # decide state with priority
                if angry_counter >= ANGRY_FRAMES_REQUIRED:
                    current_state = "ANGRY"
                elif laugh_counter >= LAUGH_FRAMES_REQUIRED:
                    current_state = "LAUGHING"
                elif smile_counter >= SMILE_FRAMES_REQUIRED:
                    current_state = "SMILING"
                else:
                    current_state = "STRAIGHT_FACE"

        # Select emoji image and label
        if current_state == "SMILING":
            emoji_to_display = smiling_emoji
            emoji_name = "😊"
        elif current_state == "LAUGHING":
            emoji_to_display = laugh_emoji
            emoji_name = "😆"
        elif current_state == "ANGRY":
            emoji_to_display = angry_emoji
            emoji_name = "😠"
        elif current_state == "HANDS_UP":
            emoji_to_display = hands_up_emoji
            emoji_name = "🙌"
        elif current_state == "STRAIGHT_FACE":
            emoji_to_display = straight_face_emoji
            emoji_name = "😐"
        else:
            emoji_to_display = blank_emoji
            emoji_name = "❓"

        camera_frame_resized = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))

        cv2.putText(camera_frame_resized, f'STATE: {current_state} {emoji_name}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(camera_frame_resized, 'Tekan "q" untuk keluar', (10, WINDOW_HEIGHT - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Camera Feed', camera_frame_resized)
        cv2.imshow('Emoji Output', emoji_to_display)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
