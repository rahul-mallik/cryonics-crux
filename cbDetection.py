import cv2
import mediapipe as mp

mp_holistic = mp.solutions.holistic

cap = cv2.VideoCapture(0)

# mediapipe model 
movement_data = []
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, smooth_landmarks=True)
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    image.flags.writeable = False                   
    results = model.process(image)                 
    image.flags.writeable = True                  
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
    return image, results

#custom colored landmark
def draw_styled_landmarks(image, results, hand_results):
    #draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                               mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                               mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1))
    # draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                               mp_drawing.DrawingSpec(color=(173,216,230), thickness=2, circle_radius=4), 
                               mp_drawing.DrawingSpec(color=(173,216,230), thickness=2, circle_radius=2))
    # draw hand connections
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS, 
                                       mp_drawing.DrawingSpec(color=(245,117,66), thickness=1, circle_radius=3), 
                                       mp_drawing.DrawingSpec(color=(245,66,230), thickness=1, circle_radius=2))

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        # reading feed from webcam
        ret, frame = cap.read()
        if not ret:
            break

        #holistic detections
        image, results = mediapipe_detection(frame, holistic)
        hand_results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        #showing landmarks in the video
        draw_styled_landmarks(image, results, hand_results)

        if results.pose_landmarks:
            #landmark coordinates
            landmarks = results.pose_landmarks.landmark
            keypoints = {mp_pose.PoseLandmark(i).name: (lm.x, lm.y) for i, lm in enumerate(landmarks)}

            left_shoulder, right_shoulder = keypoints['LEFT_SHOULDER'], keypoints['RIGHT_SHOULDER']
            left_hip, right_hip = keypoints['LEFT_HIP'], keypoints['RIGHT_HIP']
            left_knee, right_knee = keypoints['LEFT_KNEE'], keypoints['RIGHT_KNEE']
            left_wrist, right_wrist = keypoints['LEFT_WRIST'], keypoints['RIGHT_WRIST']

            #movement differences
            shoulder_diff = abs(left_shoulder[1] - right_shoulder[1])
            hip_diff = abs(left_hip[1] - right_hip[1])
            knee_diff = abs(left_knee[1] - right_knee[1])

            is_asymmetrical = shoulder_diff > 0.1 or hip_diff > 0.1  
            is_scissoring = abs(left_hip[0] - right_hip[0]) < 0.05 and knee_diff > 0.15 

            #handlandmarks
            is_fisting = False
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
                    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

                    #distances between fingertips
                    finger_distances = [
                        abs(index_tip.x - middle_tip.x),
                        abs(middle_tip.x - ring_tip.x),
                        abs(ring_tip.x - pinky_tip.x),
                        abs(pinky_tip.x - thumb_tip.x)
                    ]

                    # chekc for the fist
                    if all(dist < 0.03 for dist in finger_distances):
                        is_fisting = True

            movement_data.append({
                'asymmetry': is_asymmetrical,
                'scissoring': is_scissoring,
                'fisting': is_fisting
            })

        cv2.imshow('Cerebral Palsy Detection', image)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

    total_frames = len(movement_data)

    asymmetry_count = sum(d['asymmetry'] for d in movement_data)
    scissoring_count = sum(d['scissoring'] for d in movement_data)
    fisting_count = sum(d['fisting'] for d in movement_data)

    #percentages calculation
    asymmetry_percentage = (asymmetry_count / total_frames) * 100
    scissoring_percentage = (scissoring_count / total_frames) * 100
    fisting_percentage = (fisting_count / total_frames) * 100

    print(f"\n - Asymmetry detected in {asymmetry_percentage:.2f}% of frames.")
    print(f" - Scissoring detected in {scissoring_percentage:.2f}% of frames.")
    print(f" - Fisting detected in {fisting_percentage:.2f}% of frames.")

    risk_signs = sum([
        asymmetry_percentage > 70,
        scissoring_percentage > 20,
        fisting_percentage > 75
    ])

    if risk_signs >= 1:
        print("\n Possible Signs of Cerebral Palsy Detected. Consult a doctor for further evaluation.")
    else:
        print("\n No major risk detected. Continue monitoring your child's development.")