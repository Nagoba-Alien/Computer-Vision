import cv2
import mediapipe as mp
import numpy as np
import math


# =====================================================
# MediaPipe Setup
# =====================================================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)


# =====================================================
# Gesture Detection
# =====================================================

def get_gesture(hand_landmarks):

    lm = hand_landmarks.landmark

    index_up = lm[8].y < lm[6].y
    middle_up = lm[12].y < lm[10].y

    if index_up and not middle_up:
        return "DRAW"

    elif index_up and middle_up:
        return "ERASE"

    else:
        return "IDLE"


def get_pinch_distance(hand_landmarks, w, h):

    lm = hand_landmarks.landmark

    thumb = lm[4]
    index = lm[8]

    thumb_pos = (
        int(thumb.x*w),
        int(thumb.y*h)
    )

    index_pos = (
        int(index.x*w),
        int(index.y*h)
    )

    distance = math.hypot(
        thumb_pos[0]-index_pos[0],
        thumb_pos[1]-index_pos[1]
    )

    return distance, thumb_pos, index_pos


# =====================================================
# Camera
# =====================================================
cap = cv2.VideoCapture(0)


canvas = None


previous_point = None


eraser_radius = 30


# =====================================================
# Gesture Stabilization
# =====================================================

previous_gesture = {
    "Left": "IDLE",
    "Right": "IDLE"
}


gesture_counter = {
    "Left": 0,
    "Right": 0
}


confirmed_gesture = {
    "Left": "IDLE",
    "Right": "IDLE"
}


GESTURE_THRESHOLD = 5


# =====================================================
# Main Loop
# =====================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Mirror camera

    frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape

    if canvas is None:
        canvas = np.zeros_like(frame)

    # Convert for MediaPipe

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    result = hands.process(rgb)

    active_hand = None
    controller_hand = None

    active_mode = "IDLE"

    # =================================================
    # Hand Detection
    # =================================================

    if result.multi_hand_landmarks:

        detected_hands = result.multi_hand_landmarks

        # Draw landmarks

        for hand in detected_hands:

            mp_draw.draw_landmarks(
                frame,
                hand,
                mp_hands.HAND_CONNECTIONS
            )

        # ---------------------------------------------
        # Determine stable gestures
        # ---------------------------------------------

        for i, hand in enumerate(detected_hands):

            # Get Left/Right identity

            label = result.multi_handedness[i].classification[0].label

            gesture = get_gesture(hand)

            if gesture == previous_gesture[label]:

                gesture_counter[label] += 1

            else:

                gesture_counter[label] = 0

                previous_gesture[label] = gesture

            if gesture_counter[label] >= GESTURE_THRESHOLD:

                confirmed_gesture[label] = gesture

            # Debug display

            cv2.putText(
                frame,
                f"{label}: {confirmed_gesture[label]}",
                (20, 120+i*40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

        # ---------------------------------------------
        # Select active hand
        # ---------------------------------------------

        for i, hand in enumerate(detected_hands):

            label = result.multi_handedness[i].classification[0].label

            if confirmed_gesture[label] in ["DRAW", "ERASE"]:

                active_hand = i

                active_mode = confirmed_gesture[label]

                break

        # ---------------------------------------------
        # Remaining hand = controller
        # ---------------------------------------------

        if active_hand is not None:

            for i in range(len(detected_hands)):

                if i != active_hand:

                    controller_hand = i

        # =================================================
        # Eraser size control
        # =================================================

        if controller_hand is not None:

            distance, thumb, index = get_pinch_distance(
                detected_hands[controller_hand],
                w,
                h
            )

            eraser_radius = int(
                np.interp(
                    distance,
                    [20, 150],
                    [10, 100]
                )
            )

            # Draw pinch line

            cv2.line(
                frame,
                thumb,
                index,
                (255, 0, 255),
                3
            )

            cv2.circle(
                frame,
                thumb,
                8,
                (255, 0, 255),
                -1
            )

            cv2.circle(
                frame,
                index,
                8,
                (255, 0, 255),
                -1
            )

        # =================================================
        # Drawing hand action
        # =================================================

        if active_hand is not None:

            hand = detected_hands[active_hand]

            lm = hand.landmark

            # Index fingertip

            fingertip = (
                int(lm[8].x*w),
                int(lm[8].y*h)
            )

            cv2.circle(
                frame,
                fingertip,
                10,
                (0, 0, 255),
                -1
            )

            # -------------------------
            # DRAW
            # -------------------------

            if active_mode == "DRAW":

                if previous_point is not None:

                    cv2.line(
                        canvas,
                        previous_point,
                        fingertip,
                        (0, 255, 0),
                        5
                    )

                previous_point = fingertip

            # -------------------------
            # ERASE
            # -------------------------

            elif active_mode == "ERASE":

                cv2.circle(
                    canvas,
                    fingertip,
                    eraser_radius,
                    (0, 0, 0),
                    -1
                )

                previous_point = None

    else:

        previous_point = None

    # =================================================
    # Display
    # =================================================

    output = cv2.add(
        frame,
        canvas
    )

    cv2.putText(
        output,
        f"MODE: {active_mode}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    cv2.putText(
        output,
        f"Eraser Size: {eraser_radius}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        "Virtual Air Canvas",
        output
    )

    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()

cv2.destroyAllWindows()
