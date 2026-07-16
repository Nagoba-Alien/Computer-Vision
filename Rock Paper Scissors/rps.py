import cv2
import mediapipe as mp
import random
import time
from collections import Counter

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)

cap = cv2.VideoCapture(0)

countdown_start = time.time()
captured = False
player_move = None
computer_move = None
result = ""
samples = []

COUNTDOWN = 3
CAPTURE_TIME = 1.0


def classify(hand_landmarks):
    lm = hand_landmarks.landmark

    fingers = []

    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]

    for tip, pip in zip(tips, pips):
        fingers.append(lm[tip].y < lm[pip].y)

    if fingers == [False, False, False, False]:
        return "rock"
    elif fingers == [True, True, True, True]:
        return "paper"
    elif fingers == [True, True, False, False]:
        return "scissors"

    return None


def winner(player, computer):
    if player == computer:
        return "Tie!"
    if ((player == "rock" and computer == "scissors") or
        (player == "paper" and computer == "rock") or
            (player == "scissors" and computer == "paper")):
        return "You win!"
    return "Computer wins!"


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)

    elapsed = time.time() - countdown_start

    if elapsed < COUNTDOWN:
        cv2.putText(frame, str(COUNTDOWN - int(elapsed)),
                    (280, 100), cv2.FONT_HERSHEY_SIMPLEX, 3,
                    (0, 255, 255), 5)

    elif elapsed < COUNTDOWN + CAPTURE_TIME:
        cv2.putText(frame, "Shoot!", (220, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)

        if res.multi_hand_landmarks:
            hand = res.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            move = classify(hand)
            if move:
                samples.append(move)

    elif not captured:
        if samples:
            player_move = Counter(samples).most_common(1)[0][0]
            computer_move = random.choice(["rock", "paper", "scissors"])
            result = winner(player_move, computer_move)
        else:
            player_move = "No gesture"
            computer_move = "-"
            result = "Try again"

        captured = True
        display_start = time.time()

    else:
        cv2.putText(frame, f"You: {player_move}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"CPU: {computer_move}", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, result, (20, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        if time.time() - display_start > 2:
            countdown_start = time.time()
            captured = False
            samples.clear()

    cv2.imshow("Rock Paper Scissors", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
