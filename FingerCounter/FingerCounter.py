import cv2 as cv
import mediapipe as mp
import numpy as np


def checkPosition(hand, indexes):
    init_dist = np.sqrt((hand[0].x - hand[indexes[3]].x)**2 + (hand[0].y - hand[indexes[3]].y)**2)
    thres_dist = np.sqrt((hand[0].x - hand[indexes[2]].x)**2 + (hand[0].y - hand[indexes[2]].y)**2)
    return init_dist - thres_dist < (init_dist / 10)


def checkThumb(hand):
    return between(hand[3].x, hand[5].x, hand[17].x)


def between(x, left, right):
    return left <= x <= right or left >= x >= right


model = mp.solutions.hands.Hands(max_num_hands=4)
mpDraw = mp.solutions.drawing_utils

hand_indexes = {"thumb": (1, 2, 3, 4),
                "index": (5, 6, 7, 8),
                "middle": (9, 10, 11, 12),
                "ring": (13, 14, 15, 16),
                "pinky": (17, 18, 19, 20)}

webcam = cv.VideoCapture(0)

H, W = int(webcam.get(3)), int(webcam.get(4))

while True:
    ret, frame = webcam.read()

    frame = cv.flip(frame, 1)

    results = model.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB))

    if results.multi_hand_landmarks:
        ans = [5, 5, 5, 5, 5]
        i = 0
        for hands in results.multi_hand_landmarks:
            for finger in ["index", "middle", "ring", "pinky"]:
                if checkPosition(hands.landmark, hand_indexes[finger]):
                    ans[i] -= 1
            if checkThumb(hands.landmark):
                ans[i] -= 1
            mpDraw.draw_landmarks(frame, hands, mp.solutions.hands_connections.HAND_CONNECTIONS)
            x, y = int(hands.landmark[0].x * W), int(hands.landmark[0].y * H)
            cv.putText(frame, str(ans[i]), (x, y), cv.FONT_HERSHEY_PLAIN, 5.0, (166, 71, 71), 7)

            i += 1

    cv.imshow("hands", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv.destroyAllWindows()
