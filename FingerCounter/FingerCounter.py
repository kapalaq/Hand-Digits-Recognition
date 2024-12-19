import cv2 as cv
import mediapipe as mp
import numpy as np
import os
import torch
import torch.nn as nn
from collections import namedtuple

from holoviews.operation import threshold

mpHands = mp.solutions.hands
def get_coordinates(hand: namedtuple) -> np.ndarray:
    """
    Converts NamedTuple of coordinates to numpy array
    :param hand: - output of mediapipe hand recognizer
    :return: flatten numpy array of coordinates
    """
    hand_arr = np.zeros(63)
    i = 0
    for el in hand.landmark:
        hand_arr[i] = el.x
        hand_arr[i + 1] = el.y
        hand_arr[i + 2] = el.z
        i += 3
    return hand_arr

def recognizer(frame: np.ndarray, model: mpHands.Hands) -> namedtuple:
    """
    Recognizing hands landmarks on frame
    :param frame: - np.ndarray representing hands landmarks
    :param model: - MediaPipe solution for hand recognition
    :return: - NamedTuple of hands landmarks
    """
    image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    image.flags.writeable = False
    result = model.process(image).multi_hand_landmarks
    image.flags.writeable = True
    return result

def probability_graph(result: np.ndarray, targets: np.ndarray, frame: np.ndarray, colors: tuple):
    """
    Drawing probability distribution over all possible classes on frame
    :param result: - array of predicted probabilities
    :param targets: - array of actual values
    :param frame: - image in numpy array format
    :param colors: - colors for each class
    :return frame: - image with histogram of predicted probabilities
    """
    output_frame = frame.copy()
    for num, prob in enumerate(result):
        cv.rectangle(output_frame, (0, 55 + num * 40), (int(prob * 100), 85 + num * 40), colors[num], -1)
        cv.putText(output_frame, targets[num], (0, 70 + num * 38), cv.FONT_HERSHEY_SIMPLEX, 0.5,
                   (255, 255, 255), 1, cv.LINE_AA)
    return output_frame


if __name__ == '__main__':
    colors = (
        (255, 0, 0),  # Red
        (0, 255, 0),  # Green
        (0, 0, 255),  # Blue
        (255, 255, 0),  # Yellow
        (0, 255, 255),  # Cyan
    )

    model = nn.Sequential(
        nn.Linear(63, 16),
        nn.ReLU(),
        nn.Linear(16, 5),
        nn.Softmax(dim=1)
    )
    digits = np.array(["one", "two", "three", "four", "five"])
    model.load_state_dict(torch.load(os.path.join("saved_models", "FingerCounter(alp-1).pth")))
    mpDraw = mp.solutions.drawing_utils
    threshold = 0.8
    with mp.solutions.hands.Hands(max_num_hands=4) as mp_hands:
        webcam = cv.VideoCapture(0)
        H, W = int(webcam.get(3)), int(webcam.get(4))

        try:
            while webcam.isOpened():
                ret, frame = webcam.read()

                frame = cv.flip(frame, 1)
                result = recognizer(cv.cvtColor(frame, cv.COLOR_BGR2RGB), mp_hands)

                if result:
                    for hand in result:
                        coords = np.array([get_coordinates(hand)])
                        mpDraw.draw_landmarks(frame, hand, mp.solutions.hands_connections.HAND_CONNECTIONS)
                        prob = model(torch.tensor(coords).float()).detach().numpy()
                        x, y = int(hand.landmark[0].x * W), int(hand.landmark[0].y * H)
                        if prob.reshape(-1)[np.argmax(prob)] > threshold:
                            cv.putText(frame, str(np.argmax(prob) + 1), (x, y), cv.FONT_HERSHEY_PLAIN, 5.0, (166, 71, 71), 7)
                    frame = probability_graph(prob.reshape(-1), digits, frame, colors)
                cv.imshow("hands", frame)
                if cv.waitKey(1) & 0xFF == ord('q'):
                    break
        except Exception as e:
            print(e)
        finally:
            webcam.release()
            cv.destroyAllWindows()
