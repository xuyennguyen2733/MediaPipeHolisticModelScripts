import cv2
import numpy as np
import os
import mediapipe
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard

mp_drawing = mediapipe.solutions.drawing_utils
mp_holistic = mediapipe.solutions.holistic

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # converts color space from BGR to RGB which saves some memory
    image.flags.writeable = False                  # marks image as unwriteable
    results = model.process(image)                 # Makes prediction
    image.flags.writeable = True                   # converts back
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # converts back
    return image, results

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(478*3)
    leftHand = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rightHand = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    
    keypoints = np.concatenate([pose, face, leftHand, rightHand])
    
    return keypoints 

def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                                 mp_drawing.DrawingSpec(color=(68,64,71), thickness=1, circle_radius=1), # joint spec
                                 mp_drawing.DrawingSpec(color=(119,155,0), thickness=1, circle_radius=1) # line spec 
                             )
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                 mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), # joint spec
                                 mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2) # line spec
                             )
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                 mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), # joint spec
                                 mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2) # line spec
                             )
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                 mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), # joint spec
                                 mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) # line spec
                             )

# Path for exported data, numpy arrays
DATA_DIR = os.path.join('MP_Data')
MODEL_DIR = os.path.join('MP_Models')

#signs = np.array(['hello', 'thanks', 'name'])
signs = np.array(os.listdir(DATA_DIR))

# Thirty videos worth of data
num_examples = 30

# Videos are going to be 30 frames in length
num_frames = 30

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1692))) # 30 frames, 1662 keypoints
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(signs.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

model.load_weights(f'{MODEL_DIR}/common_signs.keras')

sequence = []
handDetected = False
threshold = 0.3

cap = cv2.VideoCapture(0) # access webcam
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

prediction = ""
prediction2 = ""
prediction3 = ""

int_min = -1
## Set mediapipe model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5, refine_face_landmarks=True) as holistic:
    while cap.isOpened(): # while accessing webcam
        # read feed (the current frame)
        # timer = 0
        ret, frame = cap.read() 
        
        # Make detections
        image, results = mediapipe_detection(frame, holistic) # holistic: model

        if (not handDetected):
            if (results.left_hand_landmarks or results.right_hand_landmarks):
                handDetected = True

        # Draw landmarks
        draw_styled_landmarks(image, results)

        image = cv2.flip(image, 1)
        
        # Prediction logic
        if handDetected:
          keypoints = extract_keypoints(results)

          sequence.append(keypoints)

          res = []
          if len(sequence) == num_frames:
              res = model.predict(np.expand_dims(sequence, axis=0))[0]
              # visualization logic
              if res[np.argmax(res)] > threshold:
                  prediction = signs[np.argmax(res)]
                  res[np.argmax(res)] = -1
                  prediction2 = signs[np.argmax(res)]
                  res[np.argmax(res)] = -1
                  prediction3 = signs[np.argmax(res)]
                  

              handDetected = False
              sequence = []
        
        cv2.putText(image, f"Prediction 1: {prediction}", (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(image, f"Prediction 2: {prediction2}", (20,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(image, f"Prediction 3: {prediction3}", (20,110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

        # show to screen (frame name, actual frame)
        cv2.imshow('OpenCV Feed', image) 
        
        # condition to close gracefully WHEN:
        #     waited for 0.01 sec for a keypress & keypress is 'q', OR
        #     the [X] button on the window is clicked
        if (cv2.waitKey(10) & 0xFF == ord('q')) or (cv2.getWindowProperty('OpenCV Feed', cv2.WND_PROP_VISIBLE) < 1): 
            break
        
cap.release()
cv2.destroyAllWindows()