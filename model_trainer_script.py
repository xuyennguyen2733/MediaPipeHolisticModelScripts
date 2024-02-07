import cv2
import numpy as np
import os
from matplotlib import pyplot
import time
import mediapipe
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
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
    print("face", face.shape)
    
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
DATA_PATH = os.path.join('MP_Data')
MODEL_PATH = os.path.join('MP_Models')

# Action that we try to detect
# actions = np.array(['hello', 'thanks', 'iloveyou', 'I', 'you', 'deaf', 'hearing', 'what_question', 'what_relative_clause'])
signs = np.array(os.listdir(DATA_PATH))
print(signs)

# Thirty videos worth of data
num_examples = 30

# Videos are going to be 30 frames in length
num_frames = 30
       
label_map = {label:num for num, label in enumerate(signs)}

sequences, labels = [], []
for sign in signs:
    for num in range(num_examples):
        window = []
        for frame_num in range(num_frames):
            res = np.load(os.path.join(DATA_PATH, sign, str(num), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[sign])

x = np.array(sequences)
y = to_categorical(labels).astype(int)
print("x shape",x.shape)
print("y shape", y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05)

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir) # web app to monitor training

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1692))) # 30 frames, 1692 keypoints
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(signs.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

model.fit(x_train, y_train, epochs=200, callbacks=[tb_callback])

print('Model sumary', model.summary())


model.save(f'{MODEL_PATH}/common_signs.keras') # save model

#model.load_weights('action.keras') # reload model after initializing and compiling (7)
# print(model.summary())
#test = np.load("C:/Users/xuyen/Documents/Project/LSTMModelForASLWithTensorFlowAndMediaPipe/test_data.npy")
#new_test = np.append(x_test, [test], axis=0)

res = model.predict(x_test)

print(model.evaluate(x_test,y_test))