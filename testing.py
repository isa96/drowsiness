import cv2
import numpy as np
import tensorflow as tf
import vlc

model = tf.keras.models.load_model("model/drowsiness_full.h5")
path = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + path)
awake = vlc.MediaPlayer(
    "assets/Jojo's Bizarre Adventure- Awaken(Pillar Men Theme).mp3")

cam_ids = [-1, 0, 1]
for i in cam_ids:
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        break

if not cap.isOpened():
    raise IOError("Cannot Open Webcam")

while True:
    ret, frame = cap.read()
    eye_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_eye.xml")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in eyes:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyess = eye_cascade.detectMultiScale(roi_gray)
        if len(eyess) == 0:
            print("Eyes are not detected")
        else:
            for (ex, ey, ew, eh) in eyess:
                eyes_roi = roi_color[ey: ey+eh, ex:ex+ew]
    try:
        final_image = cv2.resize(eyes_roi, (224, 224))
        final_image = np.expand_dims(final_image, axis=0)  # Butuh 4 Dimensi
        final_image = final_image/255.0

        Predictions = model.predict(final_image)
        if (Predictions > 0.5):
            status = "Open Eyes"
            awake.stop()
        else:
            status = "Closed Eyes"
            awake.play()
    except NameError:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    print(faceCascade.empty())
    faces = faceCascade.detectMultiScale(gray, 1.1, 4)

    # Draw Rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Use putText() method for
    # Inserting text on video
    cv2.putText(frame,
                status,
                (50, 50),
                font, 3,
                (0, 0, 255), 2,
                cv2.LINE_4)
    cv2.imshow('Drowsiness Detection', frame)

    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
