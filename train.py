import cv2
import dlib
import time
import os
import numpy as np
detector = dlib.get_frontal_face_detector()
detector1 = cv2.dnn.readNetFromCaffe('/home/bau/Desktop/thay_lam/face_recognition/face_detection_model/deploy.prototxt',
                                    '/home/bau/Desktop/thay_lam/face_recognition/face_detection_model/res10_300x300_ssd_iter_140000.caffemodel')

# Load the predictor
predictor = dlib.shape_predictor("/home/bau/Desktop/thay_lam/face_recognition/shape_predictor_68_face_landmarks.dat")

class Video1(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
    def __del__(self):
        self.video.release()
    def get_frame(self):

        count = 0
        preticked = time.time()

        while True:
            ret, frame = self.video.read()
            frame = cv2.flip(frame, 1)
            img = cv2.resize(frame, (300, 300))
            gray = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2GRAY)

            faces = detector(gray)
            for face in faces:
                x01 = face.left()  # left point
                y01 = face.top()  # top point
                x02 = face.right()  # right point
                y02 = face.bottom()  # bottom point

                landmarks = predictor(image=gray, box=face)

                x27 = landmarks.part(27).x
                y27 = landmarks.part(27).y
                x0 = landmarks.part(0).x
                y0 = landmarks.part(0).y
                y15 = landmarks.part(15).y
                x16 = landmarks.part(16).x
                y16 = landmarks.part(16).y
                y17 = landmarks.part(17).y
                y1 = landmarks.part(1).y
                y26 = landmarks.part(26).y
                y33 = landmarks.part(33).y
                x8 = landmarks.part(8).x
                x37 = landmarks.part(37).x
                x44 = landmarks.part(44).x
                y28 = landmarks.part(28).y

                d1 = int(x27 - x0)  # khoảng cách từ 0 đến 27
                d2 = int(x16 - x27)  # khoảng cách từ 27 đến 16
            currenttime = time.time()
            if y0 < y17 or y0 > y28 or y0 == y33 or y16 < y26 or x8 < x37 or x8 > x44 or d1 * 2 < d2 or d1 > d2 * 2 or y27 > y1 or y27 > y15:
                cv2.putText(frame, 'K chap nhan', (x01, y01), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 2, cv2.LINE_AA)

            else:
                imageBlob = cv2.dnn.blobFromImage(image=img, scalefactor=1.0, size=(300, 300),
                                                  mean=(104.0, 177.0, 123.0), swapRB=False, crop=False)
                # Predict face areas
                detector1.setInput(imageBlob)
                detections = detector1.forward()
                currenttime = time.time()
                box = detections[0, 0, 0, 3:7] * np.array([300, 300, 300, 300])
                box = box.astype('int')
                (startX, startY, endX, endY) = box
                if (currenttime - preticked > 0.1) and (detections[0, 0, 0, 2] > 0.9):
                    cv2.imwrite(os.path.join('/home/bau/Desktop/1', "%d.jpg" % count), frame)
                    count += 1
                    preticked = time.time()

                if count > 20:
                    # cv2.putText(frame, 'Đã nhập dữ liệu xong', (x01, y01), cv2.FONT_HERSHEY_SIMPLEX,
                    #             1, (0, 0, 255), 2, cv2.LINE_AA)
                    break
            ret, jpg = cv2.imencode('.jpg', frame)
        return jpg.tobytes()




