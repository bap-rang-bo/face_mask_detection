import cv2
import dlib

# Load the detector
detector = dlib.get_frontal_face_detector()

# Load the predictor
predictor = dlib.shape_predictor("/home/bau/Desktop/thay_lam/face_recognition/shape_predictor_68_face_landmarks.dat")

# read the image
cap = cv2.VideoCapture(0)
a = 0
while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    # Convert image into grayscale
    gray = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2GRAY)

    # Use detector to find landmarks
    faces = detector(gray)

    for face in faces:
        x01 = face.left()  # left point
        y01 = face.top()  # top point
        x02 = face.right()  # right point
        y02 = face.bottom()  # bottom point
        #         cv2.rectangle(img=frame, pt1=(x01, y01), pt2=(x02, y02), color=(0, 255, 0), thickness=4)

        # Create landmark object
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

        # Draw a circle

        if y0 < y17 or y0 > y28 or y0 == y33 or y16 < y26 or x8 < x37 or x8 > x44 or d1 * 2 < d2 or d1 > d2 * 2 or y27 > y1 or y27 > y15:
            cv2.putText(frame, 'K chap nhan', (x01, y01), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, 'Chap nhan', (x01, y01), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2, cv2.LINE_AA)
    # show the image
    cv2.imshow(winname="Face", mat=frame)

    # Exit when escape is pressed
    if cv2.waitKey(delay=1) == 27:
        break

# When everything done, release the video capture and video write objects
cap.release()

# Close all windows
cv2.destroyAllWindows()