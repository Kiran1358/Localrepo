
import cv2

fac_data = cv2.CascadeClassifier("C:/Users/shett/AppData/Local/Programs/Python/Python313/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")

video_enable = cv2.VideoCapture(0)

while True:
    hit, video_data = video_enable.read()

    if not hit:
        break
    color = cv2.cvtColor(video_data, cv2.COLOR_BGR2GRAY)



    faces = fac_data.detectMultiScale(
        color,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(video_data, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Video_Recording", video_data)

    if cv2.waitKey(10) == ord("k"):
        break

video_enable.release()
