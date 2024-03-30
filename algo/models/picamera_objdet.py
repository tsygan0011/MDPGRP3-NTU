from picamera2 import Picamera2, Preview
from objectdetection_yolov8 import detect
from time import sleep
from datetime import datetime

# picam2 = Picamera2()
# #camera_config = picam2.create_still_configuration(main={"size":(1920,1080)}, lores={"size":(640,480)}, display="lores")
# #picam2.start_preview(Preview.QTGL)
# picam2.start()
# sleep(0.5)
# picam2.capture_file(r"/home/pi/MDP/src/camera2/test.jpg")

# print("img captured")


# #camera.start_preview()
# sleep(0.5)
# now = datetime.now()
#path = f"camera/{now}.jpg"
#path = f"./{now}.jpg"
path = r"/home/pi/MDP2/src/camera2/test_sensor2_1.jpg"
#camera.capture(path)
preds = detect(path)
print(preds)
#camera.stop_preview()

