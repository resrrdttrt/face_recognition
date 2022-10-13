import cv2
import numpy as np
import face_recognition
import os
import time
import mediapipe as mp
import threading


class faceDetector():
    def __init__(self, minDet=0.5, model=0):
        self.min_detection_confidence = minDet
        self.model_selection = model
        self.mpFaces = mp.solutions.face_detection
        self.mpDraws = mp.solutions.drawing_utils
        self.faces = self.mpFaces.FaceDetection(self.min_detection_confidence, self.model_selection)
        self.result = None
        self.faceLoc = None

    def face_detect(self, imgTest):
        imgRGB = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)
        self.result = self.faces.process(imgRGB)
        if self.result.detections:
            for id, detection in enumerate(self.result.detections):
                bboxC = detection.location_data.relative_bounding_box
                h, w, c = imgTest.shape
                bbox = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
                # fancy draw
                cv2.line(imgTest, (bbox[0], bbox[1]), (bbox[0] + bbox[2] // 4, bbox[1]), (255, 0, 0), 3)
                cv2.line(imgTest, (bbox[0], bbox[1]), (bbox[0], bbox[1] + bbox[2] // 4), (255, 0, 0), 3)

                cv2.line(imgTest, (bbox[0], bbox[1] + bbox[2]), (bbox[0] + bbox[2] // 4, bbox[1] + bbox[2]),
                         (255, 0, 0), 3)
                cv2.line(imgTest, (bbox[0], bbox[1] + bbox[2]), (bbox[0], bbox[1] + bbox[2] - bbox[2] // 4),
                         (255, 0, 0), 3)

                cv2.line(imgTest, (bbox[0] + bbox[2], bbox[1]), (bbox[0] + bbox[2] - bbox[2] // 4, bbox[1]),
                         (255, 0, 0), 3)
                cv2.line(imgTest, (bbox[0] + bbox[2], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[2] // 4),
                         (255, 0, 0), 3)

                cv2.line(imgTest, (bbox[0] + bbox[2], bbox[1] + bbox[2]),
                         (bbox[0] + bbox[2], bbox[1] + bbox[2] - bbox[2] // 4),
                         (255, 0, 0), 3)
                cv2.line(imgTest, (bbox[0] + bbox[2], bbox[1] + bbox[2]),
                         (bbox[0] + bbox[2] - bbox[2] // 4, bbox[1] + bbox[2]),
                         (255, 0, 0), 3)
                cv2.rectangle(imgTest, bbox, (255, 0, 0), 1)
                self.faceLoc = [(bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3], bbox[0])]

        return imgTest



def decode_frame():
    while True:
        if detector.faceLoc:
            print("Start decode")
            faceEncodeTest = face_recognition.face_encodings(frame, known_face_locations=detector.faceLoc)[0]
            for i in range(len(encodeList)):
                print(memberList[i])
                print(face_recognition.face_distance(encodeList[i], faceEncodeTest))
                print(face_recognition.compare_faces(encodeList[i], faceEncodeTest, tolerance=0.4))
            time.sleep(5)


if __name__ == '__main__':

    file_path = 'Image/Faces/Data'
    memberList = os.listdir(file_path)
    imgList = [[] for i in range(len(memberList))]
    encodeList = [[] for i in range(len(memberList))]

    for directory in memberList:
        imgPathList = os.listdir(f'{file_path}/{directory}')
        for imgPath in imgPathList:
            img = cv2.imread(f'{file_path}/{directory}/{imgPath}')
            imgList[memberList.index(directory)].append(img)
            faceEncoding = face_recognition.face_encodings(img)[0]
            encodeList[memberList.index(directory)].append(faceEncoding)
    print("Encode completed")

    detector = faceDetector()
    vid = cv2.VideoCapture(0)
    preTime = 0
    decode_thread = threading.Thread(target=decode_frame,daemon=True)
    decode_thread.start()
    while True:
        isTrue, unflip_frame = vid.read()
        frame = cv2.flip(unflip_frame, 1)

        curTime = time.time()
        fps = int(1 / (curTime - preTime))
        preTime = curTime
        cv2.putText(frame, f'FPS:{fps}', (20, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 1)
        frame = detector.face_detect(frame)
        cv2.imshow('Title', frame)
        if cv2.waitKey(1) == 27:
            break
    vid.release()
    cv2.destroyAllWindows()
