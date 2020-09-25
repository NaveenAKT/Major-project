import darknet as yolo
import numpy as np
import cv2
import dlib
import random
import HornSchunck as hs
from vif import ViF
import glob, os
import pickle

clf = pickle.load(open('models/model-svm.sav', 'rb'))
total_frames = []
sub_sampling = 29
net = yolo.load_net("/home/naveenkumar/Downloads/car-crash/darknet/cfg/yolov3.cfg",
              "/home/naveenkumar/Downloads/car-crash/darknet/cfg/yolov3.weights", 0)
meta = yolo.load_meta("/home/naveenkumar/Downloads/car-crash/darknet/data/coco.data")
counter_sub_video = 1
data = []


class Tracker:
    def __init__(self, frame, (xmin, ymin, xmax, ymax), name, frame_index):
        self.tracker = dlib.correlation_tracker()
        self.tracker.start_track(frame, dlib.rectangle(xmin, ymin, xmax, ymax))
        self.name = name
        self.frame_index = frame_index
        self.history = [dlib.rectangle(xmin, ymin, xmax, ymax)]
        self.flow_vectors = []

    def update(self, frame):
        self.tracker.update(frame)
        return self.tracker.get_position()

    def get_position(self):
        return self.tracker.get_position()

    def add_history(self, pos):
        self.history.append(pos)


    def get_box_from_history(self, frame_width, frame_height):
        
        xmin = self.history[0].left()
        ymin = self.history[0].top()
        xmax = self.history[0].right()
        ymax = self.history[0].bottom()
        for pos in self.history:
            if pos.left() < xmin:
                xmin = pos.left()
            if pos.right() > xmax:
                xmax = pos.right()
            if pos.top() < ymin:
                ymin = pos.top()
            if pos.bottom() > ymax:
                ymax = pos.bottom()

        if xmin < 0:
            xmin = 0
        if ymin < 0:
            ymin = 0
        if xmax > frame_width:
            xmax = frame_width - 1
        if ymax > frame_height:
            ymax = frame_height - 1


        return (int(xmin), int(ymin), int(xmax), int(ymax))

    def add_vector(self, line):
        self.flow_vectors.append(line)

    def clean_flow_vector(self):
        self.flow_vectors = []

    def is_inside(self, line):
        pos = self.get_position()
        
        if line[0] > pos.left() and line[1] > pos.top() and line[0] < pos.right() and line[1] < pos.bottom():
            return True
        
        elif line[2] > pos.left() and line[3] > pos.top() and line[2] < pos.right() and line[3] < pos.bottom():
            return True
        else:
            return False



def vif(trackers,  frame_width, frame_height, frame):
    global sub_sampling
    print ("processing ViF on each tracker")
    global counter_sub_video
    for i, tracker in enumerate(trackers):
        print("processing ViF on  tracker " + str(tracker.name), tracker.get_position().right() - tracker.get_position().left(), tracker.get_position().bottom() - tracker.get_position().top())

        box = tracker.get_box_from_history(frame_width, frame_height)


        if box[2] - box[0] < 100:
            print ("very small tracker dimensions")
            continue



        print ("tracker frame_index:", tracker.frame_index, "len history:", len(tracker.history))
        if len(tracker.history) < sub_sampling:
            print ("tracker with few frames, ignore")
        else:
 
            print ("the video will be saved as: ", str(counter_sub_video), (box[2] - box[0], box[3] - box[1]))
            
            counter_sub_video += 1

            tracker_frames = []

            for j in range(tracker.frame_index, tracker.frame_index + len(tracker.history) ):

                img = total_frames[j]
                sub_image = img[box[1]:box[3], box[0]:box[2]]
                gray_image = cv2.cvtColor(sub_image, cv2.COLOR_BGR2GRAY)
                tracker_frames.append(gray_image)

              
                cv2.imshow("sub_image", sub_image)
                cv2.waitKey(0)
                #out.write(sub_image)


            print ("the tracker has  " + str(len(tracker_frames)) + " frames")
            # procesing vif
            obj = ViF()
            feature_vec = obj.process(tracker_frames)
            data.append(feature_vec)


            feature_vec = feature_vec.reshape(1, 304)
            print (feature_vec.shape)
            result = clf.predict(feature_vec)
            print "RESULT SVM", result
            font = cv2.FONT_HERSHEY_SIMPLEX
            print("RESULT ", result[0])
            if result[0] == 0.0:
                print(0)
                title = "normal"
            else:
                print(1)
                title = "car-crash"
                overlay = frame.copy()
                cv2.rectangle(overlay, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), -1)
                opacity = 0.4
                cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0, frame)

            cv2.imshow("win", frame)
            cv2.waitKey(0)



def start_process(path, net, meta):
    global total_frames
    print("reading video " + path)
    total_frames = []

    cap = cv2.VideoCapture(path)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    print(str(frame_count) + " frames y " + str(fps) + " as framerate")

    index = 0

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    
    detections = 0
    trackers = []

    while True:

        ret, frame = cap.read()
	print(cap)
        if ret:
            new_frame = frame.copy()
            total_frames.append(new_frame)

            if index > 0 and (index % sub_sampling == 0 or index == frame_count - 1):
                print ("FRAME " + str(index) + " VIF")
                vif(trackers, frame_width, frame_height, frame)

            if index % sub_sampling == 0 or index == 0 :
                print ("FRAME " + str(index) + " YOLO")
                trackers = []

                cv2.imwrite("tmp/img.jpg", frame)
                detections = yolo.detect(net, meta, "/home/naveenkumar/Downloads/car-crash/tmp/img.jpg")
                print (detections)
                for det in detections:
                    label = det[0]
                    accuracy = det[1]
                    box = det[2]

                    width = int(box[2])
                    height = int(box[3])
                    xmin = int(box[0]) - width / 2
                    ymin = int(box[1]) - height / 2

                    if label == 'car':
                        cv2.rectangle(frame, (xmin, ymin), (xmin + width, ymin + height), (0, 0, 255))

                        if xmin + width < frame_width and ymin + height < frame_height:
                            tr = Tracker(frame, (xmin, ymin, xmin + width, ymin + height), random.randrange(100), index)
                            trackers.append(tr)
                        else:
                            if xmin + width < frame_width and ymin + height >= frame_height:
                                tr = Tracker(frame, (xmin, ymin, xmin + width, frame_height - 1), random.randrange(100),
                                             index)
                            elif xmin + width >= frame_width and ymin + height < frame_height:
                                tr = Tracker(frame, (xmin, ymin, frame_width - 1, ymin + height), random.randrange(100),
                                             index)
                            else:
                                tr = Tracker(frame, (xmin, ymin, frame_width - 1, frame_height - 1),
                                             random.randrange(100), index)
                            trackers.append(tr)

            else:
                print ("FRAME " + str(index) + " UPDATE TRACKER")

                # update trackers
                for i, tracker in enumerate(trackers):
                    tr_pos = tracker.update(frame)
                    if tr_pos.left() > 0 and tr_pos.top() > 0 and tr_pos.right() < frame_width and tr_pos.bottom() < frame_height:
                        cv2.rectangle(frame, (int(tr_pos.left()), int(tr_pos.top())),
                                      (int(tr_pos.right()), int(tr_pos.bottom())), (0, 0, 255))
                        tracker.add_history(tr_pos)


            cv2.imshow("win", frame)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break

            index += 1


        else:
            break

    cv2.destroyAllWindows()
def ccw(A,B,C):
	return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])
def intersect(A,B,C,D):
	return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)
start_process("choque10.mp4", net, meta)

