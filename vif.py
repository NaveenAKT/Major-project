import numpy as np
import cv2
import HornSchunck as hs
import math
import glob, os

class ViF:
    def __init__(self):
        self.video_subsampling = 3

    def create_block_hist(self, flow, N, M):
        # creamos los histogramas por bloques
        height, width = flow.shape
        B_height = int(math.floor((height - 11) / N))
        B_width = int(math.floor((width - 11 ) / M))

        frame_hist = []
        

        for y in np.arange(6, height - B_height - 5, B_height):
            for x in np.arange(6, width - B_width - 5, B_width):
                block_hist = self.create_hist(flow[y:y+B_height-1, x:x+B_width-1])
                #print(block_hist)
                frame_hist.append(block_hist)

        return np.array(frame_hist).flatten()

    def create_hist(self, mini_flow):
        # print(mini_flow)
        H = np.histogram(mini_flow, np.arange(0, 1, 0.05))
        H = H[0]/float(np.sum(H[0]))
        return H

    def process(self, frames):
        

        flow = np.zeros([100, 134]) #rows cols
        index = 0
        N = 4
        M = 4

        for i in range(0, len(frames) - self.video_subsampling - 5, self.video_subsampling*2):

            index += 1

            prev_f = frames[i + self.video_subsampling]
            curr_f = frames[i + self.video_subsampling * 2]
            next_f = frames[i + self.video_subsampling * 3]

            prev_f = cv2.resize(prev_f, (134, 100)) #width height
            curr_f = cv2.resize(curr_f, (134, 100))
            next_f = cv2.resize(next_f, (134, 100))
            u1, v1, m1 = hs.HornSchunck(prev_f, curr_f)
            u2, v2, m2 = hs.HornSchunck(curr_f, next_f)

            delta = abs(m1 - m2)
            flow = flow + (delta > np.mean(delta))

        flow = flow.astype(float)
        if index > 0:
            flow = flow/index

        feature_vec = self.create_block_hist(flow, N, M)
        return feature_vec
