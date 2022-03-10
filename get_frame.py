import cv2
import os
import argparse

#read_path = "/home/compu/ymh/drawing/dataset/num_006/yerin.mp4"
#save_path = "/home/compu/ymh/drawing/dataset/num_006/yerin_frame/"

#read_path = "C://BBox-Label-Tool//Videos//002.mp4"
#save_path = "C://BBox-Label-Tool//Images//002//"

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='Input Path')
    parser.add_argument('--output', type=str, help='Output Path')

    args = parser.parse_args()


    vidcap = cv2.VideoCapture(args.input)
    success, image = vidcap.read()
    count = 0
    while success:
        cv2.imwrite(os.path.join(args.output, "frame_%07d.png" % count), image)
        print("Frame %d is saved." % count)
        success, image = vidcap.read()
        #print('Read a new frame: ', success)
        count += 1