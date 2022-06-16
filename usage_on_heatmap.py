from re import U
from unittest import result
import cv2
from PIL import ImageGrab, Image
from cv2 import INPAINT_NS

import torch
import numpy as np
import argparse

import NeuralNets as NN
import util
from optical_flow import OpticalFlow
import time

parser = argparse.ArgumentParser(description="활용법")

parser.add_argument('--mode',  help='camera or screen')
parser.add_argument('--optFlow', default=True, help='use Optical Flow')

FONT=cv2.FONT_HERSHEY_SIMPLEX
YOUTUBE_GRAB_AREA = (0, 250, 1600, 1050)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

opticalFLow = None

torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

def setCamera():
    capture = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 512)
    
    return capture

def applyOpticalFlow(heatMap, frame, prevImg):
    canny = cv2.Canny(frame, 128, 205 )
    gray = cv2.cvtColor(heatMap, cv2.COLOR_BGR2GRAY)
    hotEdge = cv2.bitwise_and(gray, canny)
    
    cv2.imshow("hotEdge", hotEdge)
    opticalFLow.appendTrackPoint(point=[[10, 10], [20, 20], [30, 30], [40, 40], [50, 50]])
    result_img = opticalFLow.trace(prevImg, frame, drawOn=frame)
    return result_img

if __name__ == "__main__":
    args = parser.parse_args()

    model = NN.FocusNN()
    model = NN.load_model("./weights/model_focus3.ckpt", model)

    capture = None
    if args.mode == 'camera':
        capture = setCamera()
    if args.optFlow:
        opticalFLow = OpticalFlow(100, 10)
        prevImg = None

    while True:
        frame = np.zeros((NN.INPUT_SHAPE[1], NN.INPUT_SHAPE[0],3))

        if args.mode == 'camera':
            ret, frame = capture.read()
        elif args.mode == 'screen':
            screen = np.array(ImageGrab.grab(bbox = YOUTUBE_GRAB_AREA))
            frame = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, NN.INPUT_SHAPE)
        half_frame = cv2.resize(frame, (NN.INPUT_SHAPE[0]//2, NN.INPUT_SHAPE[1]//2))
        startTime = time.time()

        data = NN.img2Tensor(frame, device)
        out = model(data)[0].cpu().detach().numpy()[0]

        heatMap = util.cvt2Heatmap(out, threshold=50)
        heatMap = cv2.cvtColor(heatMap, cv2.COLOR_RGB2BGR)
        superImposedImg = cv2.addWeighted(half_frame, 0.8, heatMap, 0.6, 0)

        cv2.putText(superImposedImg, "FPS: {:.1f}".format(1/(time.time()-startTime)), (70, 50), FONT, 1, (255, 0, 0), 2)
        cv2.imshow("heatmap", superImposedImg)
        

        if args.optFlow:
            result_img = applyOpticalFlow(heatMap, half_frame, prevImg)    
            prevImg = half_frame

            # cv2.imshow("opticalImg", result_img)
        
        if(cv2.waitKey(1) & 0xFF == ord('q')):
            cv2.destroyAllWindows()
            break
