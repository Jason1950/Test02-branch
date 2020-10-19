import cv2
import numpy as np
import time
import pyrealsense2 as rs

class VideoDevice(object):
    def __init__(self, cap_arg, w, h, save_state):
        self.save_state = save_state
        self.webcam_init(cap_arg)
        
        
    def webcam_init(self, cap_arg):
        #print ("WebCamEngine init")
        self.dirname = "" #for nothing, just to make 2 inputs the same
        self.cap = None
        self.cap_arg = cap_arg
        self.frame_count = 0
        self.webcam_delay = False
    

    def get_frame(self):
        frame = self.webcam_get_frame()
        if not(self.frame_count%3 == 0):
            return frame ,True
        else: 
            return frame ,False


    def stop(self):
        if self.cap is not None:
            self.cap.release()
            cv2.destroyAllWindows()
            print("[INFO] Stop Webcam")

        


    ## ======================================
    ##   webcam and realsense function !!
    ## ======================================
    def webcam_start(self):
        print("[INFO] Start Webcam")
        #time.sleep(1) # wait for camera to be ready
        if len(str(self.cap_arg) ) > 1:
            self.cap = cv2.VideoCapture(self.cap_arg)
            self.webcam_delay = True
        else:
            self.cap = cv2.VideoCapture(self.cap_arg, cv2.CAP_DSHOW)
            self.cap.set(cv2.CAP_PROP_FPS, 60)
            self.webcam_delay = False
        self.webcam_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.valid = False
        # 720P -> 1280×720
        self.cap_width = 640*1
        self.cap_height = 480*1
        # 設定擷取影像的尺寸大小
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cap_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cap_height)
        self.out = 0
        self.out_feature = 0
        if self.save_state:
            fourcc = cv2.VideoWriter_fourcc(*'XVID') # 使用 XVID 編碼
            file_save_time = time.strftime("_%m%d_%H%M", time.localtime())
            self.out = cv2.VideoWriter('output_'+file_save_time+'.avi', fourcc, 30.0, (int(self.cap_width) ,int(self.cap_height)))
            # self.out_feature = cv2.VideoWriter('output_feature_'+file_save_time+'.avi', fourcc, 30.0, (int(self.cap_width) ,int(self.cap_height)))
        try:
            resp = self.cap.read()
            self.shape = resp[1].shape
            self.valid = True
        except:
            self.shape = None
        return self.webcam_delay, self.out #, self.out_feature



    def webcam_get_frame(self):
        self.frame_count +=1

        if self.valid:
            _,frame = self.cap.read()
            frame = cv2.flip(frame,1)

            # if self.save_state:
            #     self.out.write(frame)
        else:
            #frame = np.ones((480,640,3), dtype=np.uint8)
            frame = np.ones((self.cap_height,self.cap_width,3), dtype=np.uint8)
            col = (0,256,256)
            cv2.putText(frame, "(Error: Camera not accessible)",
                       (65,220), cv2.FONT_HERSHEY_PLAIN, 2, col)
        '''if not self.webcam_delay:
            if not(self.frame_count%3 == 0):
                return frame
            else:
                return 
        else:
            return frame'''
        return frame



        