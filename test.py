import cv2
import numpy as np
import time

class Test(object):
    def __init__(self):
        super().__init__()
        # 選擇第二隻攝影機
        # self.cap = cv2.VideoCapture(0)
        self.cap = cv2.VideoCapture('test_oneminute.avi')
        self.cap.set(cv2.CAP_PROP_FPS, 20)
        # self.cap.set()
        self.fps_c = 0
        self.start_time = time.time()
        self.drop_c = 0
        self.steady_state = False
        self.drop = 0


    def fps(self):
        # if self.fps_c > 19:
        #     self.steady_state = False
        if (time.time()-self.start_time)>1:
            print('fps : ', self.fps_c)
            self.fps_c = 0
            self.start_time = time.time()
            self.drop = 0


    def run(self):
        tt=0
        # hr_pic = cv2.imread('hr_pic.png')
        # hr_pic = cv2.resize(hr_pic,(50,50))
        while(True):
            # 從攝影機擷取一張影像
            ret, frame = self.cap.read()

            if ret:
                self.steady_state = True
                self.fps()
                
                # 顯示圖片
                cv2.imshow('frame', frame)
                # cv2.waitKey(3)
                # print('drop <<<< 8')
                # 若按下 q 鍵則離開迴圈
                time.sleep(0.018)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                if not (tt%3 == 0):
                    self.fps_c += 1
                tt += 1
                # print('drop <<<< 8')
        
            else:
                break
                                      
        # 釋放攝影機
        self.cap.release()

        # 關閉所有 OpenCV 視窗
        cv2.destroyAllWindows()



import cv2
import time
def test01():
    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture('test_oneminute.avi')

    # Timeout to display frames in seconds
    # FPS = 1/TIMEOUT 
    # So 1/.025 = 40 FPS
    TIMEOUT = .025
    old_timestamp = time.time()

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        if (time.time() - old_timestamp) > TIMEOUT:
            # Display the resulting frame
            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            old_timestamp = time.time()

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    Test().run()
    # test01()
    a=0