import dlib
import cv2
import imutils

def d55():
    path = './face/'

    # 讀取照片圖檔
    # img = cv2.imread(path+'1 (6).png')
    # img = cv2.imread(path+'288055.jpg')

    img = cv2.imread(path+'288067.jpg')

    # 縮小圖片
    img = imutils.resize(img, width=1280)

    # Dlib 的人臉偵測器
    detector = dlib.get_frontal_face_detector()

    # 偵測人臉
    face_rects = detector(img, 0)

    # 取出所有偵測的結果
    for i, d in enumerate(face_rects):
        x1 = d.left()
        y1 = d.top()
        x2 = d.right()
        y2 = d.bottom()

    # 以方框標示偵測的人臉
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 4, cv2.LINE_AA)

    # 顯示結果
    cv2.imshow("Face Detection", img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
import numpy as np
import heapq
def e55():
    nums = [1,8,2,23,7,-4,18,23,23,37,2]
    # nums = [1,8,2,23,7,-4,18,23,24,37,2]
    result = list(map(nums.index, heapq.nlargest(3, nums)))
    temp=[]
    Inf = -np.Inf
    for i in range(3):
        temp.append(nums.index(max(nums)))
        nums[nums.index(max(nums))]=Inf
    result.sort()
    temp.sort()
    print(result)
    print(temp) 

e55()