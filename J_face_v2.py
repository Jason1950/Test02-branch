
import cv2
import dlib
import numpy as np
import imutils
from imutils import face_utils

class Face(object):
    def __init__(self):
        #dlib init
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.fa = face_utils.FaceAligner(self.predictor, desiredFaceWidth=256)
        
        #state init
        self.box_state = False       # box init state ready ?
        self.detector_state = False  # dlib face detector ?
        self.fail_state = False      # dlib box ready but face detect fail !
        self.LM_detect_state = True  # tracking not fail and small landmark distance => change to tracking
        self.tracker_flow_state = False  # cv2 TrackerMedianFlow state

        #frame parameter
        self.frame_count = 0
        self.pos1 = [-1,-1] 
        self.pos2 = [-1,-1]

        #hyperpararmeter
        self.detect_LM_frame_face = 80
        self.detect_LM_frame_eyes = 40
        self.moving_detect_scalar = 5 # based on how many LM distance changing be regards as "subject starts to move"
        self.detect_face_when_moving = 5 # detect face per which frame when moving(LM_detect_state=1 means subject are moving)
        
        #optical flow track
        self.lk_params = dict( winSize  = (30,30),
                        maxLevel = 2,
                        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    def face_detect(self,frame):
        #init state and parameter
        self.fail_state = False 
        self.detect_face_when_moving = 5

        if frame is None:
            return 

        if not(self.box_state):
            self.initial_box(frame)
            landmark_crop = np.ones((480,640,3), dtype=np.uint8)
            # Run here then preesed 's'
            # Or run here after pressed 'q'!!     #print('init!')
        else:
            LM_68_for_track = self.start_box(frame)
            if LM_68_for_track is None:
                landmark_crop = frame
            else:
                landmark_crop = self.LM_to_skin(frame, LM_68_for_track, show=False)
            #print('start!')
            self.frame_count += 1
        cv2.rectangle(frame, (self.pos1[0],self.pos1[1]), (self.pos2[0],self.pos2[1]), (160,32,240), 2, 1)
        face_x_y = self.pos1
        # face_x_y = [round((self.pos1[0]+self.pos2[1])/2),self.pos1[1]]
        self.pos1, self.pos2 = [-1,-1], [-1,-1]
        # press q => cancel heart rate measurement
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.box_state = False
        return frame, landmark_crop, face_x_y, self.box_state, self.fail_state

    def initial_box(self,frame):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #it's truly gray frame!
        self.detector_state = False
        # detect faces in the grayscale image
        rects = self.detector(frame_gray, 0)
        for (i, rect) in enumerate(rects):
            # find face region
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            bbox = (x, y, w, h)
            self.detector_state = True	# face detected => detector_is_face set to 1
            

        # No face detect !!
        if not(self.detector_state):
            print('No faces\t\t','\t\t\r', end='')
            # no face detected => re-initialize parameters
            self.frame_count = 1
        
        # face detected
        else:	
            WaitKey = cv2.waitKey(1) & 0xFF
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)	# highlight face region
            # cv2.rectangle(frame, (x-50, y-50), (x + w+50, y + h+50), (150, 30, 130), 1)	# highlight face region
            face_rectangle = frame[int(y):int(y+h),int(x):int(x+w)]			# crop face region for landmarks detection
            # xddd = frame[int(y-50):int(y+h+50),int(x-50):int(x+w+50)].copy()	

            # cv2.rectangle(xddd, (50, 50), (50 + w, 50 + h), (0, 255, 0), 2)	# highlight face region
            # xddd = cv2.resize(xddd,(int(xddd.shape[1]*2), int(xddd.shape[0]*2)))
            # cv2.imshow('Scan Face Region', xddd)
            rect = dlib.rectangle(0, 0, int(face_rectangle.shape[0]), int(face_rectangle.shape[1]))	# transfer face rectangle to dlib format
            
            if WaitKey == ord('s') : #or flip_state:	# press s => cancel heart rate measurement
                print("ok s pressed")
                # face tracker initialization
                self.tracker = cv2.TrackerMedianFlow_create()
                self.tracker_flow_state = self.tracker.init(frame, bbox)
                # start to measure heart rate => re-initialize parameters    
                self.box_state = True       
                self.frame_count = 1     


            elif WaitKey == 27:
                return

        #self.frame_count = 1

    def start_box(self,frame):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.tracker_flow_state, bbox = self.tracker.update(frame)
        if ((self.frame_count%self.detect_face_when_moving == 0 
            and self.LM_detect_state) 
            or self.frame_count%200 == 0
            ):	# change to face detection
            rects = self.detector(frame_gray, 0)
            if len(rects):
                self.detector_state = True
                # find face region
                (x, y, w, h) = face_utils.rect_to_bb(rects[0])
                bbox = (x, y, w, h)
                self.tracker = cv2.TrackerMedianFlow_create() 
                self.tracker_flow_state = self.tracker.init(frame, bbox)
            else:
                self.detect_face_when_moving = 1
        # update face bounding box position with detection result
        (x, y, w, h) = bbox
        self.old_bbox = bbox
        face_rectangle = frame[int(y):int(y+h),int(x):int(x+w)]	
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rect = dlib.rectangle(int(x), int(y), int(x+w), int(y+h))
        #print('self.frame_count : ',self.frame_count)
        
        
        if (self.frame_count == 1):	# First frame => landmark detection
            LM_68 = self.predictor(frame_gray, rect)	# LM_68 landmark detection result
            LM_68 = face_utils.shape_to_np(LM_68)
            self.LM_68_for_track = np.vstack([LM_68[0:7,:], LM_68[7:10,:], LM_68[10:17,:], LM_68[17:27,:], LM_68[36:60,:]])	# Only these LM information need to track
            self.old_gray = frame_gray.copy()	# keep previous frame for landmark tracking
            self.LM_68_old = LM_68.copy()
            #print('LM_68_old : ',self.LM_68_old)
            print('LM_68_old copy ok !')  #[[245 426] ...... [262 424] [271 425] [281 424] [295 428] [280 426] [271 427] [262 425]]
        else:
            # Still keep detecting landmark to prevent landmark tracking fail
            LM_68 = self.predictor(frame_gray, rect)
            LM_68 = face_utils.shape_to_np(LM_68)	# dlib form to numpy
            self.LM_detect_state = True
            # ========================================
            # landmark default => detection
            # ========================================
            if (LM_68.shape[0]==68)and(self.LM_68_old.shape[0]==68):	# tracking not fail
                LM_68_distance_x = LM_68[:,0] - self.LM_68_old[:,0]
                LM_68_distance_y = LM_68[:,1] - self.LM_68_old[:,1]
                LM_68_distance = np.sqrt(np.square(LM_68_distance_x) + np.square(LM_68_distance_y))
                LM_distance = np.mean(LM_68_distance)
                if (LM_distance < self.moving_detect_scalar):
                    self.LM_detect_state = False	# tracking not fail and small landmark distance => change to tracking
                if (self.frame_count%(self.detect_LM_frame_face*2) == 0)and(self.frame_count%(self.detect_LM_frame_eyes*2) == 0):
                    self.LM_detect_state = True	# some condition still need detection
            
            self.LM_68_old = LM_68.copy()	# keep previous landmark result for landmark distance
            ## ==========================
            ##     Optical  flow   !!!
            ## ==========================
            #  landmark tracking
            p0_list_array = np.reshape(self.LM_68_for_track,[-1,1,2])
            p0_list_array = p0_list_array.astype(np.float32)
            p1, st, err = cv2.calcOpticalFlowPyrLK(self.old_gray, frame_gray, p0_list_array, None, **self.lk_params)
            
            if p0_list_array.shape[0]:
                self.LM_68_for_track = p1.reshape(51, 2)
                if (self.frame_count%self.detect_LM_frame_face == 0)and(self.LM_detect_state):	# use landmark detection for face region
                    print("DETECT FACE")
                    self.LM_68_for_track[0:7,:] = LM_68[0:7,:]
                    self.LM_68_for_track[7:10,:] = LM_68[7:10,:]
                    self.LM_68_for_track[10:17,:] = LM_68[10:17,:]
                    self.LM_68_for_track[17:27,:] = LM_68[17:27,:]
                    # cv2.circle(frame,(30, 200), 10, (0, 255, 255), -1)
                
                if (self.frame_count%self.detect_LM_frame_eyes == 0)and(self.LM_detect_state):	# use landmark detection for eyes & mouth region
                    print("DETECT EYES AND MOUTH")
                    self.LM_68_for_track[27:33,:] = LM_68[36:42,:]
                    self.LM_68_for_track[33:39,:] = LM_68[42:48,:]
                    self.LM_68_for_track[39:51,:] = LM_68[48:60,:]
                    # cv2.circle(frame,(60, 200), 10, (0, 255, 255), -1)
                good_old = p0_list_array[st==1]
            
            if self.tracker_flow_state:
                #print('ok : ',self.tracker_flow_state)
                # Tracking success
                #p1 = (int(bbox[0]), int(bbox[1]))
                #p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                self.pos1[0],self.pos1[1] = int(bbox[0]), int(bbox[1])
                self.pos2[0],self.pos2[1] = int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])
        
            else :
                # Face bounding box tracking failure => hold previous bounding box result
                print("TRACK FAIL!!")
                # Keep previous face bounding box result
                #p1 = (int(self.old_bbox[0]), int(self.old_bbox[1]))
                #p2 = (int(self.old_bbox[0] + self.old_bbox[2]), int(self.old_bbox[1] + self.old_bbox[3]))
                self.pos1[0],self.pos1[1] = int(self.old_bbox[0]), int(self.old_bbox[1])
                self.pos2[0],self.pos2[1] = int(self.old_bbox[0] + self.old_bbox[2]), int(self.old_bbox[1] + self.old_bbox[3])
                self.fail_state = True
            
            ## ===================
            ## show frame
            ## ===================
            #cv2.imshow('J_FACe',frame)    
            if not(self.LM_68_for_track.shape[0] == 68) or self.LM_detect_state:	# landmark track fail => change to detection
                self.LM_68_for_track = np.vstack([LM_68[0:7,:], LM_68[7:10,:], LM_68[10:17,:], LM_68[17:27,:], LM_68[36:60,:]])
            #cv2.rectangle(frame, p1, p2, (160,32,240), 2, 1)
        
        return self.LM_68_for_track 

    def LM_to_skin(self,frame, LM_68_for_track, face_shrink=5, show=False):
        landmark_mask = np.zeros(frame.shape, dtype=np.uint8)	
        face_crop = np.concatenate([LM_68_for_track[0:7,:]+np.array([face_shrink,-face_shrink]), LM_68_for_track[7:10,:]
                                    +np.array([0,-face_shrink]), LM_68_for_track[10:17,:]
                                    +np.array([-face_shrink,-face_shrink]), LM_68_for_track[26:16:-1,:]], axis=0)
        cv2.fillPoly(landmark_mask, np.int32([face_crop]), (255,255,255))
        # exclude eyes
        eyes_crop_1 = LM_68_for_track[27:33,:]
        eves_crop_2 = LM_68_for_track[33:39,:]
        # eyes_crop_1 = LM_68_for_track[36:42,:]
        # eves_crop_2 = LM_68_for_track[42:48,:]

        cv2.fillPoly(landmark_mask, np.int32([eyes_crop_1]), (0,0,0))
        cv2.fillPoly(landmark_mask, np.int32([eves_crop_2]), (0,0,0))
        # exclude mouth
        mouth_crop = LM_68_for_track[39:51,:]
        # mouth_crop = LM_68_for_track[48:60,:]
        
        cv2.fillPoly(landmark_mask, np.int32([mouth_crop]), (0,0,0))
        landmark_crop = cv2.bitwise_and(frame, landmark_mask)


        landmark_mask_jason = np.zeros(frame.shape, dtype=np.uint8)
        # print('[26:16:-1] : ',LM_68_for_track[26:16:-1,:])
        # print('[0:7,:]  : ',LM_68_for_track[0:7,:],'  , [0:7,:]+np[5,-5] :  ',LM_68_for_track[0:7,:]+np.array([face_shrink,-face_shrink]))
        # print('[40,:]  : ',LM_68_for_track[40,:],'  , [40,:]+np[5,-5] :  ',LM_68_for_track[40,:]+np.array([face_shrink,-face_shrink]))

        triangle_crop = np.concatenate(
                            [LM_68_for_track[32:33,:]+np.array([face_shrink,face_shrink*2]),           # left eye
                            LM_68_for_track[33:34,:]+np.array([face_shrink*2,face_shrink*2]),          # right eye
                            
                            LM_68_for_track[45:46,:]+np.array([face_shrink,-face_shrink*2]),           # right mouth
                            LM_68_for_track[39:40,:]+np.array([-face_shrink-2,-face_shrink*2])], axis=0)   # left mouth
        cv2.fillPoly(landmark_mask_jason, np.int32([triangle_crop]), (255,255,255))
        cv2.fillPoly(landmark_mask_jason, np.int32([eyes_crop_1]), (0,0,0))
        cv2.fillPoly(landmark_mask_jason, np.int32([eves_crop_2]), (0,0,0))
        cv2.fillPoly(landmark_mask_jason, np.int32([mouth_crop]), (0,0,0))
        landmark_crop_jason = cv2.bitwise_and(frame, landmark_mask_jason)

        if show:
            cv2.imshow('original', landmark_mask)
            landmark_crop2 = cv2.cvtColor(landmark_crop,cv2.COLOR_BGR2GRAY)
            cv2.waitKey(1)

        cv2.imshow('ee',landmark_crop)
        return landmark_crop

        # cv2.imshow('ww',landmark_crop_jason)
        # return landmark_crop_jason