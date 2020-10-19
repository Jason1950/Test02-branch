## python moudlue 
import cv2
import numpy as np
import time
import csv
import screeninfo
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning) 
warnings.filterwarnings("ignore", category=UserWarning) 

## J class
from J_device_v2 import VideoDevice
from J_dsp_v2 import Dsp
from J_face_v2 import Face

class HR(object):
    def __init__(self):
        #debug messanger !
        self.Full_Screen_Demo_state = False

        #video and excel save !
        # self.Csv_Save_State = False
        # self.Save_Video_File_state = False    
        self.Csv_Save_State = False
        self.Save_Video_File_state = False  

        #other state !
        self.Save_Csv_TimeStamp_state = True
        self.Debug_Spectrum_Info_state = False
        self.Debug_Info_Show_state = False

        self.camera_arg()
        self.full_screen_background(screen_num=1)
        self.buffer_size = 128
        self.sb_hr_buufer = []
        self.csv_save_file()

        # hr parameter
        self.temp_hr = [0]
        self.sb_hr_fft = 0
        self.sb_hr_peak = 0
        self.sb_pass = False
        self.temp_hr_state = False
        self.sb_stable_state = False
        self.psd_threshold = False

        #init class     
        self.face = Face()
        self.dsp = Dsp(self.buffer_size)   
        self.camera = VideoDevice(
                        self.arg,  
                        self.w, 
                        self.h, 
                        self.Save_Video_File_state )
        
        # self.camera_delay, self.out, self.out_feature = self.camera.webcam_start()
        self.camera_delay, self.out = self.camera.webcam_start()
        
        # FPS para
        self.temp_time = time.time()
        self.index = 0
        self.fps = 0
        self.fps_state = False       

    def camera_arg(self):
        ## webcam index self.arg : 0 internal ; 1 external ;
        self.arg = 0
        # self.arg = 'test_oneminute.avi'
        self.w = 640
        self.h = 480

    def full_screen_background(self, screen_num = 0):
        self.window_name = '640 x 480'
        if self.Full_Screen_Demo_state:
            # get the size of the screen
            self.window_name = 'Full Screen'
            self.screen = screeninfo.get_monitors()[screen_num]
            self.screen_w, self.screen_h = self.screen.width, self.screen.height

    def csv_save_file(self):
        if self.Save_Csv_TimeStamp_state:
            file_save_time = time.strftime("_%m%d_%H%M", time.localtime())
        else:
            file_save_time =''
        if self.Csv_Save_State:
            wfp = open("./excel_hr_record/output"+file_save_time+".csv", "w", newline='')
            self.writer = csv.writer(wfp)
            self.writer.writerow(['datetime','polar_hr','sstc_hr','max fft','sec fft','third fft','4th fft'])	
        else:
            self.writer = None    
    
    def J_fps(self):
        if abs(time.time()-self.temp_time) > 1:
            self.fps = self.index
            print(f'J fps : {self.fps}')
            self.fps_state = True
            self.index = 0 
            self.temp_time = time.time()

    def main(self):
        window_name = 'Heart Rate ' + self.window_name
        if self.Full_Screen_Demo_state:
            cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
            cv2.moveWindow(window_name, self.screen.x - 1, self.screen.y - 1)
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

   
        self.test_ii = 0
        while True:
            
            self.fps_state = False
            self.J_fps()

            frame, skip = self.camera.get_frame()
            frame_clean = frame.copy()

            if ((cv2.waitKey(1) & 0xFF == 27)
                or (frame is None)):
                self.camera.stop() 
                break

            if self.camera_delay:
                cv2.waitKey(int(1)) 
            else:
                if not skip :
                    # pass
                    continue
            
            # frame_show = frame.copy()
            frame, face_segment, face_x_y, dsp_state, fail_state = self.face.face_detect(frame)
            
            face_segment_copy = face_segment.copy()
            
            # frame, face_segment, face_x_y, dsp_state, fail_state = self.face.face_detect(frame[y1-redundary:y2+redundary,x1-redundary:x2+redundary])
            
            if self.fps_state:
                if len(self.temp_hr) < 15:
                    temp0 = 0 if np.isnan(round(np.mean(self.sb_hr_buufer))) else int(round(np.mean(self.sb_hr_buufer)))
                    self.temp_hr.append( temp0 ) 
                else: 
                    temp0 = 0 if np.isnan(round(np.mean(self.sb_hr_buufer))) else int(round(np.mean(self.sb_hr_buufer)))
                    # if abs(np.mean(self.temp_hr) - round(np.mean(self.sb_hr_buufer))) < 5:
                    self.temp_hr = self.temp_hr[1:]
                    self.temp_hr.append( temp0 ) 
                    if not(0 in self.temp_hr):
                        self.temp_hr_state = True
                    # print('temp_hr', self.temp_hr,np.mean(self.sb_hr_buufer) )
            ## original used : sb_hr_buufer !!
            if self.temp_hr_state:
                cv2.putText(frame, (f"  HR : {int(round(np.mean(self.temp_hr)))} " )
                        ,(face_x_y[0]-50,face_x_y[1]-25), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (100,0,255), 2, cv2.LINE_AA)
                out_orz = int(round(np.mean(self.temp_hr)))
            
            else:
                orz = round(np.mean(self.sb_hr_buufer))
                if not np.isnan(orz):
                      orz = int(orz)
                
                cv2.putText(frame, (f"  HR : {(orz)} " )
                                        ,(face_x_y[0]-50,face_x_y[1]-25), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (100,0,255), 2, cv2.LINE_AA)            
                out_orz = orz

            cv2.imshow(window_name, frame)

            ## all parameter be default !
            if not dsp_state:
                self.sb_hr_buufer = []
                self.temp_hr = [0]
                self.sb_hr_fft = 0
                self.sb_hr_peak = 0
                self.sb_pass = False
                self.sb_stable_state = False
                self.psd_threshold = False
                self.temp_hr_state = False
            
            ## ========================
            ##   Spatial Color 
            ## ========================
            rgb_state , spatial_rgb  = self.dsp.spatial_color(dsp_state, face_segment,fail_state)       
            #print(f'state : {rgb_state} , array shape : {spatial_rgb.shape} , array : ' )



            if rgb_state:
                
                ##
                ## test save video
                ##
                self.test_ii += 1
                cv2.putText(face_segment_copy, (f"frame : {(self.test_ii)}" )
                                            , (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
                if self.Save_Video_File_state:
                    self.out.write(frame_clean)
                    # self.out_feature(face_segment_copy)

                write_time = time.strftime("%m%d%H%M%S", time.localtime())

                ## ======================= 
                ##      SB algorithm !! 
                ## ======================= 
                # sb_array = self.dsp.SB(spatial_rgb,self.fps)
                sb_array,sb_hr0 = self.dsp.SB(spatial_rgb,self.fps) #fps=15)
                # sb_svg_array = self.dsp.SVG(sb_array)
                
                self.sb_hr_fft, self.sb_hr_peak, self.sb_pass , self.sb_total_hr, self.move_distance = self.dsp.sb_hr_FPP(
                                        sb_array.copy()
                                        , self.fps, 0.65, 10
                                        , self.Csv_Save_State
                                        , self.writer
                                        , write_time
                                        , str(out_orz)
                                        , face_x_y
                                        , self.Debug_Spectrum_Info_state  #) ## Info Show is test hr , max hr ,sec hr .... spectrum
                                        , sb_hr0 ) ## Info Show is test hr , max hr ,sec hr .... spectrum
                self.sb_stable_state = self.dsp.sb_hr_filter(self.sb_hr_buufer, round(self.sb_hr_fft,1), self.sb_pass, self.sb_total_hr)
            
                if self.fps_state and self.Debug_Info_Show_state:
                    print(f'fps {self.fps} , sb hr : {round(self.sb_hr_fft)} , {round(self.sb_hr_peak)} , pass: {self.sb_pass} 。  ')
                    #print(self.sb_hr_buufer)
            else:
                if self.fps_state and dsp_state:
                    print('Loading Data ...')

            self.index += 1
            ## ====================================
            ##   透過 face segment 做計算 :)  & END
            ## ====================================

if __name__ == "__main__":
    HR().main()