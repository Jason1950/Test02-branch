import cv2
import numpy as np
import sys
import time
from scipy import signal
from  scipy.signal import savgol_filter as sg_filter
import heapq

class Dsp(object):
    def __init__(self, size):
        self.size = size
        self.pca_size = 0
        self.temp = np.zeros([0,3])
        self.temp2 = np.zeros([0,3])
        self.rgb_array = np.empty([0,3])
        self.pca_array = np.empty([0,self.pca_size])
        self.rgb_state = False
        self.pca_state = False
        self.face_xy = [0,0]
        pass

    def spatial_color(self, dsp_state, feature, fail_state ):
        if dsp_state:
            # rPPG algorithm preparation
            # frame is BGR (original is RGB but covert!)
            b = feature[:,:,0] 
            g = feature[:,:,1]
            r = feature[:,:,2]
            
            # face_b = b[b > 40]
            # face_g = g[g > 40]
            # face_r = r[r > 40]

            face_b = b[b!=0]
            face_g = g[g!=0]
            face_r = r[r!=0]

            # face_b = face_b[face_b>20]
            # face_g = face_g[face_g>20]
            # face_r = face_r[face_r>20]            
            

            mean_b = np.mean(face_b)
            mean_g = np.mean(face_g)
            mean_r = np.mean(face_r)

            if (fail_state 
                or np.isnan(mean_b) 
                or np.isnan(mean_g) 
                or np.isnan(mean_r)):
                #print('fail temp : ',self.temp)
                self.temp = self.temp2.copy()
            else:
                #print('temp : ',self.temp)
                self.temp = np.array([mean_r, mean_g , mean_b])
                self.temp2 = self.temp.copy()

            if len(self.rgb_array) < self.size  :
                #print('len : ',len(self.rgb_array))
                self.rgb_array = np.vstack([self.rgb_array,self.temp])
            else:
                self.rgb_state = True
                self.rgb_array = np.vstack([self.rgb_array[1:,:],self.temp])
                #self.rgb_array.append(self.temp)
        else:
            ## para and array data clean !
            self.pca_state = False
            self.rgb_state = False
            self.temp = np.zeros([0,3])
            self.temp2 = np.zeros([0,3])
            self.rgb_array = np.empty([0,3])
            self.pca_array = np.empty([0,self.pca_size])

        return self.rgb_state, self.rgb_array.copy().transpose()

    def SB(self, spatial_rgb, fps=20):
        # SB algorithm para
        POS_matrix = np.array([[0,1,-1],[-2,1,1]])

        # SB algorithm main
        rgb_diag = np.diag(np.mean(spatial_rgb,1))
        # print(f' rgb_diag shape : {rgb_diag.shape}')
        rgb_inv = np.linalg.inv(rgb_diag)  # inverse the martix
        # print(f' rgb_inv shape : {rgb_inv.shape}')
        Cn = np.dot(rgb_inv, spatial_rgb) - 1	 # Cn size = 3*128
        F = np.fft.fft(Cn)	# F size = 3*128
        S = np.dot(POS_matrix, F)  # S size = 2*128
        Z = S[0] + (abs(S[0]) / abs(S[1])) * S[1] # Z size = 1*128
        Zn = Z * (abs(Z) / abs(sum(F[0])))
        # Zn[:round(self.size/fps)+1] = 0     # paper is 6 
        # Zn[round(self.size/fps)*4-2:] = 0 	# paper is 24	
        Zn[:6] = 0
        Zn[24:] = 0
        # print('Zn size : ', len(Zn), ' , Zn : ', Zn)
        Znc = Zn.copy()
        Znc0 = np.abs(Znc)
        # Znc1 = 
        # print('Znc0 size : ', len(Znc0), ' , Znc0 : ', Znc0)
        arr_size = 18
        templist = self.Zn_detect_peak(Znc0.copy(),arr_size)
        # maxarg()
        Znc_index = np.argmax(Znc0)
        # SB_hr_c = 20/128 * Znc_index
        # print('SB no rfft hr : ', round(SB_hr_c*60,1), ' , index : ', Znc_index,' , len of Znc : ', len(Znc0[Znc0>0]))
        SB_hr_c = []
        for i in range(arr_size):
            hr_tmep = 0
            hr_tmep = 20/128 *templist[i]
            SB_hr_c.append(round(hr_tmep*60,1))
        # print('SB_hr_c : ', SB_hr_c)
        Pn = np.real(np.fft.ifft(Zn)) 
        P = (Pn - np.mean(Pn)) / np.std(Pn)
        return P,SB_hr_c[0]
        # return P

    def SVG(self, data):
        x_1 = data
        x_1 = x_1
        s_1 = sg_filter(x_1,125,6)
        x_2 = x_1 - s_1
        s_2 = sg_filter(x_2,45,2)
        x_3 = x_2 - s_2
        s_3 = sg_filter(x_3,29,5)
        x_4 = x_3 - s_3
        s_4 = sg_filter(x_4,15,2)
        data2 = 0.0*x_1 + 0.0*x_2 + 0.75*x_3 + 0.25*x_4        
        return data2

    def Zn_detect_peak(self,Zn_list,arr_size=4):
        # Zn_list
        # result = map(Zn_list.index, heapq.nlargest(4, Zn_list))
        # temp=[]
        # Inf = 0
        # for i in range(3):
        #     temp.append(nums.index(max(nums)))
        #     nums[nums.index(max(nums))]=Inf
        # temp.sort()
        Zn_list = Zn_list.tolist()
        temp=[]
        
        for i in range(arr_size):
            temp.append(Zn_list.index(max(Zn_list)))
            Zn_list[Zn_list.index(max(Zn_list))]=-np.Inf
        # temp.sort()

        return temp

    def detect_spec_number(self, f, x, limit, show = False):
        x = (x-np.min(x)) / (np.max(x)- np.min(x))
        #this is ok!!!!!!!!
        #peaks, _ = find_peaks(x,distance=10) # height = self.peak_height_limit)
        peaks, _ = signal.find_peaks(x,height =0.2 ,distance= 3 )
        hr = []
        hr_new = []
        hr_spec_value_new = []
        hr_spec_value = []
        if len(peaks) < 4:
            range_peak = len(peaks)
        else:
            range_peak = 4
        
        for i in range(range_peak):
            hr_spec_value.append(round(x[peaks[i]],3))
            hr.append(round(f[peaks[i]],1))
        x_pos = np.argmax(x)

        # A 排序的依據 B是被變動的
        # https://blog.csdn.net/qq_17753903/article/details/82634146
        if show:
            print('test hr_spec_value',hr_spec_value)
            print('test hr',hr)
        if len(hr_spec_value) > 0:
            Z = zip(hr_spec_value,hr)
            # print('test Z',Z) # <zip object at 0x000001D5258094C8>
            Z = sorted(Z,reverse=True)
            hr_spec_value_new, hr_new = zip(*Z)
        else: 
            hr_new.append(f[x_pos])
        if show:
            print('peaks : ',len(peaks) , ' detect hr : ', f[x_pos], hr,' haha i got it!!!')
            print('hr.sort : ', hr_new)
            print('oh value : ',hr_spec_value)
            print('oh value.sort : ',hr_spec_value_new,'\n\n')
        return hr_new, peaks #hr_new is sorted!!!

    def hr_peak_count(self, data, fps, size, limit):
        hr_peak = 0
        data = (data-np.min(data)) / (np.max(data)- np.min(data))
        peaks, _ = signal.find_peaks(data, height = limit)
        para = (fps*60)/size 
        hr_peak = len(peaks)*para #,a,b 
        return hr_peak

    def hr_spec_all_fft_cal(self, data, fps, info_show):
        hr_fft = 0
        f, Pxx_den = signal.welch(data.copy(), fps, nfft=fps*100)
        f_bpm = f*60
        # ideal bandpass filter
        f_pass = (f_bpm > 55) & (f_bpm < 180)
        f_pass_not = f_pass==0
        Pxx_den[f_pass_not] = 0
        
        #print(Pxx_den,'kkk')

        ## 前5大數值返回!!
        # max_index = 10
        # max_arr = []
        # ll = Pxx_den.copy().tolist()
        # temp=[]
        # Inf = 0
        # for i in range(max_index):
        #     max_index_temp = ll.index(max(ll))
        #     temp.append(round(f_bpm[max_index_temp],1))
        #     ll[ll.index(max(ll))]=Inf
        # print('max_arr : ', temp ,' by temp method')
        
        ## map search max 5 index!!
        # max_num_index_list = map(data.copy().tolist().index, heapq.nlargest(max_index, data.copy()))
        # print(list(max_num_index_list),'list~~~~~~~~~~~~~~~~')
        # print(max_num_index_list) <map object at 0x0000020565E60898>

        PSD_max_pos = np.argmax(Pxx_den)
        total_hr = []
        total_hr, peak_index = self.detect_spec_number(f_bpm, Pxx_den, PSD_max_pos*0.3, show=info_show)
        
        #print('max psd pos : ',PSD_max_pos,'\n\n')
        hr_fft = f_bpm[PSD_max_pos]

        return hr_fft, np.max(Pxx_den), total_hr, peak_index

    def hr_spec_all_fft_cal2(self, data, fps):
        hr_fft = 0
        f, Pxx_den = signal.welch(data.copy(), fps, nfft=fps*100)
        f_bpm = f*60
        # ideal bandpass filter
        f_pass = (f_bpm > 55) & (f_bpm < 180)
        f_pass_not = f_pass==0
        Pxx_den[f_pass_not] = 0

        PSD_max_pos = np.argmax(Pxx_den)
        total_hr = []
        total_hr, peak_index = self.detect_spec_number(f_bpm, Pxx_den, PSD_max_pos*0.3, show=False)
        hr_fft = f_bpm[PSD_max_pos]

        return hr_fft, np.max(Pxx_den), total_hr, peak_index

    def hr_fft_cal(self, data, fps):
        hr_fft = 0
        f, Pxx_den = signal.welch(data.copy(), fps, nfft=fps*100)
        f_bpm = f*60
        # ideal bandpass filter
        f_pass = (f_bpm > 55) & (f_bpm < 180)
        f_pass_not = f_pass==0
        Pxx_den[f_pass_not] = 0
        PSD_max_pos = np.argmax(Pxx_den)
        hr_fft = f_bpm[PSD_max_pos]
        return hr_fft, np.max(Pxx_den)

    def sb_hr_FPP(self, data, fps=20, sb_peak_limit  = 0.65, sb_pass_limit = 10, csv_save_state=True, writer= None, writer2= None, time_stamp= None, orz= None, face_pos = None, info_show=True,sbhr0 = 0):
    # def sb_hr_FPP(self, data, fps=20, sb_peak_limit  = 0.65, sb_pass_limit = 10, csv_save_state=True, writer= None, time_stamp= None, orz= None, face_pos = None, info_show=True):
        # FPP : FFT、Peak、Pass
        # sb_hr_fft, sb_psd_max = self.hr_fft_cal(data, fps)
        move_distance = np.sqrt((self.face_xy[0]-face_pos[0])**2+(self.face_xy[1]-face_pos[1])**2) 
        self.face_xy = face_pos 
        sb_hr_fft, sb_psd_max , total_hr, peak_index = self.hr_spec_all_fft_cal(data, fps, info_show)
        


        sb_hr_peak = self.hr_peak_count(data.copy(), fps, self.size, sb_peak_limit)	# another heart rate result based on peak detection
        sb_pass = (abs(sb_hr_fft - sb_hr_peak) < sb_pass_limit)	# small error between HRpeak - HRpsd

        if csv_save_state:
            csv_list=[]
            csv_list.append(time_stamp)
            if move_distance > 80:
                move_distance = 0 
            csv_list.append(move_distance)
            
            csv_list.append(orz)
            # csv_list.append(sb_hr_peak)
            # csv_list.append(sbhr0)
            if (sb_pass):
                for i in range(len(total_hr)):
                    csv_list.append(str(total_hr[i]))
            else:
                csv_list.append(0)
            writer2.writerow(csv_list)
        if csv_save_state:
            csv_list=[]
            csv_list.append(time_stamp)
            if move_distance > 80:
                move_distance = 0 
            csv_list.append(move_distance)
            
            csv_list.append(orz)
            # csv_list.append(sb_hr_peak)
            # csv_list.append(sbhr0)
           
            for i in range(len(total_hr)):
                csv_list.append(str(total_hr[i]))

            writer.writerow(csv_list)

        if not(sb_pass):
            sb_hr_fft = 0

        return sb_hr_fft, sb_hr_peak, sb_pass, total_hr, move_distance

    def pca_hr_FPP(self, data, fps=20, pca_peak_limit  = 0.65, pca_pass_limit = 10):
        if len(data) > 1:
            pca_hr_fft, pca_psd_max = self.hr_fft_cal(data, fps)
            # pca_hr_fft, pca_psd_max , pca_total_hr, pca_peak_index = self.hr_spec_all_fft_cal2(data, fps)
            pca_hr_peak = self.hr_peak_count(data.copy(), fps, self.pca_size, pca_peak_limit)	# another heart rate result based on peak detection
            pca_pass = (abs(pca_hr_fft - pca_hr_peak) < pca_pass_limit)
            return pca_hr_fft, pca_hr_peak, pca_psd_max, pca_pass
        else:
            return 0, 0, 0, False
    
    def sb_hr_filter(self, hr_list, hr, hr_state, total_hr):
        if (len(hr_list) < self.size): 
            # if hr_state and (hr<120):
            if (hr<130) and (hr>55):
                hr_list.append(hr)
                return True
            else:
                return False
        else:
            '''            0615舊本版
            if (abs(hr - np.mean(hr_list)) < 10) and hr_state:
                #print('QQ this is OK')
                del(hr_list[0])
                hr_list.append(hr)
                return True'''
            ## 0616 新版本 !
            # if hr_state:
            # if True:
            '''                
            for i in range(len(total_hr)):
                if abs(total_hr[i] - np.mean(hr_list)) < 10:
                    del(hr_list[0])
                    hr_list.append(total_hr[i])
                    min(total_hr, key=lambda x:abs(x-np.mean(hr_list)))
            return True'''
            min_value = min(total_hr, key=lambda x:abs(x-np.mean(hr_list)))
            if (len(total_hr)>1) and (abs(total_hr[1] - np.mean(hr_list)) < 5) :
                del(hr_list[0])
                hr_list.append(total_hr[1])
                return True

            elif abs(total_hr[0] - np.mean(hr_list)) < 10 :
                del(hr_list[0])
                hr_list.append(total_hr[0])
                return True

            elif abs(min_value - np.mean(hr_list)) <15 :
                del(hr_list[0])
                hr_list.append(min_value)
                return False

            elif abs(total_hr[0] - np.mean(hr_list)) <20 :
                del(hr_list[0])
                hr_list.append(total_hr[0])
                return False

            else:
                return False


    def pca_hr_filter(self, hr_list, hr, threshold, hr_state):
        if (len(hr_list) < self.size/2) :
            if hr_state: 
                hr_list.append(hr)
        elif (len(hr_list) < self.size) :
            if (abs(hr - np.mean(hr_list)) < 30) and hr_state and threshold:
                hr_list.append(hr)
        else:
            if (abs(hr - np.mean(hr_list)) < 30) and hr_state and threshold:
                del(hr_list[0])
                hr_list.append(hr)

    def pca_psd_list(self, psd_list, psd):
        if (len(psd_list) < self.pca_size) :
            psd_list.append(psd)
        else:
            del(psd_list[0])
            psd_list.append(psd)
        
        if len(psd_list) > 0 :
            return ( psd > np.mean(psd_list)*0.4 )
        else:
            return True


    def pca_sample(self, data, sb_stable_state = True):
        if sb_stable_state:
            if len(self.pca_array) < self.pca_size:
                self.pca_array = np.vstack([self.pca_array, data[-self.pca_size:]])
            else:
                self.pca_state = True
                self.pca_array = np.vstack([self.pca_array[1:,:],data[-self.pca_size:]])    
            
            P = self.pca_main(self.pca_array) # P size : 1*128
            return P
        else:
            return [0]
        
    def pca_main(self, data):
        H_pca, _ = self.pca_n(data.transpose(), 2)
        # print("H_pca", H_pca.shape)
        H_pca = np.asarray(H_pca)
        H_pca_trans = H_pca.copy()
        H_pca_trans = H_pca_trans.transpose()	#1*128
        H_pca_trans = np.real(H_pca_trans)
        H_pca_trans_one = H_pca_trans[0,:]
        H_pca_array = np.squeeze(np.asarray(H_pca_trans_one))
        # H_pca_array size reduced to (1, 128)
        return H_pca_array

    def pca_n(self, dataMat, n):
        def zeroMean(dataMat):		
            meanVal=np.mean(dataMat,axis=0)		#按列求均值，即求各个特征的均值
            newData=dataMat-meanVal
            return newData,meanVal
        newData,meanVal=zeroMean(dataMat)
        covMat=np.cov(newData,rowvar=0)	   #求协方差矩阵,return ndarray；若rowvar非0，一列代表一个样本，为0，一行代表一个样本
        eigVals,eigVects=np.linalg.eig(np.mat(covMat))#求特征值和特征向量,特征向量是按列放的，即一列代表一个特征向量
        eigValIndice=np.argsort(eigVals)			#对特征值从小到大排序
        n_eigValIndice=eigValIndice[-1:-(n+1):-1]	#最大的n个特征值的下标
        n_eigVect=eigVects[:,n_eigValIndice]		#最大的n个特征值对应的特征向量
        lowDDataMat=newData*n_eigVect				#低维特征空间的数据
        reconMat=(lowDDataMat*n_eigVect.T)+meanVal	#重构数据
        this_eigen = eigVals[n_eigValIndice]
        return lowDDataMat,reconMat

