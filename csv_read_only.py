import csv
import numpy as np

import matplotlib.pyplot as plt
polar_path = './excel_hr_record/'

TempName = 'output_1020_0859'
# TempName = 'output_1020_0859_hrpass'
File_Name = TempName + '.csv'

path = polar_path
time_stamp = []
polar_hr = []
sstc_hr = []
max_fft = []
sec_fft = []
third_fft = []
fourth_fft = []

t_polar = []
t_ecg = []

t1 = []
t2 = []
t3 = []
t4 = []

## 0 datetime
## 1 polar
## 2 sstc hr
## 3 max fft
## 4 sec fft
## 5 third fft
## 6 fourth fft
##
##       END
## -----------------




## ------------------
##    CSV funciton
##

def csv_convert(file_name):
    # polar_temp = polar_convert(polar_path+Polar_File_Name)
    # 開啟 CSV 檔案
    temp = []
    start_time = []
    with open(file_name, newline='') as csvfile:

        # 讀取 CSV 檔案內容
        rows = csv.reader(csvfile)

        # 以迴圈輸出每一列
        i = 0
        polar_j = 0
        time_sec = '0'
        for row in rows:
            if i > 0:
                # print('row',row)
                # time_stamp.append(str(row[1]))
                sstc_hr.append(float(row[2]))
                max_fft.append(float(row[3]))
                t1.append(i)
                
                if len(row) > 6 : #and len(row[6])>0 : # len = 7
                    fourth_fft.append(float(row[6]))
                    t4.append(i)
                    third_fft.append(float(row[5]))
                    t3.append(i)
                    sec_fft.append(float(row[4]))
                    t2.append(i)

                elif len(row) > 5 : # and len((row[5]))>0 : # len = 6
                    # print('heellloo')
                    third_fft.append(float(row[5]))
                    t3.append(i)
                    sec_fft.append(float(row[4]))
                    t2.append(i)
                elif len(row) > 4: # and len(row[4])>0 : # len = 5
                    
                    sec_fft.append(float(row[4]))
                    t2.append(i)
            i+=1
        x = np.arange(0, i-1, 1)
    return x

x = csv_convert(path+File_Name)
# print(sstc_hr)
print('2 len ',len(sec_fft))
print('3 len ',len(third_fft))
print('4 len ',len(fourth_fft))

plt.plot(x,sstc_hr,"r--",label="S200 HR")
# plt.plot(t_polar,polar_hr,"m--",label="Polar HR")
plt.plot(t1,max_fft,"g.",ms=1,label="Max power fft")
plt.plot(t2,sec_fft,"b.",ms=1,label="Sec power fft")
plt.plot(t3,third_fft,"r.",ms=1,label="Third power fft")
plt.plot(t4,fourth_fft,"y.",ms=1,label="Fourth power fft")

# plt.plot(t_polar,polar_hr,"r--")
# plt.plot(t1,max_fft,"c.",ms=1,label="Max power fft")
# plt.plot(t2,sec_fft,"c.",ms=1,label="Sec power fft")
# plt.plot(t3,third_fft,"c.",ms=1,label="Third power fft")
# plt.plot(t4,fourth_fft,"c.",ms=1,label="Fourth power fft")

plt.title("S200 HR with GroundTrue Polar 0H1+")
plt.ylabel('Heart Rate (bpm)',fontsize=10)
plt.xlabel('Data Number',fontsize=10)

# plt.grid()
# plt.yticks([0,0.05 ])
plt.legend()
plt.show()