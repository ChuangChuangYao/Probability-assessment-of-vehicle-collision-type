#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('git clone https://github.com/theAIGuysCode/yolov4-deepsort')


# In[5]:


#!python save_model.py --model yolov4


# In[14]:


# save yolov4-tiny model
#!python save_model.py --weights ./data/yolov4-tiny.weights --output ./checkpoints/yolov4-tiny-416 --model yolov4 --tiny


# In[ ]:


# Run yolov4-tiny object tracker
#python object_tracker.py --weights ./checkpoints/yolov4-tiny-416 --model yolov4 --video ./data/video/test.mp4 --output ./outputs/tiny.avi --tiny


# In[9]:


get_ipython().run_line_magic('cd', 'D:\\NCKU\\YOLOv4-DeepSORT\\yolov4-deepsort')


# In[2]:


#import tensorflow as tf
#physical_devices = tf.config.experimental.list_physical_devices('GPU')
#len(physical_devices)


# In[4]:


#練習
import pandas as pd
import numpy as np
test=pd.read_excel('D:/NCKU/YOLOv4-DeepSORT/yolov4-deepsort/final_traffic_accident_0512.xlsx','75_frames_1020') 
test_array = np.array(test)
video_name_data = np.array(test[test['ID'] == 109100401])
collision_type = video_name_data[0,1]
print(collision_type)
print(video_name_data)


# In[5]:


test_array


# In[27]:


#練習
import pandas as pd
import numpy as np
import os
video_path = 'D:/NCKU/YOLOv4-DeepSORT/yolov4-deepsort/eve_video/到1020_75_frames/109100607.mp4'
print(video_path)
file_name= os.path.basename(video_path) #從路徑中分離出檔名
print(file_name)
video_name=os.path.splitext(file_name)[0]
print('此影片檔名：',video_name)


# In[4]:


#練習
import pandas as pd
import numpy as np
test=pd.read_excel('D:/NCKU/YOLOv4-DeepSORT/yolov4-deepsort/final_traffic_accident_0512.xlsx','75_frames_1020') 
#print(test[test['ID'] == 109100401])
data_argmax = test.index[test['ID'] == 109100507][0]
print(test.index[test['ID'] == 109100507])
print(test['ID'])
data_data = test[test['ID'] == 109100507]
print('data_data=',data_data)
print('index=',data_data.index[0])
print(data_argmax)
test_array = np.array(test)
collision_type = test_array[data_argmax,1]
collision_type


# In[5]:


#練習
ID_data = test['ID']
ID_data_a = test[['ID']]
print(type(ID_data))
print(type(ID_data_a))
print(ID_data)
print(ID_data_a)
print(ID_data.index[ID_data[:] == 109100507])
print('shape: ',np.shape(ID_data.index[ID_data[:] == 109100507]))
print('type: ',type(ID_data.index[ID_data[:] == 109100507]))
data_argmax = ID_data.index[ID_data[:] == 109100507][0]
print('index: ',data_argmax)
#print(test.index[test['ID'] == video_name])
#data_argmax = test.index[test['ID'] == video_name][0]


# In[6]:


#練習
a = 0
index = 0
for i in ID_data:
    a+=1
    if i == 109100504:
        index = a-1
print(index)


# In[10]:


#跑全部影片
import pandas as pd
import numpy as np
test=pd.read_excel('D:/NCKU/YOLOv4-DeepSORT/yolov4-deepsort/final_traffic_accident_0604.xlsx','75_frames_1130') #到10/20的75frames影片
c = np.array(test)
for i in c:
    if i[1]==0:
        j='headon'
    elif i[1]==1:
        j='lateral'
    elif i[1]==2:
        j='sideswipe'
    elif i[1]==3:
        j='rearend'
    else:
        print(i[2],'什麼都不是')
    video_path = 'D:/NCKU/YOLOv4-DeepSORT/yolov4-deepsort/eve_video/到1130_75_frames/'+str(i[1])+'/'+str(i[2])+'.mp4'
    output_path = 'D:/NCKU/YOLOv4-DeepSORT/yolov4-deepsort/eve_video/output/75frames/1021-1130/'+str(i[1])+'/'+str(i[2])+'.avi'
    get_ipython().system('python object_tracker.py --video {video_path} --output {output_path} --model yolov4 --dont_show')


# In[13]:


#跑單一影片
video_path = 'D:/NCKU/YOLOv4-DeepSORT/yolov4-deepsort/eve_video/到1130_75_frames/1/109111205.mp4' 
output_path = 'D:/NCKU/YOLOv4-DeepSORT/yolov4-deepsort/eve_video/output/75frames/1021-1130/1/109111205.avi'
get_ipython().system('python object_tracker.py --video {video_path} --output {output_path} --model yolov4 --dont_show')


# In[4]:


#計算執行時間
import time
start = time.process_time()
#要測量的程式碼
get_ipython().system('python object_tracker.py --video ./eve_video/headon/109100512.mp4 --output ./eve_video/output/觀察/109100512.avi --model yolov4 --dont_show')
end = time.process_time()
print("執行時間：%f 秒" % (end - start))


# In[6]:


# run DeepSort with YOLOv4 Object Detections as backbone (enable --info flag to see info about tracked objects)
#!python object_tracker.py --video ./eve_video/headon/109100512.mp4 --output ./eve_video/output/0/109100512.avi --model yolov4 --dont_show


# In[8]:


#import pandas as pd
#test=pd.read_excel('D:/NCKU/YOLOv4-DeepSORT/存取影片檔名練習/test.xlsx','run')
#test


# In[9]:


#import numpy as np
#a= np.array(test["ID"])
#b= np.array(test["collision_type"])
#c = np.array(test)
#print(a)
#print(b)
#print(c)


# In[10]:


#j='headon'
#print(str(j))
#print(str(c[0,0]))

