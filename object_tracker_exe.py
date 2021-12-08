get_ipython().system('git clone https://github.com/theAIGuysCode/yolov4-deepsort')

#!python save_model.py --model yolov4
#!python save_model.py --weights ./data/yolov4-tiny.weights --output ./checkpoints/yolov4-tiny-416 --model yolov4 --tiny

#you must cd to the place where 'yolov4-deepsort' is
get_ipython().run_line_magic('cd', 'D:\\NCKU\\YOLOv4-DeepSORT\\yolov4-deepsort')


#import tensorflow as tf
#physical_devices = tf.config.experimental.list_physical_devices('GPU')
#len(physical_devices)

#run all the videos in a file
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


#run only one video
video_path = 'D:/NCKU/YOLOv4-DeepSORT/yolov4-deepsort/eve_video/到1130_75_frames/1/109111205.mp4' 
output_path = 'D:/NCKU/YOLOv4-DeepSORT/yolov4-deepsort/eve_video/output/75frames/1021-1130/1/109111205.avi'
get_ipython().system('python object_tracker.py --video {video_path} --output {output_path} --model yolov4 --dont_show')


#計算執行時間
import time
start = time.process_time()
#要測量的程式碼
get_ipython().system('python object_tracker.py --video ./eve_video/headon/109100512.mp4 --output ./eve_video/output/觀察/109100512.avi --model yolov4 --dont_show')
end = time.process_time()
print("執行時間：%f 秒" % (end - start))

