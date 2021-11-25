#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Read in Data
# Author: Yao Hsuan, Chuang
# Update Time: 2021/09/03
import numpy as np
import os
import glob
import cv2
import h5py
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Input, Dense, Flatten
from keras.applications.resnet50 import ResNet50
from os.path import basename
from sklearn.preprocessing import MinMaxScaler


# In[3]:


# Read the dynamic features data
df = pd.read_excel('F:/NCKU/YOLOv4-DeepSORT/選取ID資料/2347/0614資料轉換合併_轉換角度_8.xlsx')
num_data = 278 # the number of data

num_classes = 4 # the number of class
num_feature = 5 # the number of feature
num_frame = 75 # the number of frame


# In[4]:


d_y = df["collision_type"] # Store the column "accident" into d_y
d_id = df["ID"] # Store the column "ID" into d_id 
df = df.drop(["ID", "collision_type"], axis = 1) # Drop the columns "ID" and "accident"


# In[5]:


df_data = df
#df_data = df_normalize

# 宣告新的動態特徵資料表sort_df以及sort_df_test來儲存整理好的動態特徵訓練資料及測試資料
sort_df = df.copy()
sort_df = sort_df.iloc[0:0]

sort_df_test = df.copy()
sort_df_test = sort_df_test.iloc[0:0]


train_img_sequence = [] # 宣告train_img_sequence為一list儲存訓練資料集中的每一張影像
test_img_sequence = [] # 宣告test_img_sequence為一list儲存驗證資料集中的每一張影像

# 宣告載入影片檔的函數
def extract_images(videofile, train):    # videofile為影片的檔案路徑、train代表欲讀取資料進訓練資料集還是測試資料集
    
    vidcap = cv2.VideoCapture(videofile) # 使用opencv的VideoCapture函數載入影片，宣告vidcap物件
    success = True  # 將"sucess" flag 預設為 True
    img = []        # 宣告一list來暫時讀入影像

    while success:  # 當success flag為True
        for j in range(0, num_frame): # 如果success為true，則持續讀入下一幀
            success, img = vidcap.read()
            tmp_img = cv2.resize(img, (224, 224), interpolation = cv2.INTER_CUBIC) # 將影像大小轉為224 x 224
            if train == True: # 若為訓練資料集
                train_img_sequence.append(tmp_img) # 將轉換後的影像存入train_img_sequence
            else: # 若為測試資料集
                test_img_sequence.append(tmp_img) # 將轉換後的影像存入test_img_sequence
        
        if train == True: # 若為訓練資料集
            print ("此影片的shape為{shape}"            .format(shape = np.shape(train_img_sequence)))
        else: # 若為測試資料集
            print ("此影片的shape為{shape}"            .format(shape = np.shape(test_img_sequence)))
        success = False
    vidcap.release()  # 結束vidcap
    
    if train == True: # 若為訓練資料集
        return train_img_sequence # 輸出為影片的幀陣列
    else: # 若為測試資料集
        return test_img_sequence # 輸出為影片的幀陣列


# In[6]:


# 從每個影片中載入數據並保存到數據陣列，其shape為（L，90, 224, 224, 3）
# 其中L是影片數量。 使用以下函數來載入影片
def make_dataset(rand, train): # rand為輸入的資料集
    
    global sort_df, sort_df_test                        # 在函式內對已存在的全域變數做修改要宣告為global
    
    for i, file in enumerate(rand):                     # 對於資料集rand中的每一筆資料
        print ("第{i}個檔的路徑為{file}"        .format(i = i, file = file))                    # 第i個檔，檔案路徑為file
        
        filename = os.path.basename(file)               # 從路徑中分離出檔名
        print(os.path.splitext(filename)[0])            # 從檔名中去除附檔名
        
        # 處理動態特徵資料
        for j in range(num_data):
            if os.path.splitext(filename)[0] == str(d_id[j]):   # 判斷影片檔在動態特徵資料中的位置
                if train == True: # 若為訓練資料集 #df_normalize
                    sort_df = sort_df.append(df_data.iloc[[j]], ignore_index = True)   # 整理成新的動態特徵資料表
                else: # 若為測試資料集 #df_normalize
                    sort_df_test = sort_df_test.append(df_data.iloc[[j]], ignore_index = True)   # 整理成新的動態特徵資料表

        # 處理影片資料
        if file[-4:] == '.mp4':
            if train == True:
                extract_images(file, True)                        # 利用前面定義的extract_images函式來載入影片資料
            else:
                extract_images(file, False)

    if train == True: # 若為訓練資料集
        print ("此資料集格式為{shape}"        .format(shape = np.shape(train_img_sequence)))
    else: # 若為測試資料集
        print ("此資料集格式為{shape}"        .format(shape = np.shape(test_img_sequence)))

    if train == True: # 若為訓練資料集
        return train_img_sequence
    else: # 若為測試資料集
        return test_img_sequence


# In[7]:


if num_classes == 4:
    eve_video_file = '75_frames'
elif num_classes == 3:
    eve_video_file = '75_frames_3class'
elif num_classes == 2:
    eve_video_file = '75_frames_2class'

headon_files = glob.glob('F:/NCKU/YOLOv4-DeepSORT/yolov4-deepsort/eve_video/'+eve_video_file+'/0/*.mp4') # (Head-on,1)檔案路徑
lateral_files = glob.glob('F:/NCKU/YOLOv4-DeepSORT/yolov4-deepsort/eve_video/'+eve_video_file+'/1/*.mp4') # (Lateral,2)檔案路徑
sideswipe_files = glob.glob('F:/NCKU/YOLOv4-DeepSORT/yolov4-deepsort/eve_video/'+eve_video_file+'/2/*.mp4') # (Side-swipe,3)檔案路徑
rearend_files = glob.glob('F:/NCKU/YOLOv4-DeepSORT/yolov4-deepsort/eve_video/'+eve_video_file+'/3/*.mp4') # (Rear-end,4)檔案路徑
all_files = np.concatenate((headon_files, lateral_files, sideswipe_files, rearend_files))

print ("{headon}筆Head-on碰撞影片, {lateral}筆Lateral碰撞影片, {sideswipe}筆Side-swipe碰撞影片, {rearend}筆Rear-end碰撞影片".format(headon = len(headon_files),lateral = len(lateral_files),sideswipe = len(sideswipe_files),rearend = len(rearend_files)))      # 印出有幾筆無碰撞資料、有碰撞資料

# 將影片標籤轉換為one-hot encoding
def label_matrix(values):
    return np.eye(num_classes)[values]  # return matrix，1在第一欄為Head-on碰撞、1在第二欄為Lateral碰撞

labels = np.concatenate(([0]*len(headon_files), [1]*len(lateral_files), [2]*len(sideswipe_files), [3]*len(rearend_files)))  # 建立影片標籤
labels = label_matrix(labels)   #labels =np.eye(4)[labels]
print(np.shape(labels))


# In[ ]:


'''三個類別數範例
if num_classes == 4:
    eve_video_file = '75_frames'
elif num_classes == 3:
    eve_video_file = '75_frames_3class'
elif num_classes == 2:
    eve_video_file = '75_frames_2class'
    
headon_rearend_files = glob.glob('F:/NCKU/YOLOv4-DeepSORT/yolov4-deepsort/eve_video/'+eve_video_file+'/0/*.mp4') # (Head-on+Rear-end,1)檔案路徑
lateral_files = glob.glob('F:/NCKU/YOLOv4-DeepSORT/yolov4-deepsort/eve_video/'+eve_video_file+'/1/*.mp4') # (Lateral,2)檔案路徑
sideswipe_files = glob.glob('F:/NCKU/YOLOv4-DeepSORT/yolov4-deepsort/eve_video/'+eve_video_file+'/2/*.mp4') # (Side-swipe,3)檔案路徑

all_files = np.concatenate((headon_rearend_files, lateral_files,sideswipe_files))
print ("{headon_rearend}筆Head-on+Rear-end碰撞影片, {lateral}筆Lateral碰撞影片, {sideswipe}筆Side-swipe碰撞影片"\
.format(headon_rearend = len(headon_rearend_files),lateral = len(lateral_files),sideswipe = len(sideswipe_files)))

# 將影片標籤轉換為one-hot encoding變數
def label_matrix(values):
    return np.eye(num_classes)[values]  # return matrix，1在第一欄為Head-on碰撞、1在第二欄為Lateral碰撞

labels = np.concatenate(([0]*len(headon_rearend_files), [1]*len(lateral_files), [2]*len(sideswipe_files))) 
labels = label_matrix(labels) 
print(np.shape(labels))
'''


# In[8]:


# 將資料分割為訓練資料集與驗證資料集(80%訓練、20%驗證)
x_train, x_t1, y_train, y_t1 = train_test_split(all_files, labels, test_size = 0.20, random_state = 2, stratify = labels)  # 使用sklearn套件train_test_split進行分割
# 轉換為array
x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_t1)
y_test = np.array(y_t1)
print("載入影片中，請稍後...")
print("-----訓練資料集-----")
x_train = make_dataset(x_train, True)
print("-----驗證資料集-----")
x_test = make_dataset(x_test , False)


# In[9]:


# 印出基本資訊
'''
print(np.shape(x_train))
print(np.shape(x_test))
print(type(x_train))
print(np.shape(y_train))
print(np.shape(y_test))
print(y_train)
print(x_train)
'''
x_train = np.array(x_train)
x_test = np.array(x_test)
print('x_train shape:', x_train.shape)
num_train = int(x_train.shape[0] / num_frame)
num_test = int(x_test.shape[0] / num_frame)
print(num_train, '筆train samples')
print(num_test, '筆test samples')

print('動態訓練資料shape: ',sort_df.shape)
print('動態測試資料shape: ',sort_df_test.shape)
# 將動態特徵分成6個時間段，每個時間段有30個特徵
sort_df_reshape = sort_df.values.reshape((num_train, 75, num_feature))  #預設(num_train, 6, 12)
sort_df_test_reshape = sort_df_test.values.reshape((num_test, 75, num_feature))  #預設(num_test, 6, 12)
print('動態訓練資料reshape: ',sort_df_reshape.shape)
print('動態測試資料reshape: ',sort_df_test_reshape.shape)


# In[10]:


# 建置ResNet50
# 模型建置input_shape = (img_height, img_width, RGB)
model = Sequential()
model.add(ResNet50(include_top = False, weights = 'imagenet', input_tensor = None, input_shape = (224 , 224 , 3)))
model.add(Flatten())

# 編譯模型
model.compile(loss = 'binary_crossentropy', # 設定損失函數 #categorical_crossentropy
              optimizer = 'NAdam',          # 設定優化器
              metrics = ['accuracy'])       # 設定成效衡量指標


# In[11]:


# 訓練資料集透過ResNet50轉換為image features
img_feature = model.predict(x_train)
print(np.shape(img_feature))
# 資料分割回每90幀為一筆資料
img_feature_reshape = np.reshape(img_feature, (num_train, num_frame, 100352))
print(img_feature_reshape.shape)

# 驗證資料集透過ResNet50轉換為image features
img_feature_test = model.predict(x_test)
print(np.shape(img_feature_test))
# 資料分割回每90幀為一筆資料
img_feature_test_reshape = np.reshape(img_feature_test, (num_test, num_frame, 100352))
print(img_feature_test_reshape.shape)


# In[12]:


# 將資料儲存為hdf5檔，以供後續GRU/LSTM使用
# 設定儲存路徑 file_setting
file_setting = 'ReadData_2347/4class_轉換角度_8_80%20%_non_normalize_v1'

h5f = h5py.File('F:/NCKU/YOLOv4-DeepSORT/選取ID資料/Model/'+file_setting+'/X_img_train.h5', 'w')
h5f.create_dataset('dataset_1', data = img_feature_reshape)
h5f.close()

h5f = h5py.File('F:/NCKU/YOLOv4-DeepSORT/選取ID資料/Model/'+file_setting+'/X_img_test.h5', 'w')
h5f.create_dataset('dataset_1', data = img_feature_test_reshape)
h5f.close()

h5f = h5py.File('F:/NCKU/YOLOv4-DeepSORT/選取ID資料/Model/'+file_setting+'/X_img_train_noResNet.h5', 'w')
h5f.create_dataset('dataset_1', data = x_train)
h5f.close()

h5f = h5py.File('F:/NCKU/YOLOv4-DeepSORT/選取ID資料/Model/'+file_setting+'/X_img_test_noResNet.h5', 'w')
h5f.create_dataset('dataset_1', data = x_test)
h5f.close()

h5f = h5py.File('F:/NCKU/YOLOv4-DeepSORT/選取ID資料/Model/'+file_setting+'/Y_train.h5', 'w')
h5f.create_dataset('dataset_1', data = y_train)
h5f.close()

h5f = h5py.File('F:/NCKU/YOLOv4-DeepSORT/選取ID資料/Model/'+file_setting+'/Y_test.h5', 'w')
h5f.create_dataset('dataset_1', data = y_test)
h5f.close()


# In[13]:


# 將動態特徵模型儲存為hdf5檔
h5f = h5py.File('F:/NCKU/YOLOv4-DeepSORT/選取ID資料/Model/'+file_setting+'/X_d_train.h5', 'w')
h5f.create_dataset('dataset_1', data = sort_df_reshape)
h5f.close()

h5f = h5py.File('F:/NCKU/YOLOv4-DeepSORT/選取ID資料/Model/'+file_setting+'/X_d_test.h5', 'w')
h5f.create_dataset('dataset_1', data = sort_df_test_reshape)
h5f.close()

