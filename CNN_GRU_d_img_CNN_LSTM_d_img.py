#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Model CNN-GRU_d_img/Model CNN-LSTM_d_img
# Author: Yao Hsuan, Chuang
# Update Time: 2021/09/03
import numpy as np
import h5py
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import datetime
import itertools
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from keras.models import Model, Sequential
from keras.layers import Input, Dense, LSTM, concatenate, Dropout, GRU
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

from sklearn.preprocessing import label_binarize
from scipy import interp
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from keras.models import load_model


# In[53]:


#讀取已存的model
trainingModel = load_model('F:/NCKU/YOLOv4-DeepSORT/選取ID資料/Model/CNN_LSTM_d_img/model_gru_save/model_gru_4class_B_80%20%_non-normalize.h5')


# In[2]:


# 讀取h5f檔  #改資料路徑file_setting
file_setting = 'ReadData_2347/4class_80%20%_non-normalize'
h5f = h5py.File('F:/NCKU/YOLOv4-DeepSORT/選取ID資料/Model/'+file_setting+'/X_img_train.h5','r')
x_train_img = h5f['dataset_1'][:]
h5f.close()

h5f = h5py.File('F:/NCKU/YOLOv4-DeepSORT/選取ID資料/Model/'+file_setting+'/X_img_test.h5','r')
x_test_img = h5f['dataset_1'][:]
h5f.close()

h5f = h5py.File('F:/NCKU/YOLOv4-DeepSORT/選取ID資料/Model/'+file_setting+'/X_d_train.h5','r')
x_train_d = h5f['dataset_1'][:]
h5f.close()

h5f = h5py.File('F:/NCKU/YOLOv4-DeepSORT/選取ID資料/Model/'+file_setting+'/X_d_test.h5','r')
x_test_d = h5f['dataset_1'][:]
h5f.close()

h5f = h5py.File('F:/NCKU/YOLOv4-DeepSORT/選取ID資料/Model/'+file_setting+'/Y_train.h5','r')
y_train = h5f['dataset_1'][:]
h5f.close()

h5f = h5py.File('F:/NCKU/YOLOv4-DeepSORT/選取ID資料/Model/'+file_setting+'/Y_test.h5','r')
y_test = h5f['dataset_1'][:]
h5f.close()


# In[3]:


# 設定測試資料集參數
num_class = 4  #類別數
num_feature = 5  #參數量
num_frame = 75  #每部影片frame數

num_test = int(x_test_img.shape[0])
# 為了得出每個時間點的碰撞風險
# 這邊修改測試資料集中的動態特徵的維度，每次只讀取一個單位時間的資料
print(x_test_d.shape)
x_test_d_reshape = np.reshape(x_test_d, (num_test, 75, 1, 1, num_feature))
print(x_test_d_reshape.shape)

# 這邊修改測試資料集中的影片特徵的維度，每次只讀取一個單位時間的資料
print(x_test_img.shape)
x_test_img_reshape = np.reshape(x_test_img, (num_test, 75, 1, 1, 100352)) #75個時步 每時步1frame
print(x_test_img_reshape.shape) 


# **宣告模型建置的函數  (layer可用LSTM/GRU)**

# In[21]:


def createModel_new(forTraining):
    global merged
    # model for training, stateful = False, any batch size
    if forTraining == True:
        batchSize = None
        stateful = False
        num_time_d = 75 #階段數
        num_time = 75 #frame數

    # model for predicting, stateful = True, fixed batch size
    else:
        batchSize = 1
        stateful = True
        num_time_d = 1  #動態特徵每1frame一筆資料
        num_time = 1  #影片每1frame一筆資料

    # 輸入一: 動態特徵資料
    model_d = Sequential()
    model_d.add(GRU(256, return_sequences = True, input_shape = (num_time_d, num_feature))) 
    '''
    調整參數 (參考)
    #model_d.add(GRU(256, return_sequences = True, input_shape = (num_time_d, num_feature)))
    #model_d.add(Dropout(0.2))
    #model_d.add(LSTM(256, return_sequences = True, input_shape = (num_time_d, num_feature)))
    #model_d.add(Dropout(0.1))
    model_d.add(GRU(256, return_sequences = False, input_shape = ( num_time_d, num_feature)))
    #model_d.add(LSTM(128, return_sequences = False, input_shape = ( num_time_d, num_feature)))
    '''
    # 輸入二: 影片資料
    model = Sequential()
    '''
    調整參數 (參考)
    #model.add(LSTM(64, return_sequences = True, input_shape = ( num_time, 100352))) #(75,100352)
    #model.add(Dropout(0.1))
    #model.add(LSTM(256, return_sequences = False, input_shape = ( num_time, 100352))) #(75,100352)
    model.add(GRU(128, return_sequences = False, input_shape = ( num_time, 100352))) #(75,100352)
    '''
    # 合併兩種資料的LSTM網路
    merged = concatenate([model_d.output,model.output],axis=-1)
    merged_dense = Dense(num_class, activation = 'softmax')(merged) #兩分類亦可用sigmoid
    model_merge = Model(inputs = [model_d.input, model.input], outputs = merged_dense)
    opt = Adam(lr = 0.0001) # 設定優化器與learning rate
    if forTraining == True:
        model_merge.compile(loss = 'categorical_crossentropy', # 設定損失函數  #若是兩分類則改成binary_crossentropy
                  optimizer = opt,          # 設定優化器
                  metrics = ['accuracy'])   # 設定成效衡量指標
    model_merge.summary
    return model_merge


# In[23]:


# create model
trainingModel = createModel_new(forTraining = True)

start = time.process_time()
# train model
#es = EarlyStopping(monitor='val_loss', mode='min', verbose=1) #另一種earlystop
es = EarlyStopping(monitor='val_loss',patience=7, min_delta=0.001)
train_history_lstm = trainingModel.fit([x_train_d, x_train_img], y_train, batch_size = 32 , epochs = 100, validation_split=0.1,callbacks=[es])   

end = time.process_time()
# 訓練時間
print("執行時間：%f 秒" % (end - start))


# **執行評估結果前，先至下方宣告評估指標函數**

# In[24]:


if num_class == 4:
    class_setting = ['Head-on', 'Lateral','Sideswipe','Reae-end']
elif num_class == 3:
    class_setting = ['Head-on & Reae-end', 'Lateral','Sideswipe']
elif num_class == 2:
    class_setting = ['Head-on & Reae-end', 'Lateral & Sideswipe']

#Confusion metrix
start1 = time.process_time()
y_pred = trainingModel.predict([x_test_d, x_test_img])
end1 = time.process_time()
print("執行時間：%f 秒" % (end1 - start1))

y_pred_bool = np.argmax(y_pred, axis = 1)
y_test_bool = np.argmax(y_test, axis = 1)
cm = confusion_matrix(y_test_bool, y_pred_bool)
#print('預測:' + str(y_pred_bool))
#print('真實:',str(y_test_bool))

#classification_report
print(classification_report(y_test_bool, y_pred_bool,target_names= class_setting, digits=4))
#print('micro f1_score: ',f1_score(y_test_bool, y_pred_bool, average='micro'))
#print('macro f1_score: ', f1_score(y_test_bool, y_pred_bool, average='macro'))

plot_confusion_matrix(cm, classes = class_setting)
plt.show()


# In[17]:


#4class plot_graph
plot_graph(trainingModel, y_test_bool)


# In[6]:


#沒閥值ROC
ROC_nothreshold(y_test, y_pred)


# In[ ]:


#有閥值ROC
ROC_threshold(y_test, y_pred)


# In[72]:


#觀察accuracy & loss 在每個epoch的變化
Epoch_acc_loss (train_history_lstm)


# In[14]:


#2class  #plot_graph前要跑的 (為了方便調整線條顏色，因此與下個cell的plot_graph_2class分開跑)
predictModel(createModel_new) 


# In[19]:


#2class  #設定線條顏色
plot_graph_2class(y_test_bool,'lightskyblue','mediumpurple') 

#適合的顏色參考：darkkhaki, gold, sandybrown,thistle, lightcoral, lightskyblue, mediumpurple


# In[22]:


#3class plot_graph
plot_graph_3class(trainingModel, y_test_bool)


# In[65]:


#2class plot_histogram
plot_histogram_2class(y_test_bool, y_pred_bool)


# **宣告評估指標函數**

# In[7]:


#定義混淆矩陣 
def plot_confusion_matrix(cm,classes):
    #cm = confusion_matrix(y_test_bool,y_pred_bool)
    #classes = ['Head-on', 'Lateral','Sideswipe','Reae-end']
    plt.figure(figsize = (7, 7))
    plt.imshow(cm, interpolation='none', cmap=plt.cm.bone_r) #gist_gray_r #PuBu #pink_r
    plt.title('Confusion Matrix', size = 18)
    sns.set(style = "dark")
    plt.colorbar(aspect=4)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, size = 18)
    plt.yticks(tick_marks, classes, size = 18)

    fmt='d'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), fontsize = 20,
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label', size = 18)
    plt.xlabel('Predicted label', size = 18)


# In[ ]:


#ROC_nothreshold
def ROC_nothreshold(y_test, y_pred):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_class):
        print('class: ',i)
        fpr[i], tpr[i], thresholds = roc_curve(y_test[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        #計算最佳threshold
        #print('threshold = ',thresholds)
        gmean = np.sqrt(tpr[i] * (1 - fpr[i]))
        #tpr_fpr = tpr[i]-fpr[i]
        #print(tpr_fpr)
        num_index = np.argmax(gmean)
        opt_threshold = thresholds[num_index]
        print('最佳threshold = ', opt_threshold)
    
    
    # Plot ROC curve
    plt.figure(figsize=(7,5))
    plt.grid()
    colors = cycle(['yellowgreen', 'darkorange', 'cornflowerblue','mediumpurple'])
    for i, color in zip(range(num_class), colors):
        fpr[i], tpr[i], thresholds = roc_curve(y_test[:, i], y_pred[:, i])
        plt.plot(fpr[i], tpr[i], color=color, label='ROC curve of class {0} (area = {1:0.2f})'
                                      ''.format(i, roc_auc[i]))
    # micro & macro ROC #########################################################

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(),y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_class)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_class):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= num_class
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])


    #micro
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),color='mediumturquoise',linestyle='-',linewidth=1.5)
    #macro
    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]), color='hotpink', linestyle='-',linewidth=1.5) 

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    plt.show()
    #plt.savefig('F:/NCKU/YOLOv4-DeepSORT/Model/CNN_LSTM_img/ROC_lstm.png')


# In[ ]:


#ROC_threshold
def ROC_threshold(y_test, y_pred):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_class):
        print('class: ',i)
        fpr[i], tpr[i], thresholds = roc_curve(y_test[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        #計算最佳threshold
        #print('threshold = ',thresholds)
        gmean = np.sqrt(tpr[i] * (1 - fpr[i]))
        #tpr_fpr = tpr[i]-fpr[i]
        #print(tpr_fpr)
        num_index = np.argmax(gmean)
        opt_threshold = thresholds[num_index]
        print('最佳threshold = ', opt_threshold)
    
    
    # Plot ROC curve
    plt.figure(figsize=(7,5))
    plt.grid()
    colors = cycle(['yellowgreen', 'darkorange', 'cornflowerblue','mediumpurple'])
    for i, color in zip(range(num_class), colors):
        fpr[i], tpr[i], thresholds = roc_curve(y_test[:, i], y_pred[:, i])
        plt.plot(fpr[i], tpr[i], color=color, label='ROC curve of class {0} (area = {1:0.2f})'
                                   ''.format(i, roc_auc[i]))
        #畫最佳threshold的點點
        gmean = np.sqrt(tpr[i] * (1 - fpr[i]))
        num_index = np.argmax(gmean)
        print(num_index)
        plt.plot(fpr[i][num_index], tpr[i][num_index], 'o', color='black')
        #print(thresholds)
        threshold_value = round(thresholds[num_index],num_class)
        plt.text(fpr[i][num_index], tpr[i][num_index],threshold_value,ha='center', va='bottom') 



    # micro & macro ROC #########################################################

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(),y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_class)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_class):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= num_class
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])


    #micro
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),color='mediumturquoise',linestyle='-',linewidth=1.5)
    #macro
    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]), color='hotpink', linestyle='-',linewidth=1.5) 

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    plt.show()
    #plt.savefig('F:/NCKU/YOLOv4-DeepSORT/Model/CNN_LSTM_img/ROC_lstm.png')


# In[15]:


#accuracy & loss 在每個epoch的變化
def Epoch_acc_loss (train_history):
    acc = train_history.history['accuracy']
    val_acc = train_history.history['val_accuracy']
    loss = train_history.history['loss']
    val_loss = train_history.history['val_loss']
    
    epochs = range(1, len(acc) + 1)
    
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    ax = plt.subplot()
    ax.set_xlabel('Epochs', fontsize = 20);
    ax.set_ylabel('Accuracy', fontsize = 20);
    #plt.savefig('F:/NCKU/YOLOv4-DeepSORT/ModelCNN_LSTM_img/acc_gru.png')
    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    ax = plt.subplot()
    ax.set_xlabel('Epochs', fontsize = 20);
    ax.set_ylabel('Loss', fontsize = 20);
    
    plt.show()
    #plt.savefig('F:/NCKU/YOLOv4-DeepSORT/Model/CNN_LSTM_img/loss_gru.png')
    plt.figure() 


# **2 class的折線圖 & 直方圖**

# In[13]:


#2class 每frame的預測機率
def predictModel(createModel_new):
    global y_pred_new
    predictingModel = createModel_new(forTraining = False)
    predictingModel.set_weights(trainingModel.get_weights())
    # 宣告一個空陣列來儲存資料，陣列格式為(該資料集有幾個檔案,影片的shape)
    y_pred_new = np.zeros((num_test, num_frame, num_class))
    for i in range(num_test):
        for j in range(75):
            y_pred_new[i][j] = predictingModel.predict([x_test_d_reshape[i][j], x_test_img_reshape[i][j]])
        predictingModel.reset_states()
    ##print(np.shape(y_pred_new)) = #(num_test, num_frame, num_class)
    #print(y_pred_new)


# In[18]:


#2 class預測機率曲線圖  #只plot出被預測正確的樣本
def plot_graph_2class(y_test_bool,color1,color2):
    global y_risk_00, y_risk_11
    
    #predictModel(createModel_new)
    
    #各類別 預測為類別 0 (head-on)的機率 (90 timesteps)
    y_risk_0 = np.delete(y_pred_new, (1), axis = 2) #剩類別0
    y_risk_1 = np.delete(y_pred_new, (0), axis = 2) #剩類別1
    ##print(numpy.shape(y_risk_0))  #= (num_test, num_frame,1)
    #print(y_risk_0)

    timestep = list(range(1,num_frame+1)) #1-75
    y_risk_00 = np.reshape(y_risk_0, (num_test, num_frame))  #
    y_risk_11 = np.reshape(y_risk_1, (num_test, num_frame))
    #print(numpy.shape(y_risk_00)) = (num_test, num_frame)

    #曲線圖表示
    for j in range(num_class):
        y_risk_value = []
        y_risk_label = []
    
        if j == 0:
            rr = y_risk_00
            class_name = 'Head-on & Rear-end'
        if j == 1:
            rr = y_risk_11
            class_name = 'Lateral & Sideswipe'

        for i in range(num_test):
            y_risk_value.append(rr[i])
            y_risk_label.append(i)
        num_class_sample = len(y_risk_value)
        #print(num_class_sample)
        #print(y_risk_label)
        # 設定圖片大小為長15、寬10
        plt.figure(figsize = (12, 5), dpi = 300, linewidth = 2)
        # 把資料放進來並指定對應的X軸、Y軸的資料，用方形做標記(s-)，並指定線條顏色為紅色，使用label標記線條含意 #圓形做標記(o-)
        for i in range(num_class_sample):
            if y_pred_bool[i] == y_test_bool[i]: #只plot出被預測正確的樣本
                #print('預測值: ',y_pred_bool[i],'測試值: ',y_test_bool[i])
                if y_test_bool[i] == 0:
                    sampleLable = "Sample%s" % (y_risk_label[i])
                    l1 ,= plt.plot(timestep, y_risk_value[i], color = color1) #lightcoral
                if y_test_bool[i] == 1:
                    sampleLable = "Sample%s" % (y_risk_label[i])
                    l2 ,= plt.plot(timestep, y_risk_value[i], color = color2) #darkkhaki

        # 設定圖片標題，以及指定字型設定，x代表距圖最左側的距離，y代表與圖的距離
        plt.title("collision risk", x = 0.5, y = 1.0)
        # 設定刻度字體大小
        plt.xticks(fontsize = 12)
        plt.yticks(fontsize = 12)
        # 標示x軸(labelpad代表與圖片的距離)
        plt.xlabel("Timestep (s)", fontsize = 12, labelpad = 10)
        # 標示y軸(labelpad代表與圖片的距離)
        plt.ylabel("Risk", fontsize = 12, labelpad = 10)
        # 顯示出線條標記位置
        plt.legend(handles=[l1, l2], labels=['actual head-on + rear-end', 'actual lateral + sideswipe'],
                   loc = "best", fontsize = 8, bbox_to_anchor = (0.653, 1.0))
        # x軸數值由大到小
        #plt.gca().invert_xaxis()
        # 存為圖檔
        #plt.savefig('/content/gdrive/My Drive/model 5/CNN-LSTM_d_img.png')
        # 畫出圖片
        plt.show()


# In[11]:


#2class直方圖機率表示
def plot_histogram_2class(y_test_bool, y_pred_bool):
    timestep = list(range(1,num_frame+1))
    a = 0
    for i in range(num_test):
        #print('第',i,'個樣本')
        sample_0 = y_risk_00[i]
        sample_1 = y_risk_11[i]

        if y_pred_bool[i] == y_test_bool[i]: #只plot出正確預測的樣本
            if y_test_bool[i] == 0:
                class_name = 'headon'
                CLASS_NAME = 'Head-on'
            if y_test_bool[i] == 1:
                class_name = 'lateral'
                CLASS_NAME = 'Lateral'
            a += 1
            if a == 1: #第一個直方圖才需要右上角的圖標
                #for j in range(class_num):  #(0-8)
                #class_num = num_test_0
                plt.figure(figsize=(20,3))
                plt.style.use('seaborn-darkgrid')
                plt.xticks(np.arange(0,100,10), fontsize = 16)
                plt.yticks(fontsize = 16)
                #plt.subplot(class_num, 1, j+1)  #(9, 1, 1)~(9, 1, 9) #thistle #gold #sandybrown
                plt.bar(timestep,sample_1,color="paleturquoise",label="lateral + sideswipe")
                plt.bar(timestep,sample_0,color="sandybrown",bottom=np.array(sample_1),label="head-on + rear-end")
                plt.title('Probablity of '+CLASS_NAME+' in each frame',fontsize = 26)
                plt.legend(loc="lower left",fontsize = 16, bbox_to_anchor=(0.8,1.0))
                #plt.savefig('F:/NCKU/YOLOv4-DeepSORT/Model/CNN_LSTM_d_img/GRU/collision risk/預測正確/'+class_name+str(i)+'.png')
            else:
                plt.figure(figsize=(20,3))
                plt.style.use('seaborn-darkgrid')
                plt.xticks(np.arange(0,100,10), fontsize = 16)
                plt.yticks(fontsize = 16)
                #plt.subplot(class_num, 1, j+1)  #(9, 1, 1)~(9, 1, 9) #thistle #gold
                plt.bar(timestep,sample_1,color="paleturquoise",label="lateral + sideswipe")
                plt.bar(timestep,sample_0,color="sandybrown",bottom=np.array(sample_1),label="head-on + rear-end")
                #plt.savefig('F:/NCKU/YOLOv4-DeepSORT/Model/CNN_LSTM_d_img/GRU/collision risk/預測正確/'+class_name+str(i)+'.png')
        else:
            continue

    plt.show()


# **4 class的折線圖**

# In[16]:


#定義預測機率曲線圖 # 4 class
#test sample在四分類下 75 timesteps的 預測機率
def plot_graph(trainingModel, y_test_bool):
    global y_risk_00, y_risk_11, y_risk_22, y_risk_33
    global y_pred_new
    
    predictingModel = createModel_new(forTraining = False)
    predictingModel.set_weights(trainingModel.get_weights())
    # 宣告一個空陣列來儲存資料，陣列格式為(該資料集有幾個檔案,影片的shape)
    y_pred_new = np.zeros((num_test, num_frame, num_class))
    for i in range(num_test):
        for j in range(75):
            y_pred_new[i][j] = predictingModel.predict([x_test_d_reshape[i][j], x_test_img_reshape[i][j]])
        predictingModel.reset_states()
    ##print(np.shape(y_pred_new)) = #(num_test, num_frame, num_class)
    print(y_pred_new)


    #各類別 預測為類別 0 (head-on)的機率 (90 timesteps)
    y_risk_0 = np.delete(y_pred_new, (1,2,3), axis = 2) #剩類別0
    y_risk_1 = np.delete(y_pred_new, (0,2,3), axis = 2) #剩類別1
    y_risk_2 = np.delete(y_pred_new, (0,1,3), axis = 2) #剩類別2
    y_risk_3 = np.delete(y_pred_new, (0,1,2), axis = 2) #剩類別3
    ##print(numpy.shape(y_risk_0))  #= (num_test, num_frame,1)
    #print(y_risk_0)

    timestep = list(range(1,num_frame+1))
    y_risk_00 = np.reshape(y_risk_0, (num_test, num_frame))  #
    y_risk_11 = np.reshape(y_risk_1, (num_test, num_frame))
    y_risk_22 = np.reshape(y_risk_2, (num_test, num_frame))
    y_risk_33 = np.reshape(y_risk_3, (num_test, num_frame))
    #print(numpy.shape(y_risk_00)) = (num_test, num_frame)

    #曲線圖表示
    for j in range(num_class):
        y_risk_value = []
        y_risk_label = []
    
        if j == 0:
            rr = y_risk_00
            class_name = 'Head-on'
        if j == 1:
            rr = y_risk_11
            class_name = 'Lateral'
        if j == 2:
            rr = y_risk_22
            class_name = 'Sideswipe'
        if j == 3:
            rr = y_risk_33
            class_name = 'Rear-end'
        for i in range(num_test):
            y_risk_value.append(rr[i])
            y_risk_label.append(i)
        num_class_sample = len(y_risk_value)
        #print(num_class_sample)
        #print(y_risk_label)
        # 設定圖片大小為長15、寬10
        plt.figure(figsize = (12, 5), dpi = 300, linewidth = 2)
        # 把資料放進來並指定對應的X軸、Y軸的資料，用方形做標記(s-)，並指定線條顏色為紅色，使用label標記線條含意 #圓形做標記(o-)
        for i in range(num_class_sample):
            if y_test_bool[i] == 0:
                sampleLable = "Sample%s" % (y_risk_label[i])
                l1 ,= plt.plot(timestep, y_risk_value[i], color = 'lightcoral')
            if y_test_bool[i] == 1:
                sampleLable = "Sample%s" % (y_risk_label[i])
                l2 ,= plt.plot(timestep, y_risk_value[i], color = 'lightskyblue')
            if y_test_bool[i] == 2:
                sampleLable = "Sample%s" % (y_risk_label[i])
                l3 ,= plt.plot(timestep, y_risk_value[i], color = 'mediumpurple')
            if y_test_bool[i] == 3:
                sampleLable = "Sample%s" % (y_risk_label[i])
                l4 ,= plt.plot(timestep, y_risk_value[i], color = 'darkkhaki')
        # 設定圖片標題，以及指定字型設定，x代表距圖最左側的距離，y代表與圖的距離
        plt.title("collision risk in "+ class_name, x = 0.5, y = 1.0)
        # 設定刻度字體大小
        plt.xticks(fontsize = 12)
        plt.yticks(fontsize = 12)
        # 標示x軸(labelpad代表與圖片的距離)
        plt.xlabel("Timestep (s)", fontsize = 12, labelpad = 10)
        # 標示y軸(labelpad代表與圖片的距離)
        plt.ylabel("Risk", fontsize = 12, labelpad = 10)
        # 顯示出線條標記位置
        plt.legend(handles=[l1, l2, l3, l4], labels=['actual head-on', 'actual lateral', 'actual sideswipe', 'actual rear-end'],
                   loc = "best", fontsize = 8, bbox_to_anchor = (0.653, 1.0))
        # x軸數值由大到小
        #plt.gca().invert_xaxis()
        # 存為圖檔
        #plt.savefig('/content/gdrive/My Drive/model 5/CNN-LSTM_d_img.png')
        # 畫出圖片
        plt.show()


# In[12]:


#儲存模型
trainingModel.save('F:/NCKU/YOLOv4-DeepSORT/選取ID資料/Model/CNN_LSTM_d_img/model_save/model_lstm_4class_A_normalize.h5') 

