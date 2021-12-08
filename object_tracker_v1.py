import os
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
import pandas as pd
from pandas import ExcelWriter
from openpyxl import load_workbook
from os.path import basename

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
#falg.DEFINE_string(變數名稱，預設值，用法描述)
flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', './data/video/test.mp4', 'path to input video or set to 0 for webcam')
#flags.DEFINE_string('output', None, 'path to output video')
#編碼格式設為XVID
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_boolean('info', False, 'show detailed info of tracked objects')
flags.DEFINE_boolean('count', False, 'count objects being tracked on screen')

def main(_argv):
    #coordinate_data = pd.DataFrame(columns=['Frame','ID','xmin','ymin','xmax','ymax'])
    #coordinate_data=pd.read_excel('D:/NCKU/YOLOv4-DeepSORT/yolov4-deepsort/excel_output/coordinate_data.xlsx')
    new_sheet = pd.DataFrame(columns=['Frame','ID','xmin','ymin','xmax','ymax'])
    # Definition of the parameters
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 1.0
    
    # initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # initialize tracker
    tracker = Tracker(metric)

    # load configuration for object detector
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    #video_path = FLAGS.video
    test=pd.read_excel('D:/NCKU/YOLOv4-DeepSORT/存取影片檔名練習/test.xlsx','run')
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
            print('他什麼都不是')
        video_path = 'D:/NCKU/YOLOv4-DeepSORT/yolov4-deepsort/eve_video/'+str(j)+'/'+str(i[0])+'.mp4'
        output_path = 'D:/NCKU/YOLOv4-DeepSORT/yolov4-deepsort/eve_video/output/'+str(i[1])+'/'+str(i[0])+'.avi'
    # load tflite model if flag is set
        if FLAGS.framework == 'tflite':
            interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            print(input_details)
            print(output_details)
    # otherwise load standard tensorflow saved model
        else:
            saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
            infer = saved_model_loaded.signatures['serving_default']

    # begin video capture
        try:
            vid = cv2.VideoCapture(int(video_path))
        except:
            vid = cv2.VideoCapture(video_path)

        out = None
    #從路徑中擷取video name
        file_name= os.path.basename(video_path) #從路徑中分離出檔名
        video_name=os.path.splitext(file_name)[0]
        print('此影片檔名：',video_name)
    # get video ready to save locally if flag is set
    #if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format) #編碼格式設為XVID
    #XVID是遵循MPEG-4視頻編碼標準的視頻編解碼器庫
    #cv2.VideoWriter(指定輸出的檔名, 指定 FourCC, fps, frameSize 影像大小)
        out = cv2.VideoWriter(output_path,codec, fps, (width, height))

        frame_num = 0
    # while video is running
        while True:
            return_value, frame = vid.read()
            if return_value:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame)
            else:
                print('Video has ended or failed, try a different video format!')
                break
            frame_num +=1
            print('Frame #: ', frame_num)
            frame_size = frame.shape[:2]
            image_data = cv2.resize(frame, (input_size, input_size))
            image_data = image_data / 255.
            image_data = image_data[np.newaxis, ...].astype(np.float32)
            start_time = time.time()

        # run detections on tflite if flag is set
            if FLAGS.framework == 'tflite':
                interpreter.set_tensor(input_details[0]['index'], image_data)
                interpreter.invoke()
                pred = [interpreter.get_tensor(output_details[i]['index']) for i in     range(len(output_details))]
            # run detections using yolov3 if flag is set
                if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
                    boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                    input_shape=tf.constant([input_size, input_size]))
                else:
                    boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                    input_shape=tf.constant([input_size, input_size]))
            else:
                batch_data = tf.constant(image_data)
                pred_bbox = infer(batch_data)
                for key, value in pred_bbox.items():
                    boxes = value[:, :, 0:4]
                    pred_conf = value[:, :, 4:]

            boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )

        # convert data to numpy arrays and slice out unused elements
            num_objects = valid_detections.numpy()[0]
            bboxes = boxes.numpy()[0]
            bboxes = bboxes[0:int(num_objects)]
            scores = scores.numpy()[0]
            scores = scores[0:int(num_objects)]
            classes = classes.numpy()[0]
            classes = classes[0:int(num_objects)]

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
            original_h, original_w, _ = frame.shape
            bboxes = utils.format_boxes(bboxes, original_h, original_w)

        # store all predictions in one parameter for simplicity when calling functions
            pred_bbox = [bboxes, scores, classes, num_objects]

        # read in all class names from config
            class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # by default allow all classes in .names file
        # allowed_classes = list(class_names.values())
        
        # custom allowed classes (uncomment line below to customize tracker for only people)
            allowed_classes = ['person','bicycle','car','motorbike','bus','truck']

        # loop through objects and use class index to get class name, allow only classes in allowed_classes list
            names = []
            deleted_indx = []
            for i in range(num_objects):
                class_indx = int(classes[i])
                class_name = class_names[class_indx]
                if class_name not in allowed_classes:
                    deleted_indx.append(i)
                else:
                    names.append(class_name)
            names = np.array(names)
            count = len(names)
            if FLAGS.count:
                cv2.putText(frame, "Objects being tracked: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
                print("Objects being tracked: {}".format(count))
        # delete detections that are not in allowed_classes
            bboxes = np.delete(bboxes, deleted_indx, axis=0)
            scores = np.delete(scores, deleted_indx, axis=0)

        # encode yolo detections and feed to tracker
            features = encoder(frame, bboxes)
            detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

        #initialize color map
            cmap = plt.get_cmap('tab20b')
            colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima supression
            boxs = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            classes = np.array([d.class_name for d in detections])
            indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
            detections = [detections[i] for i in indices]       

        # Call the tracker
            tracker.predict()
            tracker.update(detections)

        # update tracks
            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue 
                bbox = track.to_tlbr()
                class_name = track.get_class()
            
        # draw bbox on screen
                color = colors[int(track.track_id) % len(colors)]
                color = [i * 255 for i in color]
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
                cv2.putText(frame, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
            
                new_sheet=new_sheet.append({'Frame':frame_num,'ID':str(track.track_id),'xmin':int(bbox[0]), 'ymin':int(bbox[1]), 'xmax':int(bbox[2]), 'ymax':int(bbox[3])}, ignore_index=True)

        # if enable info flag then print details about each track
                if FLAGS.info:
                    print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))

        # calculate frames per second of running detections
            fps = 1.0 / (time.time() - start_time)
            print("FPS: %.2f" % fps)
            result = np.asarray(frame)
            result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        #cv2.cvtColor(src, code[, dst[, dstCn]]) 用於將圖像從一種顏色空間轉換為另一種顏色空間
        #src:要更改其色彩空間的圖像。code:色彩空間轉換代碼。dst:與src圖像大小和深度相同的輸出圖像，是一個可選參數。dstCn:目標圖像中的頻道數。如果參數為0，則通道數自動從src和代碼得出，是一個可選參數。
            if not FLAGS.dont_show:
                cv2.imshow("Output Video", result)
        
        # if output flag is set, save video file
        #if FLAGS.output:
        out.write(result)  # 寫入影格
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        cv2.destroyAllWindows()
    #儲存影片
    
    #pp = 'D:/NCKU/YOLOv4-DeepSORT/yolov4-deepsort/excel_output/'+str(video_name)+'.xlsx'
    #if not os.path.isdir('coordinate_data'):
        w = ExcelWriter('D:/NCKU/YOLOv4-DeepSORT/yolov4-deepsort/excel_output/coordinate_data.xlsx', engine='openpyxl')
        book = load_workbook('D:/NCKU/YOLOv4-DeepSORT/yolov4-deepsort/excel_output/coordinate_data.xlsx')
        w.book=book
        new_sheet.to_excel(w, sheet_name = video_name,index=False)
        w.save()
        new_data=pd.read_excel('D:/NCKU/YOLOv4-DeepSORT/yolov4-deepsort/excel_output/coordinate_data.xlsx',video_name)
    
        print(new_data)
        print('完成',video_name,'辨識 !')

if __name__ == '__main__': #讓檔案在被引用時，不該執行的程式碼不被執行
    try:
        app.run(main)
    except SystemExit:
        pass
