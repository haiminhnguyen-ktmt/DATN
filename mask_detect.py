from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import cv2
import numpy as np
import facenet
import detect_face
import os
import time
import pickle
from PIL import Image
import tensorflow.compat.v1 as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from  mtcnn import MTCNN
video= 0
modeldir = './model/20180402-114759.pb'
classifier_filename = './class/classifier.pkl'
npy='./npy'
train_img="./train_img"
with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess, npy)
        minsize = 30  # minimum size of face
        threshold = [0.7,0.8,0.8]  # three steps's threshold
        factor = 0.709  # scale factor
        margin = 44
        batch_size =100 #1000
        image_size = 182
        input_image_size = 160
        HumanNames = os.listdir(train_img)
        HumanNames.sort()
        print('Loading Model')
        facenet.load_model(modeldir)
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        embedding_size = embeddings.get_shape()[1]
        classifier_filename_exp = os.path.expanduser(classifier_filename)
        with open(classifier_filename_exp, 'rb') as infile:
            (model, class_names) = pickle.load(infile,encoding='latin1')

        video_capture = cv2.VideoCapture(video)
        print('Start Recognition')
        maskNet = load_model("model_mask.h5",compile= False)
        # print(maskNet.summary())
        detector = MTCNN()
        while True:
            ret, frame = video_capture.read()
            #frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)    #resize frame (optional)
            timer =time.time()
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            # detector = MTCNN()
            results = detector.detect_faces(frame)
            print(results)
            if results != []:
                # det = [:, 0:4]
                # det = bounding_boxes[:, 0:4]
                dem=0
                img_size = np.asarray(frame.shape)[0:2]
                cropped = []
                scaled = []
                scaled_reshape = []
                for i in range(len(results)):

                    xmin,ymin,width,heigh = results[i]['box']
                    xmax =xmin+width
                    ymax = ymin+heigh
                    emb_array = np.zeros((1, embedding_size))
                    xmax= int(xmax)
                    xmin = int(xmin)
                    ymin= int(ymin)
                    ymax =int(ymax)
                    # cropped.append(frame[ymin:ymax, xmin:xmax,:])
                    # cropped[i] = facenet.flip(cropped[i], False)
                    try:
                        # inner exception
                        if xmin <= 0 or ymin <= 0 or xmax >= len(frame[0]) or ymax >= len(frame):
                            print('Face is very close!')
                            continue
                        face= frame[ymin:ymax, xmin:xmax]
                        face2 =face
                        # print("hai")
                        # print("hai1")
                        # face=cv2.cvtColor(face,cv2.COLOR_BGR2RGB)
                        # print("hai2")
                        face=cv2.resize(face,(224,224))
                        # print("hai3")
                        face=img_to_array(face)
                        # print("hai4")
                        face=preprocess_input(face)
                        # print("hai5")
                        face = np.expand_dims(face,axis=0)
                        mask = maskNet.predict(face)
                        # print("hai6")
                        if mask[0][0] < mask[0][1]:
                            label ="NOT MASK"
                            color=(0,255,0) if label=='MASK' else (0,0,255)

                            cv2.putText(frame,label,(xmin,ymin-35),cv2.FONT_HERSHEY_SIMPLEX,0.45,color,2)

                            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)


                        else:
                            label = "MASK"
                        # label='Mask' if mask[0][0]>mask[0][1] else 'No Mask'
                            color=(0,255,0) if label=='MASK' else (0,0,255)

                            cv2.putText(frame,label,(xmin,ymin-15),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)

                            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                            # cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                            # cv2.rectangle(frame, (xmin, ymin-20), (xmax, ymin-2), (0, 255,255), 1)
                            # cv2.putText(frame, "?", (xmin,ymin-5), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            #                     1, (0, 0, 0), thickness=1, lineType=1)

                        # cv2.rectangle(frame, (xmin, ymin-20), (xmax, ymin-2),color, -1)
                        # print("minh")

                    except:

                        print("error")

            endtimer = time.time()
            fps = 1/(endtimer-timer)
            cv2.rectangle(frame,(15,30),(135,60),(0,255,255),-1)
            cv2.putText(frame, "fps: {:.2f}".format(fps), (20, 50),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
            cv2.imshow('Face Recognition', frame)
            key= cv2.waitKey(1)
            if key== 113: # "q"
                break
        video_capture.release()
        cv2.destroyAllWindows()


