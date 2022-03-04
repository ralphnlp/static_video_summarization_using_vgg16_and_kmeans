import os
import numpy as np
import cv2
from keras.applications.vgg16 import VGG16
from keras.models import Model
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from tqdm import tqdm
import argparse

def load_frames(frames_dir):

    frames, frame_iths = [], []

    for frame_name in tqdm(os.listdir(frames_dir)):
        frame_path = os.path.join(frames_dir, frame_name)
        frame_ith = frame_name.split('.')[0]
        frame = cv2.imread(frame_path)
        frame = cv2.resize(frame, (224, 224))
        frames.append(frame)
        frame_iths.append(frame_ith)   

    frames = np.asarray(frames)
    frame_iths = np.asarray(frame_iths, dtype=np.uint8)
    print('Shape dataset: ', frames.shape, frame_iths.shape)
    return (frames, frame_iths)



def frames2vectors(frames):

    vgg16 = VGG16()
    feature_extractor = Model(inputs=vgg16.input, outputs=vgg16.layers[-2].output)
    feature_extractor.summary()

    feature_vectors = []
    for frame in tqdm(frames):
        temp = frame.reshape(1, 224, 224, 3)
        feature_vector = feature_extractor(temp)
        feature_vectors.append(feature_vector[0])
    
    feature_vectors = np.asarray(feature_vectors)
    print('Shape feature vectors', feature_vectors.shape)
    return feature_vectors



def pick_frames(feature_vectors, frame_iths):

    n_clusters = int(len(frame_iths) * 0.15)
    km = KMeans(n_clusters = n_clusters)
    km.fit(feature_vectors)

    picked_idx_frames = []
    closest, _ = pairwise_distances_argmin_min(km.cluster_centers_, feature_vectors)
    for i in closest:
        picked_idx_frames.append(frame_iths[i])

    picked_idx_frames.sort()
    return picked_idx_frames



def merge(frames_dir, picked_idx_frames, keyframes_dir, video_output_path):

    count = 1
    for id_frame in picked_idx_frames:

        frame_name = '0'*(5 - len(str(id_frame))) + str(id_frame)
        keyframe_name = '0'*(5 - len(str(count))) + str(count)
        
        frame_path = os.path.join(frames_dir, frame_name)
        keyframe_path = os.path.join(keyframes_dir, keyframe_name)

        cmd = 'cp {}.png {}.png'.format(frame_path, keyframe_path)
        os.system(cmd)
        count += 1
    
    cmd = 'ffmpeg -framerate 2 -i {}/%05d.png {}'.format(keyframes_dir, video_output_path)
    os.system(cmd)



if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('input_video_path')
    parser.add_argument('output_video_path')

    args = parser.parse_args()
    video_input_path = args.input_video_path
    video_output_path = args.output_video_path
    frames_dir = './frames'
    keyframes_dir = './keyframes'

    if os.path.exists(frames_dir):
        os.removedirs(frames_dir)
    if os.path.exists(keyframes_dir):
        os.removedirs(keyframes_dir)

    os.makedirs(frames_dir)
    os.makedirs(keyframes_dir)


    print('Sampling Frames...')
    cmd = 'ffmpeg -i {} -r 1/1 {}/%05d.png'.format(video_input_path, frames_dir)
    os.system(cmd)

    print('\nLoading Frames...')
    frames, frame_iths = load_frames(frames_dir)

    print('\nExtracting Feature Frames...')
    feature_vectors = frames2vectors(frames=frames)

    print('\nPicking Key Frame...')
    picked_idx_frames = pick_frames(feature_vectors=feature_vectors, frame_iths=frame_iths)

    print('\nMerging Key Frame...')
    merge(frames_dir, picked_idx_frames, keyframes_dir, video_output_path)

    os.system('rm -rf frames')
    os.system('rm -rf keyframes')
    print('\nDone to create Static Video Summary.')