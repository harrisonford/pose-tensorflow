import os
import sys

sys.path.append(os.path.dirname(__file__) + "/../")

from scipy.misc import imread

from config import load_config
from nnet import predict
from util import visualize
from dataset.pose_dataset import data_to_input


cfg = load_config("/home/babybrain/PycharmProjects/pose-tensorflow/demo/pose_cfg_babybrain.yaml")

# Load and setup CNN part detector
sess, inputs, outputs = predict.setup_pose_prediction(cfg)

# Read images from file
filepath = '/home/babybrain/Escritorio/frames_seg_000790'
outpath = '/home/babybrain/Escritorio/frames_seg_000790_results'
file_list = os.listdir(filepath)
for afile in file_list:
    infile = "{path}/{name}".format(path=filepath, name=afile)
    image = imread(infile, mode='RGB')
    image_batch = data_to_input(image)

    # Compute prediction with CNN
    outputs_np = sess.run(outputs, feed_dict={inputs: image_batch})
    scmap, locref, _ = predict.extract_cnn_output(outputs_np, cfg)

    # Extract maximum scoring location from the heatmap, assume 1 person
    pose = predict.argmax_pose_predict(scmap, locref, cfg.stride)

    # Visualize and save
    outfile = "{path}/{name}".format(path=outpath, name=afile)
    visualize.show_heatmaps_babybrain(cfg, image, scmap, pose, save_path=outfile,
                                      show=False, joint_list=[[0, 5], [6, 11]])
