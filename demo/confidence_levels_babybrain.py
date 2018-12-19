import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread
from config import load_config
from nnet import predict
# from util import visualize
from dataset.pose_dataset import data_to_input


def calculate_confidence(cfg, scmap, threshold=0.2):
    # Get joints processed from id
    all_joints = cfg.all_joints
    all_joints_names = cfg.all_joints_names

    confidences = []
    for pidx, part in enumerate(all_joints):
        # Calculate the result map for a joint (pre-heatmap)
        scmap_part = np.sum(scmap[:, :, part], axis=2)

        # For a part, we average every conf point higher than a threshold
        data = [x for x in scmap_part.flatten() if x > threshold]
        confidences.append(np.mean(data))
    return confidences, all_joints_names


def main():
    cfg = load_config("/home/babybrain/PycharmProjects/pose-tensorflow/demo/pose_cfg_babybrain.yaml")

    # Load and setup CNN part detector
    sess, inputs, outputs = predict.setup_pose_prediction(cfg)

    # Read images from file
    filepath = '/home/babybrain/Escritorio/000790'
    filepath_segmented = '/home/babybrain/Escritorio/frames_seg_000790'
    outpath = '/home/babybrain/Escritorio/confidences.png'
    file_list = os.listdir(filepath)
    file_list_segmented = os.listdir(filepath_segmented)
    frame_confidence = []
    frame_confidence_segmented = []

    # Subsample: iterate each n elements
    subsample = 480  # 1 is no subsample
    file_list = [afile for num, afile in enumerate(file_list) if num % subsample == 0]
    file_list_segmented = [afile for num, afile in enumerate(file_list_segmented) if num % subsample == 0]

    # Non-segmented files
    for num, afile in enumerate(file_list):

        print("{num} non-segmented files left".format(num=len(file_list)-num))
        infile = "{path}/{name}".format(path=filepath, name=afile)
        image = imread(infile, mode='RGB')
        image_batch = data_to_input(image)

        # Compute prediction with CNN
        outputs_np = sess.run(outputs, feed_dict={inputs: image_batch})
        scmap, locref, _ = predict.extract_cnn_output(outputs_np, cfg)

        # Obtain a vector with the max confidence values for each body part
        confidences, _ = calculate_confidence(cfg, scmap)
        frame_confidence.append(confidences)

    # Segmented files
    for num, afile in enumerate(file_list_segmented):

        print("{num} segmented files left".format(num=len(file_list_segmented)-num))
        infile = "{path}/{name}".format(path=filepath_segmented, name=afile)
        image = imread(infile, mode='RGB')
        image_batch = data_to_input(image)

        # Compute prediction with CNN
        outputs_np = sess.run(outputs, feed_dict={inputs: image_batch})
        scmap, locref, _ = predict.extract_cnn_output(outputs_np, cfg)

        # Obtain a vector with the max confidence values for each body part
        confidences, _ = calculate_confidence(cfg, scmap)
        frame_confidence_segmented.append(confidences)

    # Calculate mean and std for each part
    confidences = np.array(frame_confidence)
    confidences_mean = np.nanmean(confidences, axis=0)
    confidences_std = np.nanstd(confidences, axis=0)

    confidences_segmented = np.array(frame_confidence_segmented)
    confidences_segmented_mean = np.nanmean(confidences_segmented, axis=0)
    confidences_segmented_std = np.nanstd(confidences_segmented, axis=0)

    fig, ax = plt.subplots()
    bar_width = 0.35
    opacity = 0.4
    error_config = {'ecolor': '0.3'}
    n_groups = len(confidences_mean)
    index = np.arange(n_groups)

    # fig.tight_layout()
    # Non-segmented
    ax.bar(index, confidences_mean, bar_width, alpha=opacity, color='b', yerr=confidences_std,
           error_kw=error_config, label='Human-Pose')

    # Segmented
    ax.bar(index + bar_width, confidences_segmented_mean, bar_width, alpha=opacity, color='r',
           yerr=confidences_segmented_std, error_kw=error_config, label='Human Pose Segmented')

    ax.set_xlabel('Body-Part')
    ax.set_ylabel('Confidence Value')
    ax.set_title('Mean Confidence Values in one video')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(cfg.all_joints_names)
    ax.legend()

    plt.savefig(outpath, dpi=600)


if __name__ == '__main__':
    main()
