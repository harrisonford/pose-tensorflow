import json
import numpy as np
from config import load_config
from nnet import predict
from scipy.misc import imread
from dataset.pose_dataset import data_to_input
import matplotlib.pyplot as plt


# import json annotations as a dictionary
def load_annotations(json_path):

    with open(json_path) as f:
        annotations = json.load(f)

    return annotations


# load x, y for a certain data id (r-ankle, l-shoulder, etc)
def get_xy_for(part, annotations):

    frame = []
    x = []
    y = []
    frames = annotations['_via_img_metadata']
    for key, elements in frames.items():
        if elements['regions']:  # then it has annotations
            frame.append(elements['filename'])
            region_list = elements['regions']
            for region in region_list:
                if region['region_attributes']['id'] == part:
                    x.append(region['shape_attributes']['cx'])
                    y.append(region['shape_attributes']['cy'])

    return np.array(frame), np.array(x), np.array(y)


# calculate distances of output array of limb compared to ground truth
# both arrays should be same length
def calculate_distances(x_array, y_array, truth_x_array, truth_y_array, normalized=True):

    distances = [np.hypot(abs(x1-x2), abs(y1-y2)) for x1, y1, x2, y2 in zip(x_array, y_array,
                                                                            truth_x_array, truth_y_array)]
    distances = np.array(distances)

    if normalized:
        distances = (distances - distances.min())/(distances.max() - distances.min())
    return distances


# from distances array calculate the detection rate vs (normalized) distance data to plot
def detection_rate(distance_array, nsteps=10, normalized=True):
    distance_steps = np.linspace(0, distance_array.max(), nsteps)
    rates = np.empty(len(distance_steps))

    for index, a_distance in enumerate(distance_steps):
        rates[index] = np.sum(distance_array < a_distance)
    rates = np.array(rates)

    if normalized:
        rates = rates / len(distance_array)

    return distance_steps, rates


# make the graph!
def main():

    # paths to setup
    annotation_path = '/home/babybrain/Escritorio/300145_via.json'
    frames_path = '/home/babybrain/Escritorio/300145'

    # get the x-y anotations for each frame
    annotations = load_annotations(annotation_path)

    # get x, y positions for a certain part
    part_id_index = 4  # we'll get elbows, need left and right (algorithm doesn't discriminate)
    file_anno, x_anno_r, y_anno_r = get_xy_for('r-elbow', annotations)
    _, x_anno_l, y_anno_l = get_xy_for('l-elbow', annotations)

    # get the x,y model prediction for each frame annotated
    cfg = load_config("/home/babybrain/PycharmProjects/pose-tensorflow/demo/pose_cfg_babybrain.yaml")
    # Load and setup CNN part detector
    sess, inputs, outputs = predict.setup_pose_prediction(cfg)

    # run session for each frame image annotated
    x_model = np.empty(len(file_anno))
    y_model = np.empty(len(file_anno))
    for index, an_image in enumerate(file_anno):
        infile = "{path}/{name}".format(path=frames_path, name=an_image)
        image = imread(infile, mode='RGB')
        image_batch = data_to_input(image)

        # Compute prediction with CNN
        outputs_np = sess.run(outputs, feed_dict={inputs: image_batch})
        scmap, locref, _ = predict.extract_cnn_output(outputs_np, cfg)

        # Extract maximum scoring location from the heatmap, assume 1 person
        pose = predict.argmax_pose_predict(scmap, locref, cfg.stride)
        x_model[index] = pose[part_id_index, 0]
        y_model[index] = pose[part_id_index, 1]

    # now calculate distances
    distances_r = calculate_distances(x_model, y_model, x_anno_r, y_anno_r)
    distances_l = calculate_distances(x_model, y_model, x_anno_l, y_anno_l)

    # merge the best distance results
    distances = [min(xr, xl) for xr, xl in zip(distances_r, distances_l)]
    distances = np.array(distances)

    distance_steps, rates = detection_rate(distances, nsteps=50)
    rates = rates*100

    # finally plot the graph
    fig, ax = plt.subplots()
    ax.plot(distance_steps, rates)

    ax.set_xlabel('Normalized Distance')
    ax.set_ylabel('Detection %')
    ax.set_title('Distance threshold vs Detection Ratio')
    ax.set_xlim([0, 0.5])

    plt.show()


if __name__ == '__main__':
    main()
