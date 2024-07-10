#!/usr/bin/env python3

"""Create walker fits following the Troje walker model approach

Original: A gait data set which was already put into the right structure and preprocessed
Result: Update to the metadata file of each subject

in_path: Path to the folder with the .bsor files
"""

import time

import numpy as np
import sys
import yaml
import json
from hyperopt import hp, tpe, Trials, fmin
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from scipy.optimize import shgo
from os import listdir
from os.path import isfile, join


def create_walker_fits(data, fps):
    combined_sequences = []
    for sample in data:
        sample = np.array(sample)
        sample = transform_walker_to_treadmill(sample, static_point=5, axis=1)
        combined_sequences += sample.tolist()

    # with open(join("data/", file), "w") as f:
    #    json.dump(combined_sequences, f)

    pca = PCA(n_components=4)
    pca.fit(sample)
    transformed_walker = pca.transform(combined_sequences)
    fits = fit_model_to_walker_hyperopt(transformed_walker, fps)

    return fits


def plot_sinus_fit(x, y, y2):
    fig, ax = plt.subplots()
    ax.plot(x, y, color="red")
    ax.plot(x, y2, color="green")

    ax.set(xlabel="time (s)", ylabel="amplitude", title="Plot")
    ax.grid()

    plt.show()


def resample_to_fix_fps(data, original_fps, number_of_frames):
    new_poses = []

    new_fps = number_of_frames * (original_fps / len(data))

    new_indices = np.linspace(0, len(data) - 1, num=number_of_frames)
    for i in range(number_of_frames):
        start = int(new_indices[i] // 1)
        decimals = new_indices[i] % 1
        end = int(start + 1)

        if decimals == 0:
            new_poses.append(data[start])
        else:
            tmp = (data[start] * (1.0 - decimals)) + (data[end] * decimals)
            new_poses.append(tmp)

    return np.array(new_poses), new_fps


# Keep the motions of the walker relative to a given point for a specific axis
def transform_walker_to_treadmill(data, static_point=5, axis=1):
    for pose in data:
        tmp_fix = pose[3 * static_point + axis - 1]
        pose[axis - 1 :: 3] = pose[axis - 1 :: 3] - tmp_fix

    return data


# Sinus fit of the first 4 pca components as described by Troje et. al.
def fit_model_to_walker_hyperopt(transformed_postures, fps):
    # Config for the Torje dataset
    w0 = 1
    w1 = 15
    result = []
    for i in range(4):
        X = np.array(list(range(len(transformed_postures))))
        # Scale time to seconds.
        X = np.divide(X, fps)
        y = transformed_postures[:, i]

        def objective(a1, w, f):
            """Objective function to minimize"""
            return np.mean((a1 * np.sin(w * X + f) - y) ** 2)

        def objective_scipy(param):
            tmp = np.mean((param[0] * np.sin(param[1] * X + param[2]) - y) ** 2)
            return tmp

        def objective2(args):
            return objective(*args)

        # Config for the Troje dataset
        # space = [hp.uniform("a1", 0, 1000), hp.uniform("w", w0, w1), hp.uniform("f", -400, 400)]

        # Config for the Host dataset
        if i == 0:
            space = [hp.uniform("a1", 1200, 1800), hp.uniform("w", w0, w1), hp.uniform("f", 1200, 1500)]
        elif i == 1:
            space = [hp.uniform("a1", 200, 300), hp.uniform("w", w0, w1), hp.uniform("f", 900, 1500)]
        else:
            space = [hp.uniform("a1", 100, 200), hp.uniform("w", w0, w1), hp.uniform("f", -400, 400)]

        # space = [hp.uniform('a1', -500, 500),
        #           hp.uniform('w', w0, w1),
        #           hp.uniform('f', -1, 1)]

        tpe_algo = tpe.suggest
        tpe_trials = Trials()

        seed = 34562

        # bounds = [(1000,1500),(w0, w1), (900, 1500)]
        # sci_results = shgo(objective_scipy, bounds, iters=5)
        # print(sci_results.x)
        # print(sci_results.fun)

        params = fmin(fn=objective2, space=space, algo=tpe_algo, trials=tpe_trials, max_evals=200, rstate=np.random.RandomState(seed))

        # print(params)

        def sine_func(x, amp, omega, phi):
            return amp * np.sin(omega * x + phi)

        y2 = []
        for x in X:
            y2.append(sine_func(x, params["a1"], params["w"], params["f"]))
            # y2.append(sine_func(x, sci_results.x[0], sci_results.x[1], sci_results.x[2]))

        loss = objective(params["a1"], params["w"], params["f"])
        params["loss"] = loss
        result.append(params)
        # plot_sinus_fit(X, y, y2)

        if i == 0:
            w0 = params["w"]
            w1 = params["w"] + 0.001
        if i == 1:
            w0 = 2 * w0
            w1 = 2 * w1

    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        exit(1)

    data_folder = sys.argv[1]

    onlyfiles = [f for f in listdir(data_folder) if isfile(join(data_folder, f))]

    json_files = [f for f in onlyfiles if ".json" in f]
    yaml_files = [f for f in onlyfiles if ".yaml" in f]

    walker_data = {}
    for file in json_files:
        with open(join(data_folder, file), "r") as f:
            id = file.split(".")[0]
            if id not in walker_data:
                walker_data[id] = []

            walker_data[id].append(json.load(f))

    for i in walker_data:
        print(i)
        fits = create_walker_fits(np.array(walker_data[i]), 250)

        for file in yaml_files:
            if i == file.split(".")[0]:
                with open(join(data_folder, file), "r") as f:
                    metadata = json.load(f)

                metadata["fits"] = fits

                with open(join(data_folder, file), "w") as f:
                    json.dump(metadata, f)
