import argparse

import matplotlib.pyplot as plt
import yaml
import copy

from generate_run_configs import generate_configs


def calc_auc(xs, ys):
    auc = 0
    for i in range(len(xs) - 1):
        auc += (xs[i] - xs[i + 1]) * ys[i + 1]
    return round(auc / 100, 0)


def make_utility_vs_privacy_plot_multiple_anon(data, title, anon_names, boundaries):
    fig, ax = plt.subplots()
    clmap = plt.get_cmap("tab10").colors
    markers = [".", "v", "*", "+", "1", ">", "p"]
    marker_size = 100

    i = 0
    for ele in data:
        # Add the start and end on the axis
        tmp_xs = [100] + ele[0] + [min(ele[0])]
        tmp_ys = [max(ele[1])] + ele[1] + [0]

        auc = calc_auc(tmp_xs, tmp_ys)

        ax.fill_between(tmp_xs, tmp_ys, alpha=0.1, step="pre")
        ax.scatter(ele[0], ele[1], color=clmap[i], label=anon_names[i] + ", AUC: " + str(auc), marker=markers[i], s=marker_size)
        ax.plot(tmp_xs, tmp_ys, drawstyle="steps")
        i += 1

    for ele in boundaries:
        if "utility" in ele:
            ax.hlines(boundaries[ele], 0, 100, color="grey", linestyles="dashed")
        else:
            ax.vlines(boundaries[ele], 0, 100, color="grey", linestyles="dashed")

    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.set_xlabel("recognition accuracy [%]")
    ax.set_ylabel("face detection confidence [%]")
    ax.set_ylim((0, 100))
    ax.set_xlim((0, 100))
    plt.savefig(title + ".png", bbox_inches="tight")
    plt.savefig(title + ".pdf", bbox_inches="tight")


def make_utility_vs_privacy_plot_single_anon(data, title, paras, boundaries):
    fig, ax = plt.subplots()
    clmap = plt.get_cmap("viridis")
    sc = ax.scatter(data[0], data[1], c=paras, cmap=clmap)

    for ele in boundaries:
        if "utility" in ele:
            ax.hlines(boundaries[ele], 0, 1, color="grey", linestyles="dashed")
        else:
            ax.vlines(boundaries[ele], 0, 1, color="grey", linestyles="dashed")

    ax.set_title(title)
    ax.set_xlabel("Utility")
    ax.set_ylabel("Privacy")
    ax.set_ylim((0, 1))
    ax.set_xlim((0, 1))
    fig.colorbar(sc, ax=ax)
    plt.show()


def extract_results(exp_config, results, max_utility=1.2):
    all_anons_results = {}
    for anon in exp_config["anonymizations"]:
        tmp_exp_conf = copy.deepcopy(exp_config)
        tmp_exp_conf["anonymizations"] = [anon]
        run_config = generate_configs(tmp_exp_conf)
        run_config = [x["config"] for x in run_config]

        anon_results = {}
        for config in run_config:
            if str(config["anonymization"]) not in anon_results:
                anon_results[str(config["anonymization"])] = {}

            for ele in results:
                result = results[ele]
                if result["config"] == config:
                    if "privacy" in result["config"]:
                        anon_results[str(config["anonymization"])]["privacy"] = result["metrics"]["hitrate"] * 100
                    elif "utility" in result["config"]:
                        anon_results[str(config["anonymization"])]["utility"] = result["metrics"]["avg"] * 100

        all_anons_results[anon["name"]] = anon_results

    return all_anons_results


def reduce_results(all_anons_results):
    data = []
    anon_names = []
    for anon in all_anons_results:
        anon_names.append(anon)
        anon_results = all_anons_results[anon]
        tmp = [[], []]
        for ele in anon_results:
            if anon_results[ele] != {}:
                if "utility" not in anon_results[ele]:
                    print("Missing utility in", ele, anon_results[ele])
                    continue
                if "privacy" not in anon_results[ele]:
                    print("Missing privacy in", ele, anon_results[ele])
                    continue
                tmp[0].append(anon_results[ele]["privacy"])
                tmp[1].append(anon_results[ele]["utility"])
        data.append(tmp)

    return data, anon_names


if __name__ == "__main__":
    with open("anon-config.yaml", "r") as file:
        exp_config = yaml.load(file, Loader=yaml.SafeLoader)

    with open("deanon-config.yaml", "r") as file:
        exp_deanon_config = yaml.load(file, Loader=yaml.SafeLoader)

    with open("results.yaml", "r") as file:
        results = yaml.load(file, Loader=yaml.SafeLoader)

    all_anons_results = extract_results(exp_config, results)
    all_deanons_results = extract_results(exp_deanon_config, results)

    # Reuse the utility from the anon experiments for the deanon
    for anon in all_deanons_results:
        for paras in all_deanons_results[anon]:
            res = all_anons_results[anon][paras]
            if "utility" in res:
                all_deanons_results[anon][paras]["utility"] = res["utility"]

    # Reduce results to lists of scalar privacy and utility values
    anon_data, anon_names = reduce_results(all_anons_results)
    deanon_data, deanon_names = reduce_results(all_deanons_results)

    boundaries = {"privacy_chance": 1, "utility_best": 92.1}

    trans_dict = {"pick": "DeepPrivacy", "gaussianblur": "Gaussian Blur", "eyemask": "Eye Masking"}
    for i in range(len(anon_names)):
        anon_names[i] = trans_dict[anon_names[i]]

    plt.rcParams.update({"font.size": 12})

    make_utility_vs_privacy_plot_multiple_anon(anon_data, "Recognition vs. Face Detection", anon_names, boundaries)
    make_utility_vs_privacy_plot_multiple_anon(deanon_data, "De-anonymized Recognition vs. Face Detection", anon_names, boundaries)
