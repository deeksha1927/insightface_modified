import argparse
import os
import numpy as np
from os.path import join
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import warnings

warnings.filterwarnings("ignore")
def get_feat_dict(path):
    im_list = []
    with open(path + '.txt', 'r') as f:
        for i in f:
            im_list.append(i.rstrip())
    feats = np.load(path + '.npy')
    return im_list, feats

def main(model_name, dataset, home, save_dir):
    features_path = f'{home}/features_ear/{model_name}/{dataset}/{dataset}'
    names, feats = get_feat_dict(features_path)
    assert len(names) == len(feats), f'{len(names)} {len(feats)}'
    length = len(names)

    # Extract IDs based on directory structure
    ids = np.array([name.split('/')[-2] for name in names])

    # Generate genuine/imposter labels
    gen_or_imp = np.zeros((length, length), dtype=np.uint8)
    for i in range(length):
        ind = np.zeros(length)
        ind[:i+1] = -1
        ind[i+1:] = ids[i] == ids[i+1:]
        gen_or_imp[i] = ind
    gen_or_imp = gen_or_imp.flatten()

    # Calculate scores
    scores = np.dot(feats, feats.transpose()).flatten()
    gen_ind = np.where(gen_or_imp == 1)
    imp_ind = np.where(gen_or_imp == 0)

    gen_scores = scores[gen_ind]
    imp_scores = scores[imp_ind]

    # Plot histograms
    plt.rcParams["figure.figsize"] = [6, 4.5]
    plt.rcParams["font.size"] = 12
    plt.figure()
    plt.hist(imp_scores, bins='auto', color='b', density=True, histtype="step", label="imp")
    plt.hist(gen_scores, bins='auto', color='r', density=True, histtype="step", label="gen")
    plt.xlim(-.4, 1)
    plt.ylim(0, 8)
    plt.ylabel("Relative Frequency")
    plt.xlabel("Match Scores")

    title = f'data: {dataset}'
    legend1 = plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode='expand', ncol=3, borderaxespad=0,
                         fontsize=10, edgecolor="black", handletextpad=0.3, title=title)
    plt.tight_layout(pad=0.2)

    d_prime1 = (abs(np.mean(gen_scores) - np.mean(imp_scores)) /
                np.sqrt(0.5 * (np.var(gen_scores) + np.var(imp_scores))))
    print(d_prime1)
    label = "d-prime: {:.3f}".format(np.round(d_prime1, 3))
    plt.legend([Rectangle((0, 0), 1, 1, color='b', fill=True)], [label], loc='best', fontsize=10)
    plt.gca().add_artist(legend1)

    # Save the plot and d-prime value
    save_path = f'{home}/{save_dir}/{model_name}/{dataset}/{dataset}-dprime.png'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    with open(join(os.path.dirname(save_path), f'dprime_{dataset}.txt'), 'w') as f:
        f.write('{:.3f}'.format(round(d_prime1, 3)))
    print('Saved plot and d-prime value.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot and Calculate D-Prime")
    parser.add_argument('--model-name', type=str, required=True, help='Model name')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--home', type=str, default='/store01/flynn/darun', help='Home directory path')
    parser.add_argument('--save-dir', type=str, default='plots_ear', help='Directory to save the plots and results')

    args = parser.parse_args()
    main(args.model_name, args.dataset, args.home, args.save_dir)

