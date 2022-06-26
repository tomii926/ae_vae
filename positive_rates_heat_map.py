import os
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import seaborn as sns

from anomaly_detection import positive_rates
from common import device, mkdir_if_not_exists

parser = ArgumentParser()
parser.add_argument('--nepoch', type=int, help="which epochs to generate image", default=50)
parser.add_argument('--nz', type=int, help='size of the latent z vector', default=16)
parser.add_argument('--vae', action="store_true", help="choose vae model")
parser.add_argument('--kl', action="store_true", help="use only KL divergence when determining threshold.")
parser.add_argument('--no-kl', action="store_true", help="don't use KL divergence when determining threshold.")
parser.add_argument('-t', '--threshold', type=float, help="threshold", default=0.99)
parser.add_argument('-g', '--gpu-num', type=int, help='what gpu to use', default=0)
args = parser.parse_args()

device = device(args.gpu_num)

positive_rate_list = [positive_rates([], [i], args.threshold, args.nepoch, args.vae, args.nz, device) for i in range(10)]

plt.figure(figsize = (10,8))
sns.heatmap(positive_rate_list, annot=True, cmap='Blues_r', xticklabels=[str(i) for i in range(10)] + ['Fasion'])
path = os.path.join(mkdir_if_not_exists(f'graph/{"v" if args.vae else ""}ae'), f"{'onlykl' if args.kl else 'nokl' if args.no_kl else ''}_t{args.threshold:.3f}.png")
plt.title(f"Positive rates of each class when the classes used for determining threshold are non-iid")
plt.ylabel('which class is used in determining threshold value')
plt.xlabel('class')
plt.savefig(path, bbox_inches='tight')
print(f'The graph was output to {path}')






    





