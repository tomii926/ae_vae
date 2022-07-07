import os
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import seaborn as sns

from anomaly_detection import positive_rates
from common import device, mkdir_if_not_exists

parser = ArgumentParser(description="Create a heatmap of positive rate")
parser.add_argument('--nepoch', type=int, help="which epoch model to use", default=50)
parser.add_argument('--nz', type=int, help='size of the latent z vector', default=16)
parser.add_argument('--vae', action="store_true", help="use vae model")
parser.add_argument('-t', '--threshold', type=float, help="threshold", default=0.99)
parser.add_argument('-g', '--gpu-num', type=int, help='what gpu to use', default=0)
parser.add_argument('--aug', action="store_true", help="model which used data augmentation")
args = parser.parse_args()

device = device(args.gpu_num)

positive_rate_list = [positive_rates([], [i], args.threshold, args.nepoch, args.vae, args.nz, args.aug, device) for i in range(10)]

plt.figure(figsize = (10,8))
sns.heatmap(positive_rate_list, annot=True, cmap='Blues_r', xticklabels=[str(i) for i in range(10)] + ['Fasion'])
path = os.path.join(mkdir_if_not_exists(f'graph/{"v" if args.vae else ""}ae'), f"{'aug_' if args.aug else ''}t{args.threshold:.3f}_e{args.nepoch:04d}.png")
plt.title(f"Positive rates (threshold: {args.threshold * 100:.1f}%)")
plt.ylabel('The class used for determining threshold')
plt.xlabel('class')
plt.savefig(path, bbox_inches='tight')
print(f'Image saved {path}')
