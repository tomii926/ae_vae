# Comparison between Autoencoder and Variational Autoencoder in anomaly detection

## Usage
### Training model
```
usage: train.py [-h] [--nepoch NEPOCH] [--nz NZ] [-g GPU_NUM] [--vae] [-i [INPUT_NUMS ...]]

Train model

optional arguments:
  -h, --help            show this help message and exit
  --nepoch NEPOCH       number of epochs to train for
  --nz NZ               size of the latent z vector
  -g GPU_NUM, --gpu-num GPU_NUM
                        what gpu to use
  --vae                 train VAE model
  -i [INPUT_NUMS ...], --input-nums [INPUT_NUMS ...]
                        Classes used for training model
```

### Reconstruction
```
usage: reconstruct.py [-h] [--nepoch NEPOCH] [--nz NZ] [--vae] [-i [INPUT_NUMS ...]] [-t [TEST_NUMS ...]] [--image-num IMAGE_NUM] [-f]

Reconstruct images from test dataset

optional arguments:
  -h, --help            show this help message and exit
  --nepoch NEPOCH       which epoch model to use for reconstruction
  --nz NZ               size of the latent z vector
  --vae                 use VAE model
  -i [INPUT_NUMS ...], --input-nums [INPUT_NUMS ...]
                        The model trained with these classes will be used.
  -t [TEST_NUMS ...], --test-nums [TEST_NUMS ...]
                        which classes to reconstruct
  --image-num IMAGE_NUM
                        how many images to reconstruct
  -f, --fashion         use Fashion-MNIST for reconstruction
```

### Anomaly detection
```
usage: anomaly_detection.py [-h] [--nepoch NEPOCH] [--nz NZ] [--vae] [--no-kl] [--kl] [-t THRESHOLD] [-g GPU_NUM] inputnums [inputnums ...]

Anomaly detection when trained with partial MNIST classes.

positional arguments:
  inputnums             The model trained by this classes will be used.

optional arguments:
  -h, --help            show this help message and exit
  --nepoch NEPOCH       which epoch model to use for anomaly detection
  --nz NZ               size of the latent z vector
  --vae                 choose vae model
  --no-kl               KL divergence is not used in determining the threshold.
  --kl                  Only KL divergence is used when determining threshold.
  -t THRESHOLD, --threshold THRESHOLD
                        threshold
  -g GPU_NUM, --gpu-num GPU_NUM
                        what gpu to use
```

### Anomaly detection with threshold determined by non-IID data
```
usage: positive_rates_heat_map.py [-h] [--nepoch NEPOCH] [--nz NZ] [--vae] [--kl] [--no-kl] [-t THRESHOLD] [-g GPU_NUM]

Create a heatmap of positive rate

optional arguments:
  -h, --help            show this help message and exit
  --nepoch NEPOCH       which epoch model to use
  --nz NZ               size of the latent z vector
  --vae                 use vae model
  --kl                  Only KL divergence is used when determining threshold
  --no-kl               KL divergence is not used in determining the threshold
  -t THRESHOLD, --threshold THRESHOLD
                        threshold
  -g GPU_NUM, --gpu-num GPU_NUM
                        what gpu to use
```

### Appendix: Image Generation by VAE
```
usage: generate_image.py [-h] [--nepoch NEPOCH] [--nz NZ]

Generate images from random latent vectors using the learned model.

optional arguments:
  -h, --help       show this help message and exit
  --nepoch NEPOCH  number of epochs to generate images
  --nz NZ          size of the latent z vector
```
