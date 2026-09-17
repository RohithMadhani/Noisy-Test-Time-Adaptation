# Noisy Test-Time Adaptation

Experiments for **zero-shot noisy test-time adaptation (ZS-NTTA)** with
zero-shot CLIP. This repository contains Jupyter/Google Colab notebooks that
prepare and run experiments on top of the upstream
[ZS-NTTA](https://github.com/tmlr-group/ZS-NTTA) implementation.

The experiments focus on adapting a zero-shot classifier at inference time
when the incoming stream contains both in-distribution (ID) images and
out-of-distribution (OOD) images. The notebooks evaluate CIFAR-10 and
CIFAR-100 as ID datasets, SVHN/LSUN/Places as OOD datasets, and several
detector, label-smoothing, deferral, ensemble, and thresholding variants.

## Repository contents

| Notebook | Purpose |
| --- | --- |
| [`AdaND.ipynb`](./AdaND.ipynb) | Full AdaND design-space sweep over the supported ID/OOD pairs and configuration variants. |
| [`GMM_vs_Otsu.ipynb`](./GMM_vs_Otsu.ipynb) | Compares Otsu and Gaussian-mixture-model (GMM) threshold selection across six ID/OOD pairs. |
| [`Cifar100.ipynb`](./Cifar100.ipynb) | Runs the method variants on CIFAR-100 + Places with paper-matching batch-size-one settings. |

The notebooks generate experiment configuration files, result directories,
and patched files inside the cloned upstream ZS-NTTA checkout. Those generated
files are not part of this repository.

## Requirements

- Python 3.8 or newer
- Jupyter Notebook/Lab or Google Colab
- A CUDA-capable GPU is strongly recommended
- Sufficient disk space for model weights and the CIFAR/OOD datasets

The notebooks install the main Python dependencies automatically:

```text
ml-collections  absl-py  ftfy  wandb  seaborn
scikit-learn    regex    scipy  pytorch-ood
```

PyTorch and torchvision should be installed for the CUDA version available on
the machine. For a local environment, install them from the
[official PyTorch selector](https://pytorch.org/get-started/locally/) rather
than copying a CUDA-specific command blindly.

## Quick start

1. Clone this repository and open it in Jupyter or upload the notebooks to
   Google Colab:

   ```bash
   git clone https://github.com/RohithMadhani/Noisy-Test-Time-Adaptation.git
   cd Noisy-Test-Time-Adaptation
   ```

2. Open one of the notebooks. The default notebook workflow uses
   `/content/ZS-NTTA` for the upstream checkout and `/content/datasets` for
   downloaded data, which is convenient in Colab.

3. Run the setup cells first. They:
   - clone the upstream ZS-NTTA repository;
   - install the required packages;
   - replace the upstream hard-coded dataset path;
   - create result and cache directories;
   - write the experiment-specific detector, adaptation, and configuration
     code.

4. Run the dataset setup cells. SVHN is downloaded automatically by the
   underlying data pipeline. LSUN and Places are prepared through
   `pytorch-ood`; downloading these datasets may require substantial time and
   disk space.

5. Run the zero-shot CLIP baseline before the adaptation experiments for each
   ID/OOD pair, then run the selected experiment cells and inspect the printed
   result tables.

The notebooks invoke the upstream runner in the following form:

```bash
python main.py --config=configs/<experiment_config>.py \
  --test_set=<CIFAR-10-or-CIFAR-100> \
  --OOD_set=<SVHN-or-LSUN-or-Places> \
  --gpu=0
```

## Experiments

The standard sweep covers these six pairs:

```text
CIFAR-10   + SVHN
CIFAR-10   + LSUN
CIFAR-10   + Places
CIFAR-100  + SVHN
CIFAR-100  + LSUN
CIFAR-100  + Places
```

`AdaND.ipynb` additionally evaluates detector and adaptation options such as
enhanced OOD features, soft labels, adaptive/asymmetric weighting, smooth
transitions, class-weight changes, ensemble weighting, deferral margins, and
three-way classification margins.

`GMM_vs_Otsu.ipynb` writes two configurations:

- `thr_otsu.py`: chooses the OOD threshold with Otsu's method;
- `thr_gmm.py`: fits a two-component Gaussian mixture and chooses the
  corresponding threshold.

The notebook includes a sanity check that verifies the GMM configuration is
loaded and that GMM threshold computations are actually executed.

## Results and reproducibility

Results are written by the upstream code under paths such as:

```text
results/main_results/<ID dataset>/
data_analysis/score/
data_analysis/conf_pkl/
```

Experiments can take from several minutes to multiple hours depending on the
dataset, configuration, batch size, storage speed, and GPU. The CIFAR-100 +
Places notebook intentionally uses batch size 1 to match the paper settings
and can take significantly longer.

For a clean rerun, use a fresh upstream checkout or remove the generated
results/configuration files before rerunning setup cells. The notebooks back
up selected upstream files before patching them and include a final cell in
`AdaND.ipynb` for restoring those files.

## Notes

- This is an experiment repository, not an installable Python package.
- The notebooks are written primarily for Google Colab and contain Colab
  shell commands such as `!pip` and `!git`.
- Dataset availability, download URLs, upstream code, and GPU memory
  requirements may change over time.
- Do not commit downloaded datasets, model checkpoints, experiment outputs, or
  credentials such as Weights & Biases API keys.

## Acknowledgements

The experiment notebooks build on the upstream
[ZS-NTTA repository](https://github.com/tmlr-group/ZS-NTTA). Please consult
that project for the original method, implementation details, and citation
information.
