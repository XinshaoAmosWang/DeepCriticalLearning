
## Deep Critical Learning (i.e., Deep Robustness) In The Era of Big Data

Here are related papers on the fitting and generalization of deep learning:
* [ProSelfLC: Progressive Self Label Correction Towards A Low-Temperature Entropy State](https://arxiv.org/abs/2207.00118)
* [Understanding deep learning requires rethinking generalization](https://arxiv.org/abs/1611.03530)
* [A Closer Look at Memorization in Deep Networks](https://arxiv.org/abs/1706.05394)
* [ProSelfLC: Progressive Self Label Correction for Training Robust Deep Neural Networks](https://arxiv.org/abs/2005.03788)
  * Blog link: [https://xinshaoamoswang.github.io/blogs/2020-06-07-Progressive-self-label-correction/](https://xinshaoamoswang.github.io/blogs/2020-06-07-Progressive-self-label-correction/)
* [Mean Absolute Error Does Not Treat Examples Equally and Gradient Magnitude’s Variance Matters](https://github.com/XinshaoAmosWang/Improving-Mean-Absolute-Error-against-CCE)
* [Derivative Manipulation: Example Weighting via Emphasis Density Funtion in the context of DL](https://github.com/XinshaoAmosWang/DerivativeManipulation)
  * Novelty: moving from loss design to derivative design


<details><summary>See Citation Details</summary>

#### Please kindly cite the following papers if you find this repo useful.
```
@article{wang2022proselflc,
  title={ProSelfLC: Progressive Self Label Correction Towards A Low-Temperature Entropy State},
  author={Wang, Xinshao and Hua, Yang and Kodirov, Elyor and Mukherjee, Sankha Subhra and Clifton, David A and Robertson, Neil M},
  journal={bioRxiv},
  year={2022}
}
@inproceddings{wang2021proselflc,
  title={ {ProSelfLC}: Progressive Self Label Correction
  for Training Robust Deep Neural Networks},
  author={Wang, Xinshao and Hua, Yang and Kodirov, Elyor and Clifton, David A and Robertson, Neil M},
  booktitle={CVPR},
  year={2021}
}
@phdthesis{wang2020example,
  title={Example weighting for deep representation learning},
  author={Wang, Xinshao},
  year={2020},
  school={Queen's University Belfast}
}
@article{wang2019derivative,
  title={Derivative Manipulation for General Example Weighting},
  author={Wang, Xinshao and Kodirov, Elyor and Hua, Yang and Robertson, Neil},
  journal={arXiv preprint arXiv:1905.11233},
  year={2019}
}
@article{wang2019imae,
  title={{IMAE} for Noise-Robust Learning: Mean Absolute Error Does Not Treat Examples Equally and Gradient Magnitude’s Variance Matters},
  author={Wang, Xinshao and Hua, Yang and Kodirov, Elyor and Robertson, Neil M},
  journal={arXiv preprint arXiv:1903.12141},
  year={2019}
}
```
</details>

## PyTorch Implementation for ProSelfLC, Derivative Manipulation, Improved MAE
* Easy to install
* Easy to use
* Easy to extend: new losses, new networks, new datasets and batch loaders
* Easy to run experiments and sink results
* Easy to put sinked results into your technical reports and academic papers.

## Demos
* [Training the shufflenetv2 on cifar-100 with a symmetric noise rate of 40%](./demos_jupyter_notebooks/convnets_cifar100/trainer_cifar100_covnets_proselflc.ipynb)

## Install

<details><summary>See Install Guidelines</summary>

#### Set the Pipenv From Scratch
* sudo apt update && sudo apt upgrade
* sudo apt install python3.8
* curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
* python3.8 get-pip.py
* vim ~/.bashrc -> add `export PATH="/home/ubuntu/.local/bin:$PATH"` -> source ~/.bashrc
* pip3 install pipenv

#### Build env for this repo using pipenv
* git clone `this repo`
* cd `this repo`
* pipenv install -e . --skip-lock

</details>

## How to use
#### Run experiments
* cd `this repo`
* pipenv shell
* pre-commit install
* run experimenets:
  * CIFAR-100: `CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
  TOKENIZERS_PARALLELISM=false
  python -W ignore
  tests/convnets_cifar100/trainer_calibration_vision_cifar100_covnets_proselflc.py`
  * Protein classification: `CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
  TOKENIZERS_PARALLELISM=false
  python -W ignore
  tests/protbertbfd_deeploc/MS-with-unknown/test_trainer_2MSwithunknown_proselflc.py`
#### Visualize results
The results are well sinked and organised, e.g.,
* CIFAR-100: [experiments_records/cifar100_symmetric_noise_rate_0.4/shufflenetv2](experiments_records/cifar100_symmetric_noise_rate_0.4/shufflenetv2)
* Protein classification: [experiments_records/deeploc_prottrans_symmetric_noise_rate_0.0/Rostlab_prot_bert_bfd_seq/MS-with-unknown](experiments_records/deeploc_prottrans_symmetric_noise_rate_0.0/Rostlab_prot_bert_bfd_seq/MS-with-unknown)


## Examples of sinked experimental configs and resutls
<details><summary>See Sinked Results</summary>

* The results are well sinked and organised, e.g.,
[experiments_records/cifar100_symmetric_noise_rate_0.4/shufflenetv2/502_proselflc_warm0_20220606-150113](experiments_records/cifar100_symmetric_noise_rate_0.4/shufflenetv2/502_proselflc_warm0_20220606-150113)

* [Accuracy curve: shufflenetv2](experiments_records/cifar100_symmetric_noise_rate_0.4/shufflenetv2/502_proselflc_warm0_20220606-150113/accuracy.pdf)


* [Loss curve: shufflenetv2](experiments_records/cifar100_symmetric_noise_rate_0.4/shufflenetv2/502_proselflc_warm0_20220606-150113/loss.pdf)

* [accuracy_loss_normalised_entropy_max_p_metadata.xlsx](experiments_records/cifar100_symmetric_noise_rate_0.4/shufflenetv2/502_proselflc_warm0_20220606-150113/accuracy_loss_normalised_entropy_max_p_metadata.xlsx)

* [params.csv](experiments_records/cifar100_symmetric_noise_rate_0.4/shufflenetv2/502_proselflc_warm0_20220606-150113/params.csv)

</details>



## How to extend this repo
* Add dataset and dataloader: see examples in [src/proselflc/slices/datain](src/proselflc/slices/datain)
* Add losses: see examples in [src/proselflc/slices/losses](src/proselflc/slices/losses)
* Add networks: see examples in [src/proselflc/slices/networks](src/proselflc/slices/networks)
* Add optimisers: see examples in [src/proselflc/optim](src/proselflc/optim)
* Extend the slicegetter: [src/proselflc/slicegetter](src/proselflc/slicegetter)
* Write run scripts: see examples in [tests/](tests/)


## Supplementary material
#### Talks
  * [12th Aug 2022, Southern University of Science and Technology](./Poster_Slide/Talks/2022-08-12-XW-SUSTECH.pdf)
  * [17th May 2022, Loughborough University](./Poster_Slide/Talks/2022-05-17-XW-Loughborough.pdf)
#### [Link to CVPR 2021 Slide, Poster](./Poster_Slide/CVPR-2021)
#### [Link to reviewers' comments](./Reviews)

## Notes and disclaimer
* For any specific research discussion or potential future collaboration, please feel free to contact me. <br />
* This work is a personal research project and in progress using personal time.

## [LICENSE](LICENSE)
