# Domain Adaptation with Adversarial Training on Penultimate Activations
Pytorch implementation of KUDA. 
> [Domain Adaptation with Adversarial Training on Penultimate Activations](https://arxiv.org/abs/2208.12853)                 
> Tao Sun, Cheng Lu, and Haibin Ling                 
> *AAAI 2023 Oral* 

Enhancing model prediction confidence on unlabeled target data is an important objective in Unsupervised Domain Adaptation (UDA). In this paper, we explore adversarial training on penultimate activations, ie, input features of the final linear classification layer. We show that this strategy is more efficient and better correlated with the objective of boosting prediction confidence than adversarial training on input images or intermediate features, as used in previous works. Furthermore, with activation normalization  commonly used in domain adaptation to reduce domain gap, we derive two variants and systematically analyze the effects of normalization on our adversarial training. This is illustrated both in theory and through empirical analysis on real adaptation tasks. Extensive experiments are conducted on popular UDA benchmarks under both standard setting and source-data free setting. The results validate that our method achieves the best scores against previous arts. 

## Usage
### Prerequisites
We experimented with python==3.8, pytorch==1.8.0, cudatoolkit==11.1. 

### Training
To reproduce results on office home,

```shell
# train on source data
python main_Base.oh.py

# adapt to target data
# UDA
python main_APAu.oh.py
python main_APAn.oh.py

# Source-Free DA
python main_APAu.oh.SF.py
python main_APAn.oh.SF.py
```

## Citation
If you find our paper and code useful for your research, please consider citing
```bibtex
@article{sun2022domain,
    author    = {Sun, Tao and Lu, Cheng and Ling, Haibin},
    title     = {Domain Adaptation with Adversarial Training on Penultimate Activations},
    journal   = {AAAI Conference on Artificial Intelligence },
    year      = {2023}
}
```