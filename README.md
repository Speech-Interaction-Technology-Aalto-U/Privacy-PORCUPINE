# Privay-PORCUPINE: Anonymization of Speaker Attributes Using Occurrence Normalization for Space-Filling Vector Quantization

This repository contains PyTorch implementation of our paper called Privacy-PORCUPINE, which is intended to provide more privacy protection for speech signals (which are quantized in an information bottleneck) using our recently proposed Space-Filling Vector Quantization technique together with a codebook resampling technique. The paper is published in Interspeech conference, September 2024.

![alt text](https://github.com/Speech-Interaction-Technology-Aalto-U/Privay-PORCUPINE/blob/main/vq_vs_sfvq.png?raw=true)

# **Contents of this repository**
- **train SFVQ directory**
  - `spacefilling_vq.py`: contains the main class of Space-Filling Vector Quantization
  - `train_sfvq.py`: an example showing how to use and optimize Space-Filline Vector Quantization to learn a Normal distribution
  - `utils.py`: contains some utility functions used in other codes
  - `plot_training_logs.py`: plots the training logs (which was saved druring execution of "train.py") in a pdf file

Due to some limitations of TensorBoard, we prefered our own custom logging function (`plot_training_logs.py`).

- **train VQ directory**
  - `plain_vq.py`: contains the main class of Vector Quantization
  - `train_vq.py`: an example showing how to use and optimize Space-Filline Vector Quantization to learn a Normal distribution

- **Metrics Evaluation directory**
  - `plot_evaluation_metrics.py`: code to compute and plot the evaluation metrics
  - `evaluation_utils.py`: contains some utility functions used in `plot_evaluation_metrics.py` code
  - `codebooks_vq`: contains trained VQ codebooks required for computing and plotting the evaluation metrics
  - `codebooks_sfvq`: contains trained SFVQ codebooks required for computing and plotting the evaluation metrics

# **Required packages**
- Python: (version 3.9)
- PyTorch (version: 1.13.1)
- Numpy (version: 1.26.3)
- Scipy (version: 1.11.4)
- Matplotlib (version: 3.2.2)

You can create the Python environment to run this project by passing the following lines of code in your terminal window in the following order:

`conda create --name privacy_porcupine python=3.9`  
`conda activate privacy_porcupine`  
`pip install torch==1.13.1`  
`pip install numpy==1.26.3`  
`pip install scipy==1.11.4`  
`pip install matplotlib==3.2.2`

You can also install all the above requirements by running the following command in your Python environment:  
`pip install -r requirements.txt`  

The requirements to use this repository is not that much strict, becuase the functions used in the code are so basic such that they also work with higher Python, PyTorch and Numpy versions.

# **Important note about training SFVQ**

In the "spacefilling_vq.py" code, there is a boolean variable "backpropagation" which should be set based on one of the following situations:

- **backpropagation=False**: If we intend to train the SpaceFillingVQ module exclusively (independent from any other module that requires training) on a distribution. In this case, we use the mean squared error (MSE) between the input vector and its quantized version as the loss function (exactly like what we did in the "train.py").

- **backpropagation=True**: If we intend to train the SpaceFillingVQ jointly with other modules that requires gradients for training, we pass the gradients through the SpaceFillingVQ module using our recently porposed [Noise Substitution in Vector Quantization (NSVQ)](https://ieeexplore.ieee.org/abstract/document/9696322) technique. In this case, you do not need to define or add an exclusive loss term for SpaceFillingVQ optimization. The optimization loss function must only include the global loss function you use for your specific application.

# **Abstract of the paper**

Speech signals contain a vast range of private information such as its text, speaker identity, emotions, and state of health. Privacy-preserving speech processing seeks to filter out any private information that is not needed for downstream tasks, for example with an information bottleneck, sufficiently tight that only the desired information can pass through. We however demonstrate that the occurrence frequency of codebook elements in bottlenecks using vector quantization have an uneven information rate, threatening privacy. We thus propose to use space-filling vector quantization (SFVQ) together with occurrence normalization, balancing the information rate and thus protecting privacy. Our experiments with speaker identification validate the proposed method. This approach thus provides a generic tool for quantizing information bottlenecks in any speech applications such that their privacy disclosure is predictable and quantifiable.

# **Cite the paper as**

Mohammad Hassan Vali and Tom Bäckström, “Privacy PORCUPINE: Anonymization of Speaker Attributes Using Occurrence Normalization for Space-Filling Vector Quantization,” in Proceedings of Interspeech 2024.

```bibtex
@inproceedings{vali2024porcupine,
  title={{P}rivacy {PORCUPINE}: {A}nonymization of {S}peaker {A}ttributes {U}sing {O}ccurrence {N}ormalization for {S}pace-{F}illing {V}ector {Q}uantization},
  author={Vali, Mohammad Hassan and Bäckström, Tom},
  booktitle={Proceedings of Interspeech},
  year={2024}
}
```
