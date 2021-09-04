# Parallel Reverse Attention Network for Polyp Segmentation

[![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)](https://tensorflow.org)
[![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)](https://keras.io/)
[![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/)

[![GitHub license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/Thehunk1206/PRANet-Polyps-Segmentation?style=social)](https://github.com/Thehunk1206/PRANet-Polyps-Segmentation/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/Thehunk1206/PRANet-Polyps-Segmentation?style=social)](https://github.com/Thehunk1206/PRANet-Polyps-Segmentation/network/members)
[![GitHub watchers](https://img.shields.io/github/watchers/Thehunk1206/PRANet-Polyps-Segmentation?style=social)](https://github.com/Thehunk1206/PRANet-Polyps-Segmentation/watchers)


## Paper Abstract
```
Colonoscopy is an effective technique for detecting colorectal
polyps, which are highly related to colorectal cancer. In clinical practice,
segmenting polyps from colonoscopy images is of great importance since
it provides valuable information for diagnosis and surgery. However, 
accurate polyp segmentation is a challenging task, for two major reasons:
(i) the same type of polyps has a diversity of size, color and texture; and
(ii) the boundary between a polyp and its surrounding mucosa is not
sharp. To address these challenges, we propose a parallel reverse attention
network (PraNet)for accurate polyp segmentation in colonoscopy
images. Specifically, we first aggregate the features in high-level layers
using a parallel partial decoder (PPD). Based on the combined feature,
we then generate a global map as the initial guidance area for the following components. 
In addition, we mine the boundary cues using the reverse attention (RA) module,
which is able to establish the relationship between areas and boundary cues. 
Thanks to the recurrent cooperation mechanism between areas and boundaries,
our PraNet is capable of calibrating some misaligned predictions, improving the segmentation
accuracy. Quantitative and qualitative evaluations on five challenging
datasets across six metrics show that our PraNet improves the segmentation accuracy significantly,
and presents a number of advantages in terms of generalizability, and real-time segmentation efficiency
```
:scroll: Paper link: [Parallel Reverse Attention network for Polyps Segmentation](https://arxiv.org/pdf/2006.11392v4.pdf)

Checkout the original Implementation of the paper [here](https://github.com/DengPingFan/PraNet)

## Proposed Architecture in paper

The original paper is implemented in [Pytorch](https://pytorch.org/). Here I re-implemented same paper in [Tensorflow==2.6.0](https://www.tensorflow.org/api_docs/python/tf)

### PraNet(Parallel Reverse Attention Network) consist of three main component
* Feature Extractor Backbone (here [resnet50](https://arxiv.org/pdf/1512.03385.pdf) and [Mobilenetv2](https://arxiv.org/pdf/1801.04381.pdf))
* Parallel Partial Decoder Block (Main proposal of this paper)
* Reverse Attention Block ([Reverse Attention for Salient object detection](https://arxiv.org/pdf/1807.09940.pdf))

![Proposed Architecture](Illustrations/PRAnet%20architechture.png)
*Proposed Architecture in paper(Image taken from paper)*

### Feature Extractor Backbone
For extracting high level features from a pretrained network, the authors of the paper has used [Res2Net50](https://arxiv.org/pdf/1904.01169.pdf) pretrained on [Imagenet](https://www.image-net.org/). A feature extractor backbone can be any SOTA pretrained Image Recognition model. I used [resnet50](https://arxiv.org/pdf/1512.03385.pdf) for high precision output and [Mobilenetv2](https://arxiv.org/pdf/1801.04381.pdf) for low latency output pretrained on [Imagenet](https://www.image-net.org/).
The ultimate goal of using a pre-trained feature extractor backbone is to extract high level features from images (<i>i.e for resnet50 {conv3_block4_out f=512, conv4_block6_out f=1024, conv5_block3_out f=2048 }</i>) and aggregate to get a global saliency map.

![extracted_feature](Illustrations/features_at_conv2_block3_out_resnet50.png)
*Features at conv2_block3_out of resnet50 backbone*

### Parallel Partial Decoder
Most of the popular Biomedical Image Segmention model <i>like [Unet](https://arxiv.org/pdf/1505.04597.pdf) , [U-net++](https://arxiv.org/pdf/1807.10165.pdf), [ResUnet](https://arxiv.org/pdf/1904.00592.pdf) etc</i>. are full Encoder-Decoder model which aggregates all multi-level feature from backbone network. But According to [Wu et al.](https://arxiv.org/pdf/1904.08739v1.pdf) paper, <b>compared with high-level features of backbone, low-level features demand more computational resources due to their larger spatial resolutions, but contribute less to over all performance of the model.</b> Therefore, this paper introduces a PPD(Paraller Partial Decoder) block which aggregates only high-level features which is extracted from pretrained resnet50 model to get a *Global Saliency Map*. This block aggregates all high-level features with parallel connection this way the gradients can flow much faster and efficiently.

![PPD Block](Illustrations/Partial_decoder.svg)

*PPD block*

### Reverse Attention block

In a clinical setting, doctors first roughly locate the polyp region, and then carefully inspect local tissues to accurately label the polyp. Our Global Saliency map is derived from the deepest CNN layer, which can only capture a relatively rough location of the polyp tissues, without structural details (see proposed Architecture figure). To address this issue, This paper proposes a principle strategy to progressively mine discriminative polyp regions through an erasing foreground object manner as given in paper [Reverse Attention for Salient object detection](https://arxiv.org/pdf/1807.09940.pdf). Instead of aggregating features from all levels like in [Unet](https://arxiv.org/pdf/1505.04597.pdf) [U-net++](https://arxiv.org/pdf/1807.10165.pdf) and [ResUnet](https://arxiv.org/pdf/1904.00592.pdf), The paper proposes to adaptively learn the reverse attention in three parallel high-level
features. In other words, our architecture can sequentially mine complementary
regions and details by erasing the existing estimated polyp regions from highlevel side-output features, where the existing estimation is up-sampled from the deeper layer. <b>(Mostly taken from PraNet Paper, do refer [paper](https://arxiv.org/pdf/2006.11392v4.pdf) to get more info)</b>

![reverse Attention block](Illustrations/reverse_attention_block.png)

*Reverse Attention block (from paper [Reverse Attention for Salient Object Detection](https://arxiv.org/pdf/1807.09940.pdf))*

### Loss function
[PraNet](https://arxiv.org/pdf/2006.11392v4.pdf) paper uses Weighted BCE(Binary Cross Entropy loss) + Weighted IoU loss. Here I used  WBCE loss + WDice loss. Inspired by the loss function poroposed in paper [F3Net](https://vipl.ict.ac.cn/uploadfile/upload/2020071410571149.pdf). Do see the paper F3net to know more about this loss function.

![Weighted segmentation Map](Illustrations/BCE_IOU_Weights.jpeg)
*Weights for Dice and BCE loss*

## Training and Testing model

The PraNet Model was created with [Tensorflow==2.6.0](https://www.tensorflow.org/api_docs/python/tf) and was trained on [Google Colab's](https://colab.research.google.com/) Tesla T4 GPU (16GB VRAM)

### Dataset structure
Tensorflow [tf.data](https://www.tensorflow.org/guide/data) API is used to create Input Data pipeline.   
<b>In order to create a data pipeline for Image and Mask, the folder should be structured likewise:</b>
```
polyps_dataset
├── images [1000 files]
│   ├── 1.jpg
│   ├── 2.jpg
│   └── 3.jpg
└── masks [1000 files]
    ├── 1.jpg
    ├── 2.jpj
    └── 3.jpj

```

### Model settings
PraNet model was trained with two different backbone architecture
1. PraNet + Resnet50 (Bigger, Accurate but high latency model)
2. PraNet + MobilenetV2 (Smaller, Low latency but little less accurate model)

### Usage
* Clone my github repo [PraNet](https://github.com/Thehunk1206/PRANet-Polyps-Segmentation)
* Run ```$pip install -r requirements.txt``` to install required python packgages.


For Training model, run following
```
$python train_pranet.py --help
usage: train_pranet.py [-h] [--data_path DATA_PATH] [--data_split DATA_SPLIT] [--epoch EPOCH] [--lr LR] [--batchsize BATCHSIZE] [--inputsize INPUTSIZE]
                       [--backbone BACKBONE] [--gclip GCLIP] [--trained_model_path TRAINED_MODEL_PATH] [--logdir LOGDIR]

optional arguments:
  -h, --help            show this help message and exit
  --data_path DATA_PATH
                        path to dataset
  --data_split DATA_SPLIT
                        split percent of val and test data
  --epoch EPOCH         epoch number
  --lr LR               learning rate
  --batchsize BATCHSIZE
                        training batch size
  --inputsize INPUTSIZE
                        input image size
  --backbone BACKBONE   Feature Extractor backbone Arc
  --gclip GCLIP         gradient clipping margin
  --trained_model_path TRAINED_MODEL_PATH
  --logdir LOGDIR       Tensorboard logs
```
#### Example 
```
$python train_pranet.py --data_path polyps_dataset/ \
                        --data_split 0.1 \
                        --epoch 25 \
                        --lr 1e-4 \
                        --batchsize 16 \
                        --inputsize 352 \
                        --gclip 0.5 \
                        --trained_model_path trained_model/ \
                        --backbone resnet50
```

#### Testing Model with test set
```
$python test_pranet.py --data_path polyps_dataset/ \
                       --model_path trained_model/pranet_resnet/ \
                       --inputsize 352
```

#### Inferencing on single polyp image
```
$python polyps_seg_image.py --model_path trained_model/pranet_resnet \
                            --image_path sample_polyps/polyps1.jpg \
                            --inputsize 352
```
#### Inferencing on colonoscopy video:
```
$python polyps_seg_video.py --model_path trained_model/pranet_resnet/ \
                            --video_path sample_polyps/PolypColon.mp4 \
                            --inputsize 352 \
                            --threshold 0.5
```
<b>*Note: When loading PraNet model with Mobilenetv2 backbone arc, make sure the inputsize is set to 224*</b>
## Results
Model was trained on KVASIR SEG dataset which contains 1000 polyps images and 1000 corresponding mask. Data was splitted into Training, validating and Testing set in 80:10:10 ratio respectively.

*GPU: Google colab's Tesla T4 16GB, CPU: Google colab's compute engine runtime*

| PraNet + Backbone   | Learning Rate|    Epoch    | Dice (training)|   Dice (validating) | mDice(Testing)| mIoU(Testing) |mRuntime(inferencing mode) GPU|mRuntime(inferencing mode)CPU|
|---------------------|:-------------|:-----------:|---------------:|--------------------:|--------------:|--------------:|---------:|--------:|
| PraNet + resnet50   | 1e-4         | 25          | 0.97           |      0.88           |   0.87        |      0.79     |  30ms    |    510ms|
| PraNet + MobilenetV2| 1e-4         | 25          | 0.92           |      0.83           |   0.78        |      0.68     |  15ms    |     85ms|

## Metrics Result Comparison between Original Implementation(Paper) and My Implementation(Tensorflow)
*Currently the model is only trained on Kvasir-SEG dataset. If anyone is interested, they can train the model on different dataset and share their results.*

|                                       | Dataset      |  mDice   |  mIoU   | wFb   |S-measure|E-measure| MAE |
|---------------------------------------|-------------:|---------:|--------:|------:|--------:|--------:|----:|
| PraNet+Res2Net50 (Original)           | Kvasir-SEG   |   0.90   |  0.84   |  0.89 |   0.92  |  0.95   |0.030|
| PraNet+resnet50 (My Implementation)   | Kvasir-SEG   |   0.87   |  0.79   |  0.86 |   0.88  |  0.93   |0.037|
| PraNet+mobilenetv2 (My Implementation)| Kvasir-SEG   |   0.78   |  0.68   |  0.84 |   0.80  |  0.87   |0.056|

*Note: The results of My Implementation and original implementation differs because of training strategy, But overall the proposed architecture is SOTA and can converge way too fast by yielding better results.*

## Visual Results
### Single image segmentation (PraNet + resent50) 
*NOTE: The following polyps images are totally out of dataset*

![result1](results/detection_1629897892.6244857.jpg) 

![result2](results/detection_1629897991.632123.jpg)

![result3](results/detection_1629906315.877541.jpg)

### Single image segmentation (PraNet + mobilenetv2) 
*NOTE: The following polyps images are totally out of dataset*

![result4](results/detection_mobilenet_1629983960.1703162.jpg)

![result5](results/detection_mobilenetv2_1630207183.0212953.jpg)

![resul6](results/detection_mobilenet_1629984264.3444276.jpg)

### Real time polyp segmentation

![gif result](results/real_time_segmentation_PranNet_resnet50.gif)

## Trained model link
*Model was implemented in Tensorflow==2.6.0 and keras==2.6.0 by subclassing keras Model class.*
*Model was save in [SaveModel](https://www.tensorflow.org/guide/saved_model) format*

<b>*NOTE: Load the model with latest Tensorflow version*</b>

* PraNet + resnet50 [drive link](https://drive.google.com/drive/folders/1-SyjM1t2FZIrSOflBlXkNC1e_eyk3Vwk?usp=sharing)
* PraNet + Mobilenetv2 [drive link](https://drive.google.com/drive/folders/1-dXVoSffS8_zcpLuW7fnaP1NMy3b-WcW?usp=sharing)

## Dataset source
* [Kvasir SEG dataset](https://datasets.simula.no/kvasir-seg/)


# Reference Papers
*Some reference paper that was used in PraNet paper and was useful for this implementation*
* [Reverse Attention for Salient Object Detection](https://arxiv.org/pdf/1807.09940.pdf)
* [Receptive Field Block Net for Accurate and Fast Object Detection](https://arxiv.org/pdf/1711.07767.pdf)
* [A survey of loss functions for semantic segmentation](https://arxiv.org/pdf/2006.14822.pdf)
* [F3Net: Fusion, Feedback and Focus for Salient Object Detection](https://arxiv.org/pdf/1911.11445.pdf)
* [How to Evaluate Foreground Maps?](https://openaccess.thecvf.com/content_cvpr_2014/papers/Margolin_How_to_Evaluate_2014_CVPR_paper.pdf)
* [Structure-measure: A New Way to Evaluate Foreground Maps](https://arxiv.org/pdf/1708.00786.pdf)
* [Enhanced-alignment Measure for Binary Foreground Map Evaluation](https://arxiv.org/pdf/1805.10421.pdf)

# citation
Paper
```
@article{fan2020pra,
title={PraNet: Parallel Reverse Attention Network for Polyp Segmentation},
author={Fan, Deng-Ping and Ji, Ge-Peng and Zhou, Tao and Chen, Geng and Fu, Huazhu and Shen, Jianbing and Shao, Ling},
journal={MICCAI},
year={2020}
}
```

Datset
```
@inproceedings{jha2020kvasir,
title={Kvasir-seg: A segmented polyp dataset},
author={Jha, Debesh and Smedsrud, Pia H and Riegler, Michael A and Halvorsen, P{\aa}l and
de Lange, Thomas and Johansen, Dag and Johansen, H{\aa}vard D},
booktitle={International Conference on Multimedia Modeling},
pages={451--462},
year={2020},
organization={Springer}
}
```
