# Unofficial Implementation of FewShot3DKP (CVPR 2023)

This is an unofficial implementation of [FewShot3DKP](https://arxiv.org/abs/2303.17216).

Since thes code is not released, I tried to build an unofficial one.

I copied most of the parts from [Autolink](https://github.com/xingzhehe/AutoLink-Self-supervised-Learning-of-Human-Skeletons-and-Object-Outlines-by-Linking-Keypoints), the authors' another project, as their architecture looks super similar.

## Setup

##### Setup environment

```
conda create -n fewshot3dkp python=3.8
conda activate fewshot3dkp
pip install -r requirements.txt
```

##### Download datasets

The [WFLW](https://wywu.github.io/projects/LAB/WFLW.html), can be found on their websites.

I provide the pre-processing code for WFLW make them `h5` files. The code is based on [3FabRec](https://github.com/browatbn2/3FabRec).

I also provide the processed h5 file in Github Release.

##### Download pre-trained models

The pre-trained models can be downloaded in Github Release.

## Things to notice:

1. In the 3D loss, the similarity transformation needs to be detached. Otherwise the model will break. (confirmed by the authors)
  
2. The edge map needs to be multiplied by a small number before being concatenated with the masked image. Otherwise the model may generate weird edges. (confirmed by the authors)
  
3. In the 2D loss, if the keypoints are outside the image after affine transformation, they should be ignored. (confirmed by the authors)

4. The depth needs to be devided by a large number so that it is not too crazy in the beginning. (confirmed by the authors)
  
5. The few-shot examples can significantly affects the model. Better choose different poses or shapes.
  

## Things different from the original paper (TODOs):

1. I implemented the 3D loss on the whole object instead of parts.
  
2. I do not minimize the difference between minimal pairs but random pairs.
  
3. I am not sure how the detector and decoder looks exactly, so I just copied from Autolink, which might be too small.
  
4. I do not use ViT perceptual loss but VGG perceptual loss as in Autolink.
  
5. I do not use linearly increased augmentation range, but fix them.
  
6. Instead of using KMeans to choose few-shot examples, I manually picked them.
  
7. I use fixed edge thickness instead of learnable ones.
  

## Performance

NME on WFLW (10 shots): 11.7 vs 9.19 (original paper)
