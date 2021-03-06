## CornerNet

Reproduce of [Cornernet](https://arxiv.org/pdf/1808.01244v1.pdf)

The original pytorch implementation repository is [here](https://github.com/princeton-vl/CornerNet)

## Requirements

* You will need python modules: cv2, matplotlib and numpy.

* To compile corner pooling layer, yo need to install mxnet 1.3.0, then put the files in cxx_operator into src/operator/nn/ in mxnet source code and compile it, then run

```
cd ${YOUR_MXNET_ROOT}
export PYTHONPATH=$(pwd)/lib/libmxnet.so:${PYTHONPATH}
```

to make sure you import the correct mxnet library.

Alternatively, you can uncomment line 92 and 93 and comment line 94, 95 in symbols/cornernet.py  to use python implementation of cornerpooling layer, which would be much slower.

* run init.sh to compile nms and pycocotools

## Demo results

![demo1](https://github.com/BigDeviltjj/mxnet-cornernet/blob/master/images/image_0000.jpg)

![demo2](https://github.com/BigDeviltjj/mxnet-cornernet/blob/master/images/image_0084.jpg)

## mAP
|        Model          | Training data    | Test data |  mAP |
|:-----------------:|:----------------:|:---------:|:----:|
| [CornerNet_coco_511x511](https://drive.google.com/drive/folders/1kPZaK4bRwzVuyij_uC0_niupw-VlLmcV) | train2014+valminusminival2014| minival2014| 38.9|

## TRAIN

You need to put the coco image files in date.

You can change the batch_size in config/cfg.py according to your gpu number and their computation abilies, but make sure that batch_size number is proportional to the number of gpus.

```
python train.py --gpus 0,1
```

## TEST

Download the compressed model from [CornerNet_coco_511x511](https://drive.google.com/drive/folders/1kPZaK4bRwzVuyij_uC0_niupw-VlLmcV) and unzip it then put it in model/, then run

```
python test.py --prefix model/cornernet --epoch 100 --gpus 0
```

if you want to visualize the test results:

```
python test.py --prefix model/cornernet --epoch 100 --gpus 0 --debug True
```

images will be saved in images/
