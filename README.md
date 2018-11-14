## CornerNet

reproduce of [Cornernet](https://arxiv.org/pdf/1808.01244v1.pdf)

developing...,  almost finished.

## requirements

* To compile corner pooling layer, yo need toinstall mxnet 1.3.0, then put the files in cxx_operator into src/operator/nn/ in mxnet source code and compile it, then run

```
export PYTHONPATH=${YOUR_MXNET_ROOT}/lib/libimxnet.so:${PYTHONPATH}
```

to make sure you import the correct mxnet library.
Alternatively, you can uncomment line 92 and 93 and comment line 94, 95 in symbols/cornernet.py  to use python implementation of cornerpooling layer, which would be much slower.

* run init.sh to compile nms and pycocotools

## TRAIN
you can change the batch_size in config/cfg.py according to your gpus' number and their computation abilies, but make sure that batch_size number is proportional to the number of gpus.

```
python train.py --gpus 0,1
```

## TEST
train model will be uploaded soon.

```
python test.py --prefix models/cornernet --epoch 100 --gpus 0
```

if you want to visualize the test results:

```
python test.py --prefix models/cornernet --epoch 100 --gpus 0 --debug True
```

images will be saved in images/
