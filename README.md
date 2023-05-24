# Deepfake-Detection
------------------
The Pytorch implemention of Deepfake Detection based on [Faceforensics++](https://github.com/ondyari/FaceForensics)

The Backbone net is XceptionNet, and we also reproduced the [MesoNet with pytorch version](https://github.com/HongguLiu/MesoNet-Pytorch), and you can use the mesonet network in this project.

## Install & Requirements
The code has been tested on pytorch=1.3.1 and python 3.6, please refer to `requirements.txt` for more details.
### To install the python packages
`python -m pip install -r requirements.txt`

Although you can install all dependencies at a time. But it is easy to install dlib via `conda install -c conda-forge dlib`


## Dataset
If you want to use the opensource dataset [Faceforensics++](https://github.com/ondyari/FaceForensics), you can use the script './download-FaceForensics_v3.py' to download the dataset accroding the instructions of [download section](https://github.com/ondyari/FaceForensics/blob/master/dataset/README.md).

You can train the model with full images, but we suggest you take only face region as input.

## Pretrained Model
The model provided just be used to test the effectiveness of our code. We suggest you train you own models based on your dataset. 

And we will upload models which have better performance as soon as possible.

we provide some [pretrained model](https://drive.google.com/drive/folders/1GNtk3hLq6sUGZCGx8fFttvyNYH8nrQS8?usp=sharing) based on FaceForensics++
- FF++\_c23.pth
- FF++\_c40.pth

## Usage
**To test with videos**

`python detect_from_video.py --video_path ./videos/003_000.mp4 --model_path ./pretrained_model/df_c0_best.pkl -o ./output --cuda`

**To test with images**

`python test_CNN.py -bz 32 --test_list ./data_list/Deepfakes_c0_299.txt --model_path ./pretrained_model/df_c0_best.pkl`

**To train a model**

`python train_CNN.py`
(Please set the arguments after read the code)

## About
If our project is helpful to you, we hope you can star and fork it. If there are any questions and suggestions, please feel free to contact us.

Thanks for your support.
## License
The provided implementation is strictly for academic purposes only. Should you be interested in using our technology for any commercial use, please feel free to contact us.
