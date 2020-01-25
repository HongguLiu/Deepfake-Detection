# Deepfake-Detection
------------------
The Pytorch implemention of Deepfake Detection based on [Faceforensics++](https://github.com/ondyari/FaceForensics)
## Install & Requirements
The code has been tested on pytorch=1.3.1 and python 3.6, please refer to `requirements.txt` for more details.
### To install the python packages
`python -m pip install -r requiremnets.txt`

## Usage
**To test with videos**

`python detect_from_video.py --video_path ./videos/003_000.mp4 --model_path ./pretrained_model/df_c0_best.pkl -o ./output --cuda`

**To test with images**

`python test_CNN.py -bz 32 --test_list ./data_list/Deepfakes_c0_299.txt --model_path ./pretrained_model`

**To train a model**

`python train_CNN.py`
(Please set the arguments after read the code)

## License
The provided implementation is strictly for academic purposes only. Should you be interested in using our technology for any commercial use, please feel free to contact us.