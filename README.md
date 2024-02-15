# Pedestrian detection and description system
System allows to detect,track and descript attributes of each people presented on video. Program performs also couting persistence time in two RoI's (Region of Interest). At the end is created results file. 
## People detecting and tracking
System performs detecting and tracking people and assigns ID to each unique person using [Yolov8](https://docs.ultralytics.com/) as detector and [BoT-SORT](https://github.com/NirAharon/BoT-SORT) as tracker
## Pederastian Attributes Recognizing
PAR is performed with our own models based on [ResNet](https://arxiv.org/abs/1512.03385) feature extraction and multi task learning with [Convolutional Block Attention Module](https://arxiv.org/pdf/1807.06521.pdf). System recognizes gender, presence of gender and hat and colors of upper and lower part of body.
## Features
System has implemented few interesting features as strategies of choosing the best global prediction according of prediction in each frame and strategy of avoiding disapperance for very short time 

# Installation and usage
## Installation
Clone project package
```bash
git clone https://github.com/jakadam2/AVProject.git
```
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install.

```bash
pip install -r requirements.txt
```
Download weights for PAR models
```bash
python download_weights.py
```
## Usage

```bash
python main.py --video your_video_name.mp4 --configuration roi_config_file.txt --results result_file.txt
```

## Authors

[Cuomo Ferdinando](https://github.com/FerdinandoCuomoUnisa), [D’Amora Domenico Pio](https://github.com/DamoraDomenicoPio), [Della Porta Assunta](https://github.com/xAspox), [Wozny Adam](https://github.com/jakadam2)


## License

[MIT](https://choosealicense.com/licenses/mit/)
