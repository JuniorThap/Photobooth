# Photobooth

## Set up
### Create env
``` bash
conda create --name photobooth python=3.11
```
``` bash
conda activate photobooth
```
### Install requirements
``` bash
pip install requirements.txt
```
Install pytorch cuda seperately
``` bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

## How to use
``` bash
python main.py
```