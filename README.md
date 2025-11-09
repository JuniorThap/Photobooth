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
python server.py
```
``` bash
ngrok http 5000
```
Copy Forwarding link from ngrok to base_url in qrcode
``` bsah
python main.py
```