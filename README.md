

# card_scanner_api

[![Run on Google Cloud](https://deploy.cloud.run/button.svg)](https://deploy.cloud.run)

## How to run API:

## 1. download repo
git clone https://github.com/Antonyo314/card_scanner_api.git

## 2. download assets
https://drive.google.com/file/d/1Z90uYwPzbEjYEuE8qK3KrUe0ujW9VELH/view?usp=sharing <br/>
and move files to card_scanner_api/assets

## 3. install requirements

pip3 install -r requirements.txt

## 4. run API

gunicorn --bind 0.0.0.0:5000 wsgi:app

## 5. Make request

python3 request.py

## How does it work?

* I scrapped all sport cards <br/>
* After that I created embedding vector for each card with img2vec_pytorch (https://pypi.org/project/img2vec-pytorch/)<br/>
* I trained detector to detect single card on the talbe (YOLO v5 detector trained on generated dataset) 

## Working pipeline works this way: 

* Detector detect card <br/>
* Card image is cropped (based on the detector's prediction) and a feature vector is calculated <br/>
* The most similar card is searched (in the embedding vectors database) by euclidean metric
