import json

import torch

from utils.datasets import LoadImages
from utils.general import check_img_size, find_max_confidence_bbox, scale_coords
from utils.similar_images_finder import SimilarImagesFinder


class Model:
    def __init__(self, weights='detector_weights/weights.pt', img_size=640, conf_thres=0.25,
                 iou_thres=0.45):
        self.weights = weights
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.similar_images_finder = SimilarImagesFinder(
            'assets/all_sports_embeddings.npy',
            'assets/all_sports_embeddings_matrix.npy'
        )
        self.model = self.load_model(weights)
        self.metadata = json.load(open(f'assets/all_sports_metadata.json'))

    @staticmethod
    def load_model(weights):
        model = torch.load(weights)['model'].float().eval()
        return model

    def detect_card_and_find_in_db(self, img):
        device = torch.device('cpu')
        stride = int(self.model.stride.max())  # model stride
        imgsz = check_img_size(self.img_size, s=stride)  # check img_size
        dataset = LoadImages()

        # Run inference
        if device.type != 'cpu':
            self.model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(self.model.parameters())))  # run once

        img, im0 = dataset.get_item(img)
        img = torch.from_numpy(img).to(device)
        img = img.float()
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        prediction = self.model(img)[0]

        prediction = find_max_confidence_bbox(prediction)[0]

        # Rescale boxes from img_size to im0 size
        prediction[:, :4] = scale_coords(img.shape[2:], prediction[:, :4], im0.shape).round()

        xyxy = prediction[0][:4]
        predictions = self.similar_images_finder.find_card_in_db(xyxy, im0)

        metadata_result = [self.metadata[prediction] for prediction in predictions]

        result = list()

        for metadata in metadata_result:
            metadata_d = dict()
            metadata_d['set'] = metadata['console-name']
            metadata_d['name'] = metadata['product-name']
            metadata_d[
                '_url'] = f"https://commondatastorage.googleapis.com/images.pricecharting.com/{metadata['cover-art']}/1600.jpg"
            metadata_d['product_id'] = metadata['id']
            result.append(metadata_d)

        return result
