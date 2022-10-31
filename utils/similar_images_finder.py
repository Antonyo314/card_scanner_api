import cv2
import numpy as np
from PIL import Image
from img2vec_pytorch import Img2Vec
from scipy import spatial


class SimilarImagesFinder:
    def __init__(self, embeddings_dict_list_path, embeddings_matrix_path, top_n=5):
        self.embeddings_dict_list = np.load(embeddings_dict_list_path, allow_pickle=True)
        self.embeddings_matrix = np.load(embeddings_matrix_path)

        self.embeddings_d = {}

        for i in self.embeddings_dict_list:
            self.embeddings_d[list(i.keys())[0]] = list(i.values())[0]

        self.img2vec = Img2Vec(cuda=False)
        self.top_n = top_n

    def embeddings(self, im):
        im = Image.fromarray(im)
        try:
            vector = self.img2vec.get_vec(im)
        except:
            imRGB = np.repeat(np.array(im)[:, :, np.newaxis], 3, axis=2)
            vector = self.img2vec.get_vec(Image.fromarray(imRGB))
        return vector.tolist()

    @staticmethod
    def distance(img_1, img_2):
        return np.linalg.norm(np.array(img_1) - np.array(img_2))

    def find_most_similar(self, target_value):
        embeddings_target = np.array(target_value)

        distance_vector = spatial.distance.cdist([embeddings_target], self.embeddings_matrix, metric='euclidean')
        distance_vector = distance_vector[0]

        ind = np.argpartition(-distance_vector, -self.top_n)[-self.top_n:]
        ind = ind[np.argsort(distance_vector[ind])]

        results = [list(self.embeddings_d)[i] for i in ind]

        return results

    def find_card_in_db(self, x, img):
        crop_img = img[int(x[1]):int(x[3]), int(x[0]):int(x[2])]
        crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)

        input_emb = self.embeddings(crop_img)

        result = self.find_most_similar(input_emb)

        return result
