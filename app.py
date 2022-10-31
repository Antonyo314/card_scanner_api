import numpy as np
from PIL import Image
from flask import Flask, request, jsonify

from utils.model import Model

API_TOKEN = 'e121166d02ab7dde5dff5bffed004b72'


def token_is_valid(token):
    return token == API_TOKEN


app = Flask(__name__)
model = Model()


@app.route('/api', methods=['POST'])
def process_image():
    data = request.form.to_dict()

    token = data['token']
    if not token_is_valid(token):
        output = {'status': {
            'code': 403,
            'text': 'The provided API token is not valid'}}
        return jsonify(output)

    file = request.files['image']

    img = Image.open(file.stream)

    img = np.array(img)
    img = img[:, :, ::-1]  # BGR
    metadata_result = model.detect_card_and_find_in_db(img)

    output = {'status': {
        'code': 200,
        'text': 'OK'
    },
        'answer_records': metadata_result}

    return jsonify(output)


if __name__ == '__main__':
    app.run(debug=True)
