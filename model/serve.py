import json

from flask import Flask, request, jsonify


def get_api_server(inverted_index_file="./inverted_index.json"):
    app = Flask(__name__)

    version = 'v1.0'

    info = {
        'name': 'yelp crm api',
        'version': version
    }

    with open(inverted_index_file, 'r') as fp:
        inverted_index = json.load(fp)

    @app.route("/info")
    def info():
        return jsonify({'server info': info})

    @app.route("/b2b", methods=['POST'])
    def get_item2item():
        content = request.json
        business_id = content['business_id']

        # DL 모델 학습을 하지 못해서 데이터를 가지고 오는 부분을 구현하지 못했습니다.

        if business_id in inverted_index:
            recommended_business_ids = inverted_index[business_id]
            response = {
                "code": "SUCCESS",
                'business_id': business_id,
                'recommended_business_ids': recommended_business_ids
            }
        else:
            response = {
                "code": "FAILURE",
                "message": f"{business_id} does not exist",
                'business_id': business_id,
            }

        return jsonify(response)
    return app


def serve():
    api_server = get_api_server()
    api_server.run(host='0.0.0.0', port=5000, debug=True)


if __name__ == '__main__':
    serve()
