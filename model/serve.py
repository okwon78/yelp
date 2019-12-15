from flask import Flask, request, jsonify


def get_api_server():
    app = Flask(__name__)

    version = 'v1.0'

    info = {
        'name': 'ap recsys api',
        'version': version
    }

    @app.route("/info")
    def info():
        return jsonify({'server info': info})

    @app.route("/b2b", methods=['POST'])
    def get_item2item():
        content = request.json
        business_id = content['business_id']

        # DL 모델 학습을 하지 못해서 데이터를 가지고 오는 부분을 구현하지 못했습니다.
        response = {
            'business_id': business_id,
            'recommended_business_ids': []
        }

        return jsonify(response)
    return app


def serve():
    api_server = get_api_server()
    api_server.run(host='0.0.0.0', port=5000, debug=True)


if __name__ == '__main__':
    serve()
