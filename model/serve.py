import jsonify as jsonify
from flask import Flask, request


def get_api_server():
    app = Flask()

    version = 'v1.0'

    info = {
        'name': 'ap recsys api',
        'version': version
    }

    @app.route("/info")
    def info():
        return jsonify({'server info': info})

    @app.route("/item2item", methods=['POST'])
    def get_item2item():
        content = request.json
        business_id = content['business_id']

        response = {
            'business_id': business_id,
            'recommended_business_ids': []
        }

        return jsonify(response)
    return app


def serve():
    api_server = get_api_server()
    api_server.run(host='0.0.0.0', debug=True)


if __name__ == '__main__':
    serve()
