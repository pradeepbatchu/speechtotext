from flask import Flask, jsonify, request
import logging,sys
from flask_cors import CORS, cross_origin
from toruch_audio import audiototext
sys.setrecursionlimit(15000)
app = Flask(__name__)
CORS(app)
cors = CORS(app, resources={r"/*": {"origins": "*"}})
logging.getLogger().level = logging.DEBUG


@app.route('/api/audio/text', methods=['POST'])
@cross_origin(origin='*')
def speech_to_text():
    audio = request.files['audio']
    text = audiototext(audio)
    #print(text)
    output = {'speechtotext': text}
    return jsonify(output)


@app.route('/api/help', methods=['GET'])
def help():
    endpoints = [rule.rule for rule in app.url_map.iter_rules()
                 if rule.endpoint !='static']
    return jsonify(dict(api_endpoints=endpoints))


@app.route("/api/routes", methods=["GET"])
def get_Routes():
    routes = {}
    for r in app.url_map._rules:
        routes[r.rule] = {}
        routes[r.rule]["functionName"] = r.endpoint
        routes[r.rule]["methods"] = list(r.methods)

    routes.pop("/static/<path:filename>")

    return jsonify(routes)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
