from flask import Flask, request
from flask_cors import CORS, cross_origin
app = Flask(__name__)
from bert import process
from translate import convert, src_lang
CORS(app, support_credentials=True, withCredentials=True,
     CORS_SUPPORTS_CREDENTIALS=True)

@app.route('/api/receive-text', methods=['POST'])
def receive_text():
    received_text = request.data.decode('utf-8')
    print("Văn bản nhận được:", received_text)
    eng_text = convert(received_text,'en')
    print(eng_text)
    a = process(eng_text)
    a['summary']=convert(a['summary'],src_lang(received_text))
    return a

from os import environ

if __name__ == '__main__':
    HOST = environ.get('SERVER_HOST', 'localhost')
    try:
        PORT = int(environ.get('SERVER_PORT', '5555'))
    except ValueError:
        PORT = 5555
    app.run(HOST, PORT, debug=True)
