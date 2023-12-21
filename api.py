from flask import Flask, request

from CoreQA import CoreQA
from initialization import create_vectordb_from_scratch

app = Flask(__name__)


@app.route('/')
def home_api():
    return 'Welcome to Question Answering service'


@app.route('/message', methods=['POST'])
def question():
    try:
        if request.is_json:
            data = request.get_json()
            input_text = data['question']
            response = core_qa.inference(input_text)
            return response, 200
        return {"error": "request format not valid ..."}
    except Exception as eee:
        return f"internal error or bad request {eee}"


@app.route('/create_vectordb', methods=['POST'])
def create_vectordb():
    try:
        if request.is_json:
            data = request.get_json()
            data_path = data['file_path']
            embedding_name = data['embedding_name']
            vector_db_path = data['vector_db_path']
            create_vectordb_from_scratch(data_path, embedding_name, vector_db_path)
            return "VectorDB created successfully!", 200
        return {"error": "request format not valid ..."}
    except Exception as eee:
        return f"internal error or bad request {eee}"


if __name__ == '__main__':

    config = {
        'embedding_name': "sentence-transformers/LaBSE",
        'vectordb_path': "assets/faiss_index",
        'data_path': 'assets/faq_samples.csv',
    }
    core_qa = CoreQA(config)

    app.run(debug=True, host='0.0.0.0', port=5059)
