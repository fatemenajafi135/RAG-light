import os
from langchain.document_loaders import CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS


def data_loader(file_path):
    loader = CSVLoader(file_path)
    data = loader.load()
    return data


def hf_embedding(model_name):
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embedding = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    return embedding


def create_vectordb(data, embedding, path_to_save):
    vectordb = FAISS.from_documents(data, embedding)
    vectordb.save_local(path_to_save)
    return vectordb


def create_vectordb_from_scratch(data_path, embedding_name, vector_db_path):
    data = data_loader(data_path)
    embedding = hf_embedding(embedding_name)
    vectordb = create_vectordb(data, embedding, vector_db_path)
    return vectordb


def load_vectordb(vectordb_path, data_path, embedding):
    if os.path.exists(vectordb_path):
        vectordb = FAISS.load_local(vectordb_path, embedding)
    else:
        vectordb = create_vectordb_from_scratch(data_path, embedding, vectordb_path)
    return vectordb


