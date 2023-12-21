import os
import openai
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

from initialization import hf_embedding, load_vectordb


class CoreQA:

    def __init__(self, config):

        self.embedding = hf_embedding(model_name=config['embedding_name'])
        self.vectordb = load_vectordb(
            vectordb_path=config['vectordb_path'],
            data_path=config['data_path'],
            embedding=self.embedding
        )
        self.qa_chain = self.agent()

    def agent(self):

        load_dotenv('assets/.env')
        openai.api_key = os.environ['OPENAI_API_KEY']

        template = """You're a Question Answering System responding based on a provided context. 
            Given retrieved texts and a question, answer in at most four sentences, ensuring linguistic completeness. 
            Adhere to rules: maintain brevity, retain numbers and entities unchanged, 
            reply "I don't know!" if unknown or for irrelevant or missing info. 

            {context}

            Question: {question}
            Answer: """
        qa_prompt = PromptTemplate.from_template(template)

        qa_chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(
                model_name="gpt-3.5-turbo",
                temperature=0,
            ),
            return_source_documents=True,
            chain_type="stuff",
            retriever=self.vectordb.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
            ),
            # verbose=True,
            chain_type_kwargs={"prompt": qa_prompt}
        )

        return qa_chain

    def inference(self, question):
        result = self.qa_chain({"query": question})
        answer = result["result"]
        return answer
