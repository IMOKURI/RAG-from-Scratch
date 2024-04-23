import logging
from operator import itemgetter
import os

import bs4
from langchain import hub
from langchain.load import dumps, loads
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.web_base import WebBaseLoader
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


def main():
    model_name = os.getenv("MODEL_NAME", "gpt-3.5-turbo")

    logging.basicConfig(level=logging.INFO)

    ################################################################################
    # INDEXING
    ################################################################################

    # Load Documents
    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    docs = loader.load()

    # Split
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=300, chunk_overlap=50
    )

    splits = text_splitter.split_documents(docs)

    # Embed
    vectorstore = Chroma.from_documents(
        documents=splits, embedding=OpenAIEmbeddings(model=model_name)
    )
    retriever = vectorstore.as_retriever()

    ################################################################################
    # GENERATE MULTI QUERIES
    ################################################################################

    # Prompt
    # Multi Query: Different Perspectives
    # MultiQueryRetrieverは、LLMを用いて、与えられたユーザ入力クエリに対して異なる視点から複数のクエリを生成することで、
    # プロンプトチューニングのプロセスを自動化する。
    template = """You are an AI language model assistant. Your task is to generate five 
    different versions of the given user question to retrieve relevant documents from a vector 
    database. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search. 
    Provide these alternative questions separated by newlines. Original question: {question}"""
    """
    あなたはAI言語モデルアシスタントです。あなたの仕事は、ベクトル・データベースから関連文書を検索するために、
    与えられたユーザーの質問に対して5つの異なるバージョンを生成することです。
    ユーザの質問に対する複数の視点を生成することで、あなたのゴールは、
    ユーザが距離ベースの類似検索の制限のいくつかを克服するのを助けることです。
    改行で区切られたこれらの代替の質問を提供してください。
    """
    prompt_perspectives = ChatPromptTemplate.from_template(template)

    # Chain
    generate_queries = (
        prompt_perspectives
        | ChatOpenAI(model=model_name, temperature=0)
        | StrOutputParser()
        | (lambda x: x.split("\n"))
    )

    # Retrieve
    question = "What is task decomposition for LLM agents?"
    retrieval_chain = generate_queries | retriever.map() | get_unique_union

    docs = retrieval_chain.invoke({"question": question})
    logging.info(f"Number of retrieval documents: {len(docs)}")
    logging.info(f"Retrieval documents: {docs}")

    ################################################################################
    # RAG
    ################################################################################

    # RAG
    template = """Answer the following question based on this context:

    {context}

    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    llm = ChatOpenAI(model=model_name, temperature=0)

    final_rag_chain = (
        {"context": retrieval_chain, "question": itemgetter("question")}
        | prompt
        | llm
        | StrOutputParser()
    )

    result = final_rag_chain.invoke({"question": question})

    logging.info(f"Result: {result}")


def get_unique_union(documents: list[list]):
    """Unique union of retrieved docs"""
    # Flatten list of lists, and convert each Document to string
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    # Get unique documents
    unique_docs = list(set(flattened_docs))
    # Return
    return [loads(doc) for doc in unique_docs]


if __name__ == "__main__":
    main()
