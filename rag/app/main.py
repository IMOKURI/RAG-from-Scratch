import logging
from operator import itemgetter
import os

import bs4
from langchain.load import dumps, loads
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.web_base import WebBaseLoader
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


def main():
    logging.basicConfig(level=logging.INFO)

    model_name = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
    query_translation = os.getenv("QUERY_TRANSLATION", "RAG_FUSION")

    logging.info(f"Settings: {query_translation=}")

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

    if query_translation == "MULRI_QUERY":
        # Prompt
        # Multi Query: Different Perspectives
        # MultiQueryRetrieverは、LLMを用いて、与えられたユーザ入力クエリに対して異なる視点から複数のクエリを生成することで、
        # プロンプトチューニングのプロセスを自動化する。
        # https://python.langchain.com/docs/modules/data_connection/retrievers/MultiQueryRetriever/
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
        retrieval_chain = generate_queries | retriever.map() | get_unique_union

        question = "What is task decomposition for LLM agents?"
        docs = retrieval_chain.invoke({"question": question})
        logging.info(f"Number of retrieval documents: {len(docs)}")
        logging.info(f"Retrieval documents: {docs}")

    ################################################################################
    # RAG-FUSION
    ################################################################################

    elif query_translation == "RAG_FUSION":
        """
        RAG-Fusion の流れ
        1. 複数クエリの生成: LLMを用いて、ユーザーのクエリを似ているが異なるクエリへと変換
        2. ベクトル検索: 元のクエリと新たに生成されたクエリに対してベクトル検索を実行
        3. コンテキストのリランキング: Reciprocal Rank Fusionを用いて、すべての結果を集約し、整える
        4. Outputの生成: 選び抜かれた結果を新しいクエリと組み合わせ、すべてのクエリとリランキングされた結果のリストを
        考慮しながら、大規模言語モデルを質の高いOutputへ導く
        https://zenn.dev/ozro/articles/abfdadd0bfdd7a
        """

        # RAG-Fusion: Related
        template = """You are a helpful assistant that generates multiple search queries based on a single input query. \n
        Generate multiple search queries related to: {question} \n
        Output (4 queries):"""
        prompt_rag_fusion = ChatPromptTemplate.from_template(template)

        generate_queries = (
            prompt_rag_fusion
            | ChatOpenAI(model=model_name, temperature=0)
            | StrOutputParser()
            | (lambda x: x.split("\n"))
        )

        retrieval_chain = generate_queries | retriever.map() | reciprocal_rank_fusion

        question = "What is task decomposition for LLM agents?"
        docs = retrieval_chain.invoke({"question": question})
        logging.info(f"Number of retrieval documents: {len(docs)}")
        logging.info(f"Retrieval documents: {docs}")

    else:
        raise Exception("Invalid query_translation")

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


def reciprocal_rank_fusion(results: list[list], k=60):
    """Reciprocal_rank_fusion that takes multiple lists of ranked documents
    and an optional parameter k used in the RRF formula"""

    # Initialize a dictionary to hold fused scores for each unique document
    fused_scores = {}

    # Iterate through each list of ranked documents
    for docs in results:
        # Iterate through each document in the list, with its rank (position in the list)
        for rank, doc in enumerate(docs):
            # Convert the document to a string format to use as a key (assumes documents can be serialized to JSON)
            doc_str = dumps(doc)
            # If the document is not yet in the fused_scores dictionary, add it with an initial score of 0
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            # Retrieve the current score of the document, if any
            previous_score = fused_scores[doc_str]
            # Update the score of the document using the RRF formula: 1 / (rank + k)
            fused_scores[doc_str] += 1 / (rank + k)

    # Sort the documents based on their fused scores in descending order to get the final reranked results
    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    # Return the reranked results as a list of tuples, each containing the document and its fused score
    return reranked_results


if __name__ == "__main__":
    main()
