import logging
import os

import bs4
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.web_base import WebBaseLoader
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


def main():
    model_name = os.getenv("MODEL_NAME", "gpt-3.5-turbo")

    logging.basicConfig(level=logging.INFO)

    #### INDEXING ####

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
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # Embed
    vectorstore = Chroma.from_documents(
        documents=splits, embedding=OpenAIEmbeddings(model=model_name)
    )

    retriever = vectorstore.as_retriever()

    #### RETRIEVAL and GENERATION ####

    # Prompt
    prompt = hub.pull("rlm/rag-prompt")

    # LLM
    llm = ChatOpenAI(model=model_name, temperature=0)

    # Post-processing
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Question
    result = rag_chain.invoke("What is Task Decomposition?")

    logging.info(f"Result: {result}")
    logging.info("Done...")

if __name__ == "__main__":
    main()
