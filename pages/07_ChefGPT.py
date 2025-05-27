# import os
# from pinecone import Pinecone, ServerlessSpec
# from langchain_openai import OpenAIEmbeddings
# from langchain_pinecone import PineconeVectorStore
# import time
# from langchain.document_loaders import CSVLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st

st.set_page_config(
    page_title="ChefGPT",
    page_icon="�‍🍳",
)

st.markdown(
    """
# ChefGPT  

**To be continued...**

Recipe recommendations and cooking assistant
"""
)

# pc = Pinecone(
#     api_key=os.environ.get("PINECONE_API_KEY"),
# )

# # splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder()

# # loader = CSVLoader("./recipes.csv")

# # docs = loader.load_and_split(
# #     text_splitter=splitter,
# # )


# embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# index_name = "recipes"

# # if not pc.has_index(index_name):
# #     pc.create_index(
# #         name=index_name,
# #         dimension=1536,
# #         metric="cosine",
# #         spec=ServerlessSpec(
# #             cloud="aws",
# #             region="us-east-1",
# #         ),
# #     )

# index = pc.Index(index_name)

# # # Functions for batch processing
# # def process_in_batches(documents, batch_size=100):
# #     for i in range(0, len(documents), batch_size):
# #         batch = documents[i : i + batch_size]
# #         try:
# #             vector_store = PineconeVectorStore.from_documents(
# #                 documents=batch, embedding=embeddings, index_name=index_name
# #             )
# #             print(
# #                 f"Processed batch {i//batch_size + 1} of {len(documents)//batch_size + 1}"
# #             )
# #             time.sleep(1)
# #         except Exception as e:
# #             print(f"Error processing batch {i//batch_size + 1}: {e}")
# #             continue


# # # Processing in batches
# # process_in_batches(docs, batch_size=50)

# vector_store = PineconeVectorStore(
#     index=index,
#     embedding=embeddings,
# )

# results = vector_store.similarity_search(
#     "tofu",
#     k=2,
# )
# for res in results:
#     print(f"* {res.page_content} [{res.metadata}]")
