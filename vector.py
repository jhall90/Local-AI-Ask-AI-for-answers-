# local database using ChromaDB
# allows us to quickly look up info to pass to the model
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

# df = dataframe and is commenly used
# **** CHANGE THIS TO THE CORRECT CSV IF YOU WANT TO PARSE ANOTHER ****
df = pd.read_csv("realistic_restaurant_reviews.csv")
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = "./chroma_langchain_db"
# check if database exists
# if it does it means we already converted the csv into vectors and adding to the db
add_documents = not os.path.exists(db_location)

if add_documents:
    documents = []
    ids = []

    for i, row in df.iterrows():
        document = Document(
            # for the restaurant csv we are taking the Title and Review (This is what is used to query)
            page_content=row["Title"] + " " + row["Review"],
            # Data included but not used for querying
            metadata={"rating": row["Rating"], "date": row["Date"]},
            id=str(i)
        )
        ids.append(str(i))
        documents.append(document)

# create the vector store
vector_store = Chroma(
    collection_name="restaurant_reviews",
    persist_directory=db_location, # stores it permanently instead of in memory
    embedding_function=embeddings
)

if add_documents:
    vector_store.add_documents(documents=documents, ids=ids) # type: ignore

# allows us to grab relevant documents 
retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}
)