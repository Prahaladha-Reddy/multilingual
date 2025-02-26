import dotenv
import os
from pinecone import Pinecone
dotenv.load_dotenv()
from mongo import collection
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from bson.objectid import ObjectId


PINECONE_API_KEY=os.getenv('PINECONE_API_KEY')
GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("lumora-test")
uri=os.getenv('uri')
collection=collection(uri)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=GOOGLE_API_KEY)

def get_result(query,similar_result=3):
  embedding=embeddings.embed_query(query)
  result=index.query(
    vector=embedding,
    top_k=similar_result,
  )
  mylist=[]
  list_text=[]
  for i in  range(len(result["matches"])):
    value=result["matches"][i]['id']
    mylist.append(collection.find_one({"_id": ObjectId(value)}))
  for i in range(len(mylist)):
    list_text.append(mylist[i]['fullplot'])
  
  return list_text

  

  






