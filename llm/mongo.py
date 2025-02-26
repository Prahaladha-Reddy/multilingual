from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from typing import TypedDict
uri = "mongodb+srv://lumora-test:lumora-test@lumora-test.nabm4.mongodb.net/?retryWrites=true&w=majority&appName=lumora-test"

def collection(uri):
  client = MongoClient(uri, server_api=ServerApi('1'))
  db=client['lumora-test']
  collection=db['lumora-test-collection']
  return collection
