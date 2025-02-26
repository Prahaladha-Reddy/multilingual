from pinecone_retriever import get_result
from typing import Dict, List, Any
from langchain.chains.base import Chain
from retrieval import RetrievalSystem
from chunks import chunks_maker
from retrieval import RetrievalSystem
from rag import RAGPipeline
from typing import Dict, List, Any
from langchain.chains.base import Chain
import re
import base64
from langchain.chains import SequentialChain
from langchain.chains import LLMChain
from langchain.schema import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
llm=ChatGroq()
prompt_template="""
You are a helpful event manager , you help users know about all the events and what are those events. you take a look the below events and explain the user about those events in the format of when the event is and venue if present and explain about the event 

Date:
venue:
explanation:


Events:
{events}

"""

prompt = PromptTemplate.from_template(prompt_template)


def ask_event_manager(query):
  results=get_result(query=query)
  new_prompt=prompt.format(events=results)
  res=llm.invoke(new_prompt)
  return res

