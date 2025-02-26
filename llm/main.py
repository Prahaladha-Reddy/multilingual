from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationSummaryMemory
from langchain.chains import LLMChain
from langchain.output_parsers import StructuredOutputParser
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from fastapi import FastAPI, HTTPException

from pydantic import BaseModel
from typing import List,Dict



llm=ChatGroq(model="mistral-saba-24b")


buffer_memory=ConversationBufferMemory()

# Now we can override it and set it to "AI Assistant"
from langchain_core.prompts.prompt import PromptTemplate

template = """The following is a friendly conversation between a human and an AI. The AI is is multilingual speaks telugu , hindi and english and expert in various achademic domain and helps humans in their won language. 
Current conversation:
{history}
Human: {input}
AI Assistant:"""


PROMPT = PromptTemplate(input_variables=["history", "input"], template=template)

conversation = ConversationChain(
    prompt=PROMPT,
    llm=llm,
    verbose=True,
    memory=ConversationBufferMemory(ai_prefix="AI Assistant"),
)

app = FastAPI(
    title="RAG API",
    description="API for Retrieval-Augmented Generation",
    version="1.0.0"
)

class AnswerResponse(BaseModel):
    answer: str


class QuestionRequest(BaseModel):
    question: str


@app.post("/ask_multilingual", response_model=AnswerResponse)
async def ask_question_multilingual(request: QuestionRequest):
    try:
        retrieved_docs = conversation.predict(input=request.question)

        return AnswerResponse(
            answer=retrieved_docs
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing question: {str(e)}"
        )


@app.post("/ask_events", response_model=AnswerResponse)
async def ask_question_multilingual(request: QuestionRequest):
    try:
        retrieved_docs = conversation.predict(input=request.question)

        return AnswerResponse(
            question=retrieved_docs[2],
            answer=retrieved_docs[0],
            Encoded_Images=retrieved_docs[1]
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing question: {str(e)}"
        )
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# To run the API
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)