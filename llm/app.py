
from rag2 import ImageRetrievalChain,DocumentRetrievalChain,Encode_Images,LLMChain
from schema import QuestionRequest,AnswerResponse
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from langchain.chains import SequentialChain
from events_pipeline import ask_event_manager

retrieval_chain = DocumentRetrievalChain()
image_retrieval_chain = ImageRetrievalChain()
encode_images=Encode_Images()
llm_chain = LLMChain()
pipeline = SequentialChain(
    chains=[retrieval_chain, image_retrieval_chain,encode_images, llm_chain],
    input_variables=["query"],
    output_variables=["response"]
)


app = FastAPI(
    title="RAG API",
    description="API for Retrieval-Augmented Generation",
    version="1.0.0"
)


@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    try:
        retrieved_docs = pipeline.run({"query":request.question})

        return AnswerResponse(
            question=retrieved_docs[2],
            answer=retrieved_docs[0],
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing question: {str(e)}"
        )




@app.post("/ask", response_model=AnswerResponse)
async def ask_events(request: QuestionRequest):
    try:
        retrieved_docs = ask_event_manager(query=request.question)

        return AnswerResponse(
            question=request.question,
            answer=retrieved_docs['content'],
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



