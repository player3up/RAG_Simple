# main.py
import os
import shutil
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag_core import rag_engine

app = FastAPI(title="RAG Knowledge Base API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    sources: list


@app.post("/upload_doc")
async def upload_document(file: UploadFile = File(...)):
    try:
        os.makedirs("data", exist_ok=True)
        file_location = os.path.join("data", file.filename)
        with open(file_location, "wb") as f:
            shutil.copyfileobj(file.file, f)

        rag_engine.load_and_process_documents(file_location)

        return {"info": f"文件 '{file.filename}' 已成功上传并处理。"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    try:
        answer, sources = rag_engine.query(request.question)

        formatted_sources = [
            {"source": doc.metadata.get('source', '未知来源'), "content": doc.page_content}
            for doc in sources
        ]

        return QueryResponse(answer=answer, sources=formatted_sources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    return {"message": "RAG API is running. Go to /docs for API documentation."}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)