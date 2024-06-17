from langchain_community.document_loaders import PyPDFLoader
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain.chains.summarize import load_summarize_chain
import shutil
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
import os
import uuid
import requests
from fastapi.staticfiles import StaticFiles
from datetime import datetime as Date
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

# Initialize the LLM with the API key
llm = ChatGoogleGenerativeAI(model="gemini-pro")

# Configure CORS
origins = [
    "http://127.0.0.1",
    "http://127.0.0.1:8000/process",
    "http://127.0.0.1:5500",
    "http://localhost:5173",
]

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/saves", StaticFiles(directory="saves"), name="saves")
app.mount("/assignment", StaticFiles(directory="assignment"), name="assignment")

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/process/")
async def process_file(file: UploadFile = File(...)):
    try:
        unique_filename = f"{uuid.uuid4()}_{file.filename}"
        file_path = f"saves/{unique_filename}"
        print("Saving file to:", file_path)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        loader = PyPDFLoader(file_path)
        docs = loader.load_and_split()
        
        prompt_template = """Provide a detailed and thorough summary of the following document: "{text}" """
        prompt = PromptTemplate.from_template(template=prompt_template)
        
        chain = load_summarize_chain(llm, chain_type="map_reduce", token_max=6000)
        
        all_summaries = []
        for _ in range(3):
            summaries = []
            for doc in docs:
                summary = chain.run([doc])
                summaries.append(summary)
            combined_summary = " ".join(summaries)
            all_summaries.append(combined_summary)
           
        print("Returning response:", {"filename": unique_filename, "summaries": all_summaries})
        
        requests.post("http://localhost:3000/material/", json={"fileNameUrl": unique_filename, "summary": all_summaries})
        
        return JSONResponse(content={"filename": unique_filename, "summaries": all_summaries}, status_code=200)
    except Exception as e:
        print("Error:", e)
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/upload_assignment/")
async def upload_assignment(
    assignmentTitle: str = Form(...),
    description: str = Form(""),
    deadline: str = Form(...),
    totalPoints: int = Form(...),
    subject: str = Form(...),
    chapterId: str = Form(...),
    file: UploadFile = File(...)
):
    try:
        unique_filename = f"{uuid.uuid4()}_{file.filename}"
        file_path = f"assignment/{unique_filename}"
        print("Saving assignment file to:", file_path)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        assignment_data = {
            "assignmentTitle": assignmentTitle,
            "description": description,
            "deadline": deadline,
            "totalPoints": totalPoints,
            "subject": subject,
            "chapterId": chapterId,
            "assignmentUrl": f"http://localhost:8000/assignment/{unique_filename}",
            "publishedAt": Date.now().isoformat(),
        }

        submission_url = "http://localhost:3000/assignment/createAssignment"
        headers = {"Content-Type": "application/json"}
        print("Sending assignment data to external server:", assignment_data)
        response = requests.post(submission_url, json=assignment_data, headers=headers)
        print("Received response status:", response.status_code)
        print("Received response content:", response.content)

        if response.status_code != 201:
            raise HTTPException(status_code=response.status_code, detail=response.json())

        return JSONResponse(content={"filename": unique_filename, "assignment": response.json()}, status_code=200)
    except Exception as e:
        print("Error:", e)
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/submit_assignment/")
async def submit_assignment(
    assignmentId: str = Form(...),
    studentId: str = Form(...),
    file: UploadFile = File(...)
):
    try:
        unique_filename = f"{uuid.uuid4()}_{file.filename}"
        file_path = f"saves/{unique_filename}"
        print("Saving student submission file to:", file_path)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        submission_data = {
            "assignmentId": assignmentId,
            "studentId": studentId,
            "content": f"http://localhost:8000/saves/{unique_filename}",
            "submissionDate": Date.now().isoformat(),
            "status": "Submitted",
        }

        submission_url = "http://localhost:3000/submission/createSubmission"
        headers = {"Content-Type": "application/json"}
        response = requests.post(submission_url, json=submission_data, headers=headers)

        if response.status_code != 201:
            raise HTTPException(status_code=response.status_code, detail=response.json())

        return JSONResponse(content={"filename": unique_filename, "submission": response.json()}, status_code=200)
    except Exception as e:
        print("Error:", e)
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
