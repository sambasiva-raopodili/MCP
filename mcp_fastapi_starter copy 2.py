# MCP Starter: FastAPI + LangChain + Bitbucket Integration (Updated for LangChain Local LLM with Startup Validation and Background Processing)

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict
import requests
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_community.llms import Ollama
from git import Repo
import tempfile
from dotenv import load_dotenv
from uuid import uuid4
from threading import Lock

load_dotenv()  # Load variables from .env if available

app = FastAPI()

# === Configuration === #
BITBUCKET_REPO_URL = os.getenv("BITBUCKET_REPO_URL", "https://bitbucket.org/your-org/your-repo.git")
BITBUCKET_BRANCH = os.getenv("BITBUCKET_BRANCH", "feature/mcp-generated")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")  # default to llama3

# === Validate Ollama Startup === #
def validate_ollama_model(model: str):
    try:
        response = requests.get(f"http://localhost:11434/api/tags")
        if response.status_code != 200:
            raise Exception("Ollama server not responding")

        available_models = [m["name"] for m in response.json().get("models", [])]
        if model not in available_models:
            raise Exception(f"Model '{model}' not found. Please run: ollama pull {model}")

    except Exception as e:
        raise RuntimeError(f"Ollama validation failed: {e}")

validate_ollama_model(OLLAMA_MODEL)
llm = Ollama(model=OLLAMA_MODEL)

# === Request & Status Model === #
class MCPRequest(BaseModel):
    prompt: str
    service_name: str
    context_files: List[str]

# === Status Tracking === #
task_status: Dict[str, str] = {}
task_lock = Lock()

# === Utils === #
def load_context(files: List[str]) -> str:
    context_data = ""
    for file in files:
        if os.path.exists(file):
            with open(file, "r") as f:
                context_data += f"\n# From {file}:\n" + f.read() + "\n"
    return context_data

def build_prompt_template() -> PromptTemplate:
    return PromptTemplate(
        input_variables=["context", "task"],
        template="""
You are a backend developer working in a Java-based microservices architecture for a fintech company.

Below is the coding context:

{context}

Write **Java** business logic and its corresponding **JUnit test case** for the following requirement:

{task}

Ensure:
- Use Java syntax
- Apply standard Java coding practices
- Include necessary class structure, imports, and method stubs
- Unit tests should follow JUnit 5 standards
"""
    )

def clone_repo(temp_dir: str):
    return Repo.clone_from(BITBUCKET_REPO_URL, temp_dir)

def push_code(service_name: str, code: str):
    with tempfile.TemporaryDirectory() as tmp_dir:
        repo = clone_repo(tmp_dir)
        file_path = os.path.join(tmp_dir, f"services/{service_name}.java")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            f.write(code)

        repo.git.checkout("-b", BITBUCKET_BRANCH)
        repo.index.add([file_path])
        repo.index.commit(f"Add {service_name} service via MCP")
        origin = repo.remote(name='origin')
        origin.push(BITBUCKET_BRANCH)

def run_generation(req: MCPRequest, task_id: str):
    try:
        context = load_context(req.context_files)
        prompt_template = build_prompt_template()
        runnable = prompt_template | llm

        response = runnable.invoke({
            "context": context,
            "task": req.prompt
        })

        if response:
            push_code(req.service_name, response)
            with task_lock:
                task_status[task_id] = "completed"
        else:
            with task_lock:
                task_status[task_id] = "failed: empty response"
    except Exception as e:
        with task_lock:
            task_status[task_id] = f"failed: {str(e)}"

# === Endpoints === #
@app.get("/")
def health_check():
    return {"status": f"MCP service is running (Ollama mode: {OLLAMA_MODEL})"}

@app.post("/generate")
def generate_code(req: MCPRequest, background_tasks: BackgroundTasks):
    task_id = str(uuid4())
    with task_lock:
        task_status[task_id] = "started"
    background_tasks.add_task(run_generation, req, task_id)
    return {"message": "Generation task started in background.", "task_id": task_id}

@app.get("/status/{task_id}")
def get_status(task_id: str):
    with task_lock:
        status = task_status.get(task_id, "unknown")
    return {"task_id": task_id, "status": status}
