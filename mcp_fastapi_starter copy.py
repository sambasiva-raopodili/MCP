# MCP Starter: FastAPI + LangChain + Bitbucket Integration (Enhanced for Repository Context Awareness)

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict
import requests
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_community.llms import Ollama
import tempfile
from dotenv import load_dotenv
from uuid import uuid4
from threading import Lock
import subprocess

load_dotenv()  # Load variables from .env if available

app = FastAPI()

# === Configuration === #
BITBUCKET_REPO_URL = os.getenv("BITBUCKET_REPO_URL", "https://github.com/sambasiva-raopodili/spring-boot-microservices-series.git")
BITBUCKET_BRANCH = os.getenv("BITBUCKET_BRANCH", "master")
LOCAL_REPO_PATH = os.getenv("LOCAL_REPO_PATH", "/tmp/mcp_repo")
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
task_result: Dict[str, str] = {}
task_lock = Lock()

# === Utils === #
def clone_repo_if_needed():
    if not os.path.exists(LOCAL_REPO_PATH):
        subprocess.run(["git", "clone", BITBUCKET_REPO_URL, LOCAL_REPO_PATH], check=True)
    subprocess.run(["git", "-C", LOCAL_REPO_PATH, "checkout", BITBUCKET_BRANCH], check=True)

def scan_all_code_files(root_dir: str) -> List[str]:
    context = []
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f.endswith(".java") or f.endswith(".kt") or f.endswith(".txt"):
                context.append(os.path.join(dirpath, f))
    return context

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
You are a senior Java backend engineer working in a Spring Boot-based microservices architecture.

You are provided with relevant project context and a functional requirement:

{context}

Write detailed, enterprise-grade Java code using **Spring Boot**, applying the following standards:

1. Generate a complete business logic class (e.g., service or controller) using Spring annotations.
2. Use DTOs and validation annotations (`@Valid`, `@NotNull`, etc.) appropriately.
3. Handle exceptions using Spring's `@ControllerAdvice` or inline try-catch.
4. Include proper logging with SLF4J (`LoggerFactory`).
5. Adhere to clean architecture: service, repository, model separation.
6. Java 11+ features (streams, var, Optional) can be used where relevant.

Also, generate a comprehensive **JUnit 5** test class that:
- Mocks dependencies with Mockito
- Tests both normal and edge-case scenarios
- Uses `@SpringBootTest` or `@WebMvcTest` as applicable

Output only the source code blocks in Java.
"""
    )

def run_generation(req: MCPRequest, task_id: str):
    try:
        clone_repo_if_needed()
        all_repo_files = scan_all_code_files(LOCAL_REPO_PATH)
        combined_files = list(set(all_repo_files + req.context_files))
        context = load_context(combined_files)

        prompt_template = build_prompt_template()
        runnable = prompt_template | llm

        response = runnable.invoke({
            "context": context,
            "task": req.prompt
        })

        if response:
            with task_lock:
                task_status[task_id] = "completed"
                task_result[task_id] = response
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
def generate_code(req: MCPRequest, background_tasks: BackgroundTasks, sync: bool = False):
    task_id = str(uuid4())
    with task_lock:
        task_status[task_id] = "started"

    if sync:
        run_generation(req, task_id)
        with task_lock:
            return {"task_id": task_id, "status": task_status[task_id], "result": task_result.get(task_id, "")}
    else:
        background_tasks.add_task(run_generation, req, task_id)
        return {"message": "Generation task started in background.", "task_id": task_id}

@app.get("/status/{task_id}")
def get_status(task_id: str):
    with task_lock:
        status = task_status.get(task_id, "unknown")
        result = task_result.get(task_id)
    return {"task_id": task_id, "status": status, "result": result}
