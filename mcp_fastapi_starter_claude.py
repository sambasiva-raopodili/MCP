# MCP Starter: FastAPI + Claude Sonnet (Anthropic) + Bitbucket Integration

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict
import requests
import os
import subprocess
import anthropic
from langchain_core.prompts import PromptTemplate
import tempfile
from dotenv import load_dotenv
from uuid import uuid4
from threading import Lock
import hashlib

load_dotenv()

app = FastAPI()

# === Configuration === #
BITBUCKET_WORKSPACE = os.getenv("BB_WORKSPACE")
BITBUCKET_USER = os.getenv("BB_USER")
BITBUCKET_APP_PASSWORD = os.getenv("BB_APP_PASSWORD")
BITBUCKET_LOCAL_CLONE_DIR = os.getenv("BB_LOCAL_CLONE_DIR", "cloned_repos")
BITBUCKET_PROJECT_FILTER = os.getenv("BB_PROJECT_FILTER", "").split(",")
USE_CLAUDE = os.getenv("USE_CLAUDE", "true").lower() == "true"
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-3-sonnet-20240229")

# === Claude Client === #
claude_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# === Request & Status Model === #
class MCPRequest(BaseModel):
    prompt: str
    service_name: str
    context_files: List[str]

# === Status Tracking and Cache === #
task_status: Dict[str, str] = {}
task_result: Dict[str, str] = {}
task_lock = Lock()
context_cache: Dict[str, str] = {}

# === Repo Utilities === #
def fetch_all_bitbucket_repos():
    url = f"https://api.bitbucket.org/2.0/repositories/{BITBUCKET_WORKSPACE}"
    repos = []

    while url:
        response = requests.get(url, auth=(BITBUCKET_USER, BITBUCKET_APP_PASSWORD))
        if response.status_code != 200:
            break
        data = response.json()
        repos.extend(data.get("values", []))
        url = data.get("next")

    paths = []
    os.makedirs(BITBUCKET_LOCAL_CLONE_DIR, exist_ok=True)
    for repo in repos:
        project_key = repo.get("project", {}).get("key", "")
        if BITBUCKET_PROJECT_FILTER and project_key not in BITBUCKET_PROJECT_FILTER:
            continue

        name = repo['name']
        clone_url = repo['links']['clone'][0]['href']
        dest = os.path.join(BITBUCKET_LOCAL_CLONE_DIR, name)
        if not os.path.exists(dest):
            subprocess.run(["git", "clone", clone_url, dest])
        paths.append(dest)
    return paths

def extract_context_from_repo(repo_path, file_types=(".java", ".md", ".yml")):
    repo_hash = hashlib.md5(repo_path.encode()).hexdigest()
    if repo_hash in context_cache:
        return context_cache[repo_hash]

    context = ""
    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.endswith(file_types):
                try:
                    with open(os.path.join(root, file), "r", encoding="utf-8", errors="ignore") as f:
                        context += f"\n# {file}:\n" + f.read()[:3000]
                except:
                    continue

    context_cache[repo_hash] = context
    return context

# === Local Context Loader === #
def load_context(files: List[str]) -> str:
    context_data = ""
    for file in files:
        if os.path.exists(file):
            with open(file, "r") as f:
                context_data += f"\n# From {file}:\n" + f.read() + "\n"
    return context_data

# === Prompt Template === #
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

# === Claude Runner === #
def call_claude(prompt: str) -> str:
    response = claude_client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=2048,
        temperature=0.7,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text

# === Code Generation Runner === #
def run_generation(req: MCPRequest, task_id: str):
    try:
        repo_paths = fetch_all_bitbucket_repos()
        total_context = "".join([extract_context_from_repo(p) for p in repo_paths])
        total_context += load_context(req.context_files)

        prompt_template = build_prompt_template()
        prompt = prompt_template.format(context=total_context, task=req.prompt)
        response = call_claude(prompt) if USE_CLAUDE else "Claude is disabled."

        with task_lock:
            task_status[task_id] = "completed"
            task_result[task_id] = response
    except Exception as e:
        with task_lock:
            task_status[task_id] = f"failed: {str(e)}"

# === FastAPI Endpoints === #
@app.get("/")
def health_check():
    return {"status": f"MCP service is running (Claude mode: {CLAUDE_MODEL if USE_CLAUDE else 'DISABLED'})"}

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
