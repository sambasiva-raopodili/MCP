# MCP Starter: FastAPI + LangChain + Bitbucket Integration (Updated for langchain-openai)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import requests
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_openai import OpenAI
from git import Repo
import tempfile
from dotenv import load_dotenv

load_dotenv()  # Load variables from .env if available

app = FastAPI()
#
# === Configuration === #
BITBUCKET_REPO_URL = os.getenv("BITBUCKET_REPO_URL", "https://bitbucket.org/your-org/your-repo.git")
BITBUCKET_BRANCH = os.getenv("BITBUCKET_BRANCH", "feature/mcp-generated")
MODEL_API_KEY = os.getenv("OPENAI_API_KEY")

llm = OpenAI(
    model="gpt-3.5-turbo-instruct",  # You can also use "gpt-4"
    temperature=0.2
)

# === Request Model === #
class MCPRequest(BaseModel):
    prompt: str
    service_name: str
    context_files: List[str]

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
You are a backend developer for a fintech company. Below is the coding context:

{context}

Write business logic and corresponding unit test for the following requirement:

{task}
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

# === Endpoints === #
@app.get("/")
def health_check():
    return {"status": "MCP service is running"}

@app.post("/generate")
def generate_code(req: MCPRequest):
    context = load_context(req.context_files)
    prompt_template = build_prompt_template()
    runnable = prompt_template | llm

    response = runnable.invoke({
        "context": context,
        "task": req.prompt
    })

    if not response:
        raise HTTPException(status_code=500, detail="LLM generation failed")

    push_code(req.service_name, response)
    return {"message": "Service generated and pushed to Bitbucket successfully."}
