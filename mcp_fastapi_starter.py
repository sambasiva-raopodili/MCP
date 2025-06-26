# MCP Starter: FastAPI + LangChain + Bitbucket Integration

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import requests
import os
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from git import Repo
import tempfile

app = FastAPI()

# === Configuration === #
BITBUCKET_REPO_URL = "https://github.com/sambasiva-raopodili/myfirstrepo.git"
BITBUCKET_BRANCH = "feature/mcp-generated"
BITBUCKET_TOKEN = os.getenv("BITBUCKET_TOKEN")
MODEL_API_KEY = os.getenv("OPENAI_API_KEY")

llm = OpenAI(temperature=0.2, openai_api_key=MODEL_API_KEY)

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

def build_prompt(context: str, task: str) -> str:
    template = PromptTemplate(
        input_variables=["context", "task"],
        template="""
You are a backend developer for a fintech company. Below is the coding context:

{context}

Write business logic and corresponding unit test for the following requirement:

{task}
"""
    )
    return template.format(context=context, task=task)

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

# === Endpoint === #
@app.post("/generate")
def generate_code(req: MCPRequest):
    context = load_context(req.context_files)
    full_prompt = build_prompt(context, req.prompt)
    response = llm(full_prompt)

    if not response:
        raise HTTPException(status_code=500, detail="LLM generation failed")

    push_code(req.service_name, response)
    return {"message": "Service generated and pushed to Bitbucket successfully."}
