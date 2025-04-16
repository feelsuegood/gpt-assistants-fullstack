1. create pinecone account

2. https://platform.openai.com/ create account and register credit card

   - set usage limit for credit card

3. setup

   - git init . -> .gitignore
   - **python3.11** -m venv env (create virtual environment)
   - source env/bin/activate (get into Virtual environment) ↔︎ deactivate
   - pip install --upgrade pip setuptools wheel (Upgrade the package installation tool)
   - pip install -r requirements.txt
   - .env → put into .gitignore

4. delete virtual environment

   - deactivate
   - rm -rf env

5. Model: Orca Mini 3B (orca-mini-3b-gguf2-q4_0.gguf)

   - Developer: Microsoft
   - License: CC BY-NC-SA 4.0 (https://spdx.org/licenses/CC-BY-NC-SA-4.0)
   - Usage: This model is used for non-commercial research and experimentation purposes only, with attribution to Microsoft as the original creator.

6. Code Challenges

   - [x] DocumentGPT: apply memory by using session state
   - [x] PrivateGPT: provide multiple model and let a user choose one.

7. Reference

- https://docs.streamlit.io/develop/api-reference
