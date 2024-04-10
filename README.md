# AI Chatbot

Simple AI Chatbot Example App built with Python, OpenAI and LangChain

## Coding Exercise

All chat bots ultimately follow the same pattern. Therefore, we are not going to have a custom exercise, but instead refer to an external, already existing example.

Here we are combining two technologies who are going to be useful in the future. OpenAI’s API and LangChain, a known framework when dealing with different AI models.

Please go through the entire exercise and try to set up the chatbot in a way that you can input a query and the chatbot responds with an answer that is accurate and relevant.

Ref: https://medium.com/@dash.ps/build-chatbot-with-llms-and-langchain-9cf610a156ff

## Dev Workspace Setup

Setup instructions for frontend developers with little to no python experience.

This example assumes you have homebrew installed (for mac and linux users):

1. Install Python

```zsh
brew install python
```

2. Verify Installation

```zsh
python3 --version # Check python version
pip3 --version # Check pip (python's package manager) version
```

3. Setup Virtual Environment

```zsh
python3 -m venv venv
```

This will create a `venv` folder in your project's root dir. Once you've confirmed this, source you virtual environment:

```zsh
source venv/bin/activate
```

4. Install Required Libraries

```zsh
pip install openai langchain
```

You can also setup a `requirements.txt` file to keep track of all your required dependencies and versions. Refer to this article for more info: https://www.freecodecamp.org/news/python-requirementstxt-explained/

If you setup a `requirements.txt` file, you can install all your dependencies like this:

```zsh
pip install -r requirements.txt
```

5. Setup Envrionment Variables

You'll need an OpenAI API key mapped to an environment variable.

Create your key here: https://platform.openai.com/api-keys and add it to your `.env` file

```zsh
OPENAI_API_KEY=sk-your_api_key_goes_here
```

### Setup Python Alias

If you prefer to use the `python` command instead of `python3`, you'll need to setup an alias:

1. Open your `.zshrc`

```zsh
nano ~/.zshrc
```

2. Add your python alias

```zsh
alias python=python3
```

3. Source the changes

```zsh
source ~/.zshrc
```

## Running the Examples

1. Source the virtual environment

```zsh
source venv/bin/activate
```

2. Run the bot

Run the Python app from the project's root.

```zsh
python examples.py
```

Or, if you haven't setup an alias for python:

```zsh
python3 examples.py
```
