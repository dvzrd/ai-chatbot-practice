# AI Chatbot

Simple AI Chatbot Example App built with Python, OpenAI and LangChain

## Dev Workspace Setup

Setup instructions for developers with little to no python experience.

For windows users, download latest release here: https://www.python.org/downloads/windows/

For mac and linux users, assuming you have homebrew installed:

1. Install Python

```zsh
brew install python
```

>**Note:** I was on `Python v3.12.2` at the time I was working on this.

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

## Running the Code

1. Source the virtual environment

```zsh
source venv/bin/activate
```

2. Run the examples

Run the the examples from the project's root.

```zsh
python examples.py
```

Or, if you haven't setup an alias for python:

```zsh
python3 examples.py
```

3. Run the bot

Same as the examples above, run the bot inside the virtual environment:

```zsh
python bot.py
```

## Resources and References

- [Primary article for this exercise](https://medium.com/@dash.ps/build-chatbot-with-llms-and-langchain-9cf610a156ff)
- [Math agent with custom tools](https://towardsdatascience.com/building-a-math-application-with-langchain-agents-23919d09a4d3)
- [OpenAI docs](https://platform.openai.com/docs/overview)
- [LangChain docs](https://python.langchain.com/docs/get_started/introduction)
