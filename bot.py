import aiohttp
import os
import yaml
from langchain.chains import LLMChain, LLMMathChain, ConversationChain
from langchain_core.prompts import PromptTemplate
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain_openai import ChatOpenAI
from langchain.memory import (
  ConversationBufferMemory, 
  ReadOnlySharedMemory, 
  ConversationSummaryMemory, 
  ConversationBufferWindowMemory, 
  ConversationSummaryBufferMemory, 
  ConversationEntityMemory,
  ReadOnlySharedMemory
)
from langchain.output_parsers import PydanticOutputParser
from langchain.chains.conversation.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.agents.conversational_chat.base import ConversationalChatAgent
from langchain.agents.agent import AgentExecutor
from langchain.prompts.prompt import PromptTemplate
from langchain.agents import ZeroShotAgent
from langchain.agents.agent_toolkits.openapi import planner
from langchain.agents.tools import Tool
from dotenv import load_dotenv
from pathlib import Path
from pprint import pprint
from pydantic import BaseModel, Field
from typing import Any, Callable, Dict, List, Optional

# Load the OpenAI API key from the .env file
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# API Requests wrapper for making HTTP requests
class RequestsWrapper:
    def __init__(self, headers: Dict[str, str] = None):
        self.headers = headers or {}

    async def post(self, url: str, data: Any, headers: Dict[str, str] = None) -> Any:
        async with aiohttp.ClientSession(headers={**self.headers, **(headers or {})}) as session:
            async with session.post(url, json=data) as response:
                return await response.json()

    # Note: Add more methods for GET, PUT, DELETE as needed

# API Planner tool for analyzing API spec and planning endpoint calls
class APIPlannerTool(Tool):
    name = "API Planner"
    description = "Analyzes API spec to determine endpoints for querying."

    def __init__(self, api_spec, llm):
        self.api_spec = api_spec
        self.llm = llm

    def _run(self, query, **kwargs):
        # TODO: Add logic to analyze API spec and plan endpoint calls based on the query
        pass

# API Controller tool for making API calls
class APIControllerTool(Tool):
    name = "API Controller"
    description = "Manages API calls to fetch and process data."

    def __init__(self, api_spec, requests_wrapper, llm):
        self.api_spec = api_spec
        self.requests_wrapper = requests_wrapper
        self.llm = llm

    def _run(self, query, **kwargs):
        # TODO: Add logic to make API calls using requests_wrapper and process data
        pass

# Calculator input schema for math questions
class CalculatorInput(BaseModel):
    question: str = Field()

# Main ChatBot class
class ChatBot:
    # TODO: Setup API with flask
    # ref: https://flask.palletsprojects.com/en/3.0.x/
    api_url = ""
    login_access_token = ""
    with open("part_doc.yml") as f:
        api_data = yaml.load(f, Loader=yaml.Loader)

    def __init__(self, email, password):
        self.email =  email
        self.password = password

    async def login(self):
        login_data = {
            "email": self.email,
            "password": self.password
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(self.api_url+"/auth/token", data=login_data) as response:
                response_data =  await response.json()
                self.login_access_token = f'Bearer {response_data["access"]}'

    def _handle_error(error) -> str:
        return str(error)[:50]

    def _reduce_openapi_spec(api_spec: dict) -> dict:
        """
        Simplifies the OpenAPI spec to a format that is easier to work with for the chatbot.

        Args:
            api_spec (dict): The full OpenAPI specification as a Python dictionary.

        Returns:
            dict: A simplified version of the OpenAPI spec.
        """

        reduced_spec = {
            "endpoints": [],
            # Extract other necessary details from the API spec as needed.
        }

        for path, operations in api_spec.get("paths", {}).items():
            for operation in operations.values():
                reduced_spec["endpoints"].append({
                    "path": path,
                    "description": operation.get("summary", "No description available"),
                    # Include other relevant details here.
                })

        return reduced_spec

    def _create_openapi_agent(api_spec, requests_wrapper, llm, handle_parsing_errors, shared_memory):
        """
        Creates an agent capable of interpreting and interacting with an API based on the OpenAPI spec.

        Args:
            api_spec (dict): Simplified OpenAPI spec.
            requests_wrapper (RequestsWrapper): Utility for making HTTP requests.
            llm (BaseLanguageModel): The language model for generating query prompts or interpreting API responses.
            handle_parsing_errors (Callable): Function to handle errors during parsing.
            shared_memory (ReadOnlySharedMemory): Memory object for sharing state.

        Returns:
            A function that takes a query and returns a response based on API interaction.
        """
        def agent_run(input: str) -> str:
            # TODO: Add logic to interpret the input, decide which API endpoint(s) to hit,
            # and how to formulate the request(s) based on the api_spec.
            # This will likely involve calling the requests_wrapper with specific endpoints.
            print("API Agent received input:", input)
            return "Response based on API interaction"

        return Tool(
            name="openapi_agent",
            func=agent_run,
            description="Interacts with the API based on OpenAPI spec",
        )

    def ask_api_questions(self, question):
        llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=0.0, model="gpt-4")
        openai_api_spec = self._reduce_openapi_spec(self.api_data)
        headers = {
            "Authorization": self.login_access_token,
            "Content-Type": "application/json"
        }
        requests_wrapper = RequestsWrapper(headers=headers)

        messages = [
            HumanMessage(content="Hey, I am Damir"),
            AIMessage(content="Hello Damir, how can I help you?"),
        ]
        tools=[]
        llm_math_chain = LLMMathChain(llm=llm, verbose=True)

        tools.append(
            Tool.from_function(
                func=llm_math_chain.run,
                name="Calculator",
                description="useful for when you need to answer questions about math",
                args_schema=CalculatorInput
                # coroutine= ... <- you can specify an async method if desired as well
            )
        )

        def _create_planner_tool(llm, shared_memory):

            def _create_planner_agent(question: str):
                agent = self._create_openapi_agent(
                    openai_api_spec, 
                    requests_wrapper, 
                    llm, 
                    handle_parsing_errors=self._handle_error,
                    shared_memory=shared_memory,
                )
                return agent.run(input=question)

            return Tool(
                name="api_planner_controller",
                func=_create_planner_agent,
                description="Can be used to execute a plan of API calls and adjust the API call to retrieve the correct data for Kickbite",
            )

        prefix = """
            You are an AI assistant developed by Damir.
        """

        suffix = """Begin!"

        {chat_history}
        Question: {input}
        {agent_scratchpad}"""

        prompt = ZeroShotAgent.create_prompt(
           tools, 
            prefix=prefix, 
            suffix=suffix, 
            input_variables=["input", "chat_history", "agent_scratchpad"]
        )

        chat_history = ConversationBufferMemory(messages=messages)
        window_memory = ConversationSummaryBufferMemory(llm=llm, chat_memory=chat_history, input_key="input", memory_key="chat_history")
        shared_memory = ReadOnlySharedMemory(memory=window_memory)
        tools.append(_create_planner_tool(llm, shared_memory))

        llm_chain = LLMChain(llm=llm, prompt=prompt, memory=window_memory)

        agent = ZeroShotAgent(
            llm_chain=llm_chain, 
            tools=tools, 
            verbose=True,     
            handle_parsing_errors="Check your output and make sure it conforms!",
            prompt=prompt
        )

        agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent, 
            tools=tools, 
            memory=window_memory
        )

        agent_executor.verbose = True

        output = agent_executor.run(input=question)
        print("LOL! 🦜🔗")
        pprint(output)
