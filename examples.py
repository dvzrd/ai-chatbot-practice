# imports for environment variables
import os
from dotenv import load_dotenv
# imports for core functionality
from langchain_openai import OpenAI
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
# imports for agents example
from langchain.agents import AgentType, initialize_agent, load_tools
# imports for memory example
from langchain.memory import ConversationBufferMemory
# imports for custom tools example
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from typing import Optional
from langchain.tools.base import BaseTool
from datetime import datetime

# Load the OpenAI API key from the .env file
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize the OpenAI llm model with the API key
llm = OpenAI(openai_api_key=openai_api_key)

# Define custom tools
# - Note: This tool is used to give GPT-3 context of the current date, since its information cuts off after 2021
class GetCurrentDate(BaseTool):
    name = "get_current_date"
    description = "Use this tool to get the current date, to calculate dates before or after the current date."
    
    def _run(
        self, query, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        return datetime.today().strftime('%Y-%m-%d')

    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")

# Chains example refs:
# - https://python.langchain.com/docs/integrations/llms/openai/
# - https://medium.com/@dash.ps/build-chatbot-with-llms-and-langchain-9cf610a156ff
def query_openai_with_prompt_template(template: str, input: str) -> str:
    """
    Queries the OpenAI model using a specified template and input string.

    Args:
        template (str): The template string to use for the prompt. Must contain '{input}' placeholder.
        input (str): The string to insert into the template placeholder.

    Returns:
        str: The response text from the model.
    """

    # Create a PromptTemplate instance with the provided template
    prompt = PromptTemplate(
        input_variables=["input"],  # This must match the placeholder in the template
        template=template
    )

    # Initialize the LLMChain with the prompt and model
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    # Format the input as a dictionary to match the expected input_variables
    formatted_input = {"input": input}

    # Use the LLMChain's invoke method with the formatted input
    response = llm_chain.invoke(formatted_input)

    return response

# Agents example refs:
# - https://python.langchain.com/docs/integrations/llms/amazon_api_gateway/#agent
def openai_math_agent(input: str) -> str:
    # Load the tools and initialize the agent
    tools = load_tools(["llm-math"], llm=llm)
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

    # Invoke the agent with the input
    response = agent.invoke(input)

    return response

# Memory example refs:
# - https://python.langchain.com/docs/modules/memory/
# - https://python.langchain.com/docs/expression_language/how_to/message_history/
def query_openai_with_memory(input: str, memory: ConversationBufferMemory) -> str:
    template = """You are a nice chatbot having a conversation with a human.
        Previous conversation: {chat_history}
        New human question: {question}
        Response:"""
    prompt = PromptTemplate.from_template(template)

    conversation = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=True,
        memory=memory
    )

    response = conversation({"question": input})

    return response

# Custom tools refs:
# - https://python.langchain.com/docs/modules/tools/custom_tools/
def query_openai_with_custom_tool(input: str) -> str:
    # Define custom tools to use in the agent
    tools = [GetCurrentDate()]
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )

    response = agent.invoke(input)

    return response

# Main function to run examples
if __name__ == "__main__":
    # Query OpenAI with a question and print the response
    question_template = "Question: {input} Answer: Let's think step by step."
    question_response = query_openai_with_prompt_template(question_template, "What NFL team won the Super Bowl in the year Justin Bieber was born?")
    print(question_response)

    # Query OpenAI with the city description and print the response
    city_description_template = "Describe the perfect day in {input}."
    city_description_response = query_openai_with_prompt_template(city_description_template, "Paris")
    print(city_description_response)

    # Use the OpenAI math agent to solve a math problem
    # Note: Still working on this example, the math agent needs additional tools to solve math word problems correctly
    # Ref: https://towardsdatascience.com/building-a-math-application-with-langchain-agents-23919d09a4d3
    # math_problem = "If my age is half of my dad's age and he is going to be 60 next year, what is my current age?"
    # math_response = openai_math_agent(math_problem)
    # print(math_response)

    # Initialize ConversationBufferMemory to maintain chat_history across queries
    # Note: Can also be defined as a global variable to maintain chat history across multiple queries, if needed
    chat_history = ConversationBufferMemory(memory_key="chat_history")
    # Query OpenAI with memory to maintain chat history
    memory_response_1 = query_openai_with_memory("What is the capital of France?", chat_history)
    print(memory_response_1)
    memory_response_2 = query_openai_with_memory("What is a popular landmark there?", chat_history)
    print(memory_response_2)
    memory_response_3 = query_openai_with_memory("Is it popular with tourists?", chat_history)
    print(memory_response_3)

    # Query OpenAI with custom tool
    custom_tool_response = query_openai_with_custom_tool("What is the current date?")
    print(custom_tool_response)
