import os
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain_openai import OpenAI

# Load the OpenAI API key from the .env file
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize the OpenAI llm model with the API key
llm = OpenAI(openai_api_key=openai_api_key)

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
# - https://medium.com/@dash.ps/build-chatbot-with-llms-and-langchain-9cf610a156ff
# - https://python.langchain.com/docs/integrations/llms/amazon_api_gateway/#agent
def openai_math_agent(input: str) -> str:
    # Load the tools and initialize the agent
    tools = load_tools(["llm-math"], llm=llm)
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

    # Invoke the agent with the input
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
    math_problem = "If my age is half of my dad's age and he is going to be 60 next year, what is my current age?"
    math_response = openai_math_agent(math_problem)
    print(math_response)
