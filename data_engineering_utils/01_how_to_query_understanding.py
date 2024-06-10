import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""Given an original question, you will rephrase the question so that it can be better understood. 
    You must follow the below rule while rephrasing the question:
    
    # Rule:
    1. The performance of a product is determined by the gross revenue it earns
    2. Any performance trend related question need to consider last three years starting from 2024
    
    Here are few examples to follow:    
    
    # Examples
    Original question: "What are the most performing products?"
    Rephrased question: "Which products did the best with respect to gross revenue?"
    
    Original question: "What is the performance trend for product ABC?"
    Rephrased question: "What is the trend of gross revenue for product ABC from 2022 till 2024?"
    
    {question}
    
    rephrased_query:
    """
)


llm = ChatOpenAI(temperature=0)

llm_chain = QUERY_PROMPT | llm

# rephrased_query = llm_chain.invoke({"question":"What are the best selling products?"})
rephrased_query = llm_chain.invoke({"question":"What is the performance trend for product prd3"})

print(rephrased_query)