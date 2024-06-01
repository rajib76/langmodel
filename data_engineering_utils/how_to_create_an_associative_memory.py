import os

from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_groq import ChatGroq
from langchain_openai import OpenAI

load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
NEO4J_URI = os.environ.get('NEO4J_URI')
NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

graph = Neo4jGraph(url=NEO4J_URI,
                   username=NEO4J_USERNAME,
                   password=NEO4J_PASSWORD)

prompt_template = "Answer the {question}"
prompt = PromptTemplate(
    input_variables=["question"], template=prompt_template
)
llm = OpenAI()
chain = prompt | llm

# llm = ChatOpenAI(temperature=0, model_name="gpt-4")


transformer_llm = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768")
llm_transformer = LLMGraphTransformer(llm=transformer_llm)

while True:
    question = input("Ask your question:\n")
    response = chain.invoke(question)
    print(response)
    text=f"""
    question:{question}
    answer:{response}
    """
    documents = [Document(page_content=text)]
    graph_documents = llm_transformer.convert_to_graph_documents(documents)

    print(f"Nodes:{graph_documents[0].nodes}")
    print(f"Relationships:{graph_documents[0].relationships}")

    graph.add_graph_documents(
        graph_documents,
        baseEntityLabel=True,
        include_source=True
    )
