import os

from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, PropertyGraphIndex
from typing import Literal
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.legacy.embeddings import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI

load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
NEO4J_URI = os.environ.get('NEO4J_URI')
NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")


def return_extractor():
    # List the entitied you want to extract
    entities = Literal["PERSON", "SKILL", "EMPLOYER", "ADDRESS", "PROFILE", "PHONE"]
    # List the relations
    relations = Literal["HAS", "PART_OF", "EMPLOYED", "BELONGS_TO", "WORKED_FOR", "WORKED_FROM", "WORKED_TILL"]

    # Define the schema
    validation_schema = {
        "PERSON": ["HAS", "PART_OF", "WORKED_FOR", "WORKED_FROM", "WORKED_TILL"],
        "ADDRESS": ["BELONGS_TO"],
        "PROFILE": ["HAS"],
        "PHONE": ["BELONGS_TO"],
        "SKILL": ["HAS", "BELONGS_TO"],
        "EMPLOYER": ["EMPLOYED"],
    }
    kg_extractor = SchemaLLMPathExtractor(
        llm=OpenAI(model="gpt-4o"),
        possible_entities=entities,
        possible_relations=relations,
        kg_validation_schema=validation_schema,
        strict=True,
    )

    return kg_extractor


def return_graph_strore():
    graph_store = Neo4jPropertyGraphStore(
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        url=NEO4J_URI,
    )

    return graph_store


if __name__ == "__main__":
    documents = SimpleDirectoryReader("/Users/joyeed/langmodel/langmodel/data").load_data()
    vec_store = None
    kg_extractor = return_extractor()
    graph_store = return_graph_strore()
    index = PropertyGraphIndex.from_documents(
        documents,
        kg_extractors=[kg_extractor],
        embed_model=HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5"),
        property_graph_store=graph_store,
        vector_store=vec_store,
        show_progress=True,
    )
