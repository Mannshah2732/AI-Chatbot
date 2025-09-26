from dotenv import load_dotenv
import os
from langchain_openai import OpenAIEmbeddings
 
from langchain_core.output_parsers import StrOutputParser
 
from langchain_core.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    MessagesPlaceholder,
)
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.tools import tool, Tool
from langchain.agents import (
    AgentExecutor,
)
from langchain.tools.retriever import create_retriever_tool
 
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from pinecone import Pinecone as PineconeClient
from langchain_community.vectorstores import Pinecone
from langchain_openai import AzureChatOpenAI,ChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
from langchain_community.tools import QuerySQLDatabaseTool
 
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

 
openai_embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
  
)

output_parser = StrOutputParser()
llm = ChatOpenAI(model="gpt-4.1")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
pinecone_index_name = "robeck-dental"
pinecone = PineconeClient(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
index = pinecone.Index(pinecone_index_name)
namespace = "robeck_dental"

data = []

def namespace_exists(index, namespace):
    namespaces = index.describe_index_stats()["namespaces"]
    return namespace in namespaces


if namespace_exists(index, namespace):
    pinecone_vectorstore = Pinecone.from_existing_index(
        embedding=openai_embeddings,
        index_name="robeck-dental",
        namespace="robeck_dental",
    )

retriever = pinecone_vectorstore.as_retriever()

# print("after retriever")
message_history = ChatMessageHistory(session_id="robeck_dental")
memory = ConversationBufferMemory(memory_key="history", return_messages=True)
 
 
def create_agent():
    retriever_tool = create_retriever_tool(
        retriever,
        "Documents_retrieval",
        "Query a retriever to get information about Robeck Dental Clinic  overview, services, information etc. Use the information present in retriever only to answer user's question otherwise say I am not sure of that.",
    )
    tools = [retriever_tool]

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                Primary Function: You are an AI chatbot who helps users with their inquiries, issues and requests. 
                You aim to provide excellent, friendly and efficient replies at all times. 
                Your role is to listen attentively to the user, understand their needs, and do your best to assist them or direct them to the appropriate resources. 
                If a question is not clear, ask clarifying questions. 
                Make sure to end your replies with a positive note. Instead of saying Robeck Dental provides this say that We at robek dental provides this.
                Use retriever tool to get information about the clinic
                - Conversational Tone:
 
                1. Always respond in a friendly, conversational manner, guiding the customer through their process. Always say We offer this rather than saying Lekise offer this.
                Keep your responses concise and clear, focusing on assisting the customer efficiently without unnecessary preambles. Always, first try to understand user's needs and then only provide recommendation of our product.Do not include any additional text, explanation, or multiple JSON blocks.
                Make sure there are *no extra characters, no newlines before or after, and no extra JSON objects*.

                Lead-capture policy:
                • Always answer the question first. Do NOT gate answers behind a form.
                • Do not ask for contact info in the first two user turns.
                • After two helpful answers OR when the user shows high intent (booking, price, insurance, emergency, callback),
                politely ask if we may contact them and request name, phone and email .
                • If the user consents AND provides contact details,  then store it in memory and return in json as :
                "type: : "leads", "text" : <your answer>, "leads" : <list of email, name, number>

                Rules:
                - when there is basic question answer then return in json as:
                "type" : "text", "text" : <your answer>
                - when there's leads give response in json as :
                "type: : "leads", "text" : <your answer>, "leads" : <list of email, name, number>
 
                    
                """,
            ),
            MessagesPlaceholder(variable_name="history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    # print("before agent")
    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_to_openai_tool_messages(
                x["intermediate_steps"]
            ),
            "history": lambda x: x["history"],
        }
        | prompt
        | llm
        | OpenAIToolsAgentOutputParser()
    )
    # print("before agent executor")
    agent_executor = AgentExecutor(
        agent=agent ,tools=tools, verbose=True, memory=memory, max_iterations=5
    )
    # print("after")
    agent_with_chat_history = RunnableWithMessageHistory(
        agent_executor,
        lambda session_id: message_history,
        input_messages_key="input",
        history_messages_key="history",
    )
    # print("before return")
    return agent_with_chat_history
