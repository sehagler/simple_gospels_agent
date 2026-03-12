# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 14:56:36 2026

@author: sehag
"""

#
#from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest
#from langchain_classic.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool

#
class Agent_object(object):
    
    #
    def __init__(self):
        self.model = None
        
    #
    def _agent_output(self, agent, query):
        for step in agent.stream(
            {"messages": [{"role": "user", "content": query}]},
            stream_mode="values",
        ):
            step["messages"][-1].pretty_print()
           
    #
    def _prompt_with_context(self, vector_store, system_message_txt):
        @dynamic_prompt
        def prompt_with_context(request: ModelRequest) -> str:
            """Inject context into state messages."""
            last_query = request.state["messages"][-1].text
            retrieved_docs = vector_store.similarity_search(last_query)
            docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)
            system_message = (system_message_txt, f"\n\n{docs_content}")
            system_message = ''.join(system_message)
            return system_message
        return prompt_with_context
    
    #
    def _retrieve_context(self, vector_store, query):
        @tool(response_format="content_and_artifact")
        def retrieve_context(query: str):
            """Retrieve information to help answer a query."""
            retrieved_docs = vector_store.similarity_search(query, k=2)
            serialized = "\n\n".join(
                (f"Source: {doc.metadata}\nContent: {doc.page_content}")
                for doc in retrieved_docs
            )
            return serialized, retrieved_docs
        return retrieve_context
    
    #
    def create_model(self):
        self.model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    #
    def dynamic_prompt_agent(self, vector_store, system_message, query):
        prompt_with_context = self._prompt_with_context(vector_store,
                                                        system_message)
        agent = create_agent(self.model, tools=[], middleware=[prompt_with_context])
        self._agent_output(agent, query)
    
    #
    def prompt_agent(self, vector_store, prompt, query):
        retrieve_context = self._retrieve_context(vector_store, query)
        tools = [retrieve_context]
        agent = create_agent(self.model, tools, system_prompt=prompt)
        self._agent_output(agent, query)