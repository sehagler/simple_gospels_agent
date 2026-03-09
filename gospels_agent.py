# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 09:58:34 2026

@author: sehag
"""

#
import os
import re
import sys

#
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = "..."
os.environ['USER_AGENT'] = 'gospels_agent'

#
path = os.path.dirname(__file__)
sys.path.insert(0, path + '/lib')

#
def  cleanup_data(documents):
    for i in range(len(documents)):
        text = documents[i].page_content
        text = re.sub(r'\n', ' ', text)
        text = re.sub(r' +', ' ', text)
        text = re.sub(r'\d+\:\d\d\d\:\d\d\d ', '', text)
        documents[i].page_content = text
    return documents

#
def main():
    
    #
    from agent import Agent_object
    from query import Query_object
    from vector_store import Vector_store_object
    
    #
    raw_data_paths = [ "data/matthew/matthew_text.txt",
                       "data/mark/mark_text.txt",
                       "data/luke/luke_text.txt",
                       "data/john/john_text.txt" ]
    
    #
    chunk_size=1000         # chunk size (characters)
    chunk_overlap=200       # chunk overlap (characters)
    add_start_index=True    # track index in original document
    
    #
    agent_object = Agent_object()
    agent_object.create_model()
    
    #
    query_object = Query_object()
    
    #
    vector_store_object = Vector_store_object()
    vector_store_object.load_raw_data(raw_data_paths)
    vector_store_object.cleanup_data(cleanup_data)
    vector_store_object.split_data(chunk_size, chunk_overlap, add_start_index)
    vector_store_object.create_vector_store()
    
    #
    if True:
        prompt = (
            "You have access to a tool that retrieves context from the Gospels. "
            "Answer queries using information available in the tool."
            )
        query_object.query_loop_prompt(agent_object, vector_store_object, prompt)
    else:
        prompt = (
            "You are an assistant that answers question from the Gospels. "
            "Use the following context from the Gospels in your response:"
            )
        query_object.query_loop_dynamic_prompt(agent_object, vector_store_object, 
                                               prompt)
        
#
if __name__ == "__main__":
    main()