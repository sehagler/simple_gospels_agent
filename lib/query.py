# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 11:42:53 2026

@author: sehag
"""

#
class Query_object(object):

    #
    def _query_loop(self, func, vector_store_object, prompt):
        
        #
        vector_store = vector_store_object.get_vector_store()
        
        #
        stop_flg = False
        while not stop_flg:
            query = input('Enter your query [(Q/q) to exit]: ')
            if query not in ['Q', 'q']:
                func(vector_store, prompt, query)
            else:
                stop_flg = True
                
    #
    def query_loop_dynamic_prompt(self, agent_object, vector_store_object, prompt):
        self._query_loop(agent_object.dynamic_prompt_agent,
                         vector_store_object, prompt)
                
    #
    def query_loop_prompt(self, agent_object, vector_store_object, prompt):
        self._query_loop(agent_object.prompt_agent, vector_store_object, prompt)