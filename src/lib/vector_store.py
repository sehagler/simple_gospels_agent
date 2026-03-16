# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 14:08:28 2026

@author: sehag
"""

#
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore

#
class Vector_store_object(object):
    
    #
    def __init__(self):
        self.data = None
        self.documents = None
        self.vector_store = None
    
    #
    def cleanup_data(self, cleanup_func):
        self.documents = cleanup_func(self.documents)
    
    #
    def create_vector_store(self):
        embeddings = \
            HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vector_store = InMemoryVectorStore(embeddings)
        self.vector_store.add_documents(documents=self.data)
    
    #
    def get_vector_store(self):
        return self.vector_store
    
    #
    def load_raw_data(self, raw_data_paths):
        for i in range(len(raw_data_paths)):
            loader = TextLoader(raw_data_paths[i])
            if self.documents == None:
                self.documents = loader.load()
            else:
                self.documents += loader.load()

    #
    def split_data(self, chunk_size, chunk_overlap, add_start_index):
        text_splitter = \
            RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                           chunk_overlap=chunk_overlap,
                                           add_start_index=add_start_index)
        self.data = text_splitter.split_documents(self.documents)