import time
import tqdm
import pickle
import os
import pprint
import subprocess
import requests
import sys
import argparse
import yaml

import pandas as pd
import numpy as np

import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

def download(id,filename):
    """
    Download pkl file from Google Drive.  Nice to use for getting models 
    or embeddings.
    
    the file we are pulling is too big, so Google requires you to accept 
    the conditions of a download warning where scanning is not possible.
    """
    URL = "https://docs.google.com/uc?export=download&confirm=1"

    session = requests.Session()

    response = session.get(URL, params={"id": id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {"id": id, "confirm": token}
        response = session.get(URL, params=params, stream=True)

    CHUNK_SIZE = 32768
    with open(filename, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value

    return None


class TextEmbeddings:
    """
    class to support creating and saving, loading, and comparing embeddings.
    
    1. creating - will create embeddings from a Pandas Data Frame and save off the embeddings in a PKL file along with metadata related to the embeddings
    2. loading - helper classes to support loading directly from PKL file (i.e. so you don't have to generate again.
    3. comparing - allows you to compare (i.e. cosine similarity) a string (i.e. query) with the embedding_matrix which needs to be created / loaded prior to calling.
    """
    def __init__(self,encoder=None):
        self.files = []
        self.embedding_matrix = []
        self.embedding_labels = []
        self.data_frame = None
        self.scores = None
        
        if not encoder:
            # Load model from HuggingFace Hub
            self._default_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L12-v2')
            self._default_model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L12-v2')
            self.encoder = self._default_encoder
        else:
            self.encoder = encoder
        
    def create_embeddings(self, input_filename, block_size=10, print_label="Artist"):

        # assumes a yaml file input
        output_filename = input_filename[:-5] + "_embeddings.pkl"
        
        # To Create Embeddings, you need data and you need create a dataframe of the data you want to parse....
        # Load the YAML file
        with open(input_filename, "r") as yaml_file:
            lyrics = yaml.safe_load(yaml_file) 

        df_input = pd.DataFrame(lyrics).reset_index()
        label_key = "songs"
        data_key = "song_lyrics"
        df_input.columns = ['songs','song_lyrics']
        
        self.data_frame = df_input
        self.data_key = data_key
        self.label_key = label_key            
        self.block_size = block_size

        self._saveEmbeddingChunks(print_label=print_label)
        self._combineEmbeddingChunks()
        self._saveCombinedEmbedding(filename=output_filename)
        self._cleanup()

    def add_data_frame(self, df):
        self.data_frame = df
    
    def load(self,file,data_frame=None):
        """
        Loading embeddings from file.
        """
        with open(file, 'rb') as f:
            loaded_data = pickle.load(f)
        self.label_key = loaded_data['label_key']
        self.data_key = loaded_data['data_key']
        self.embedding_matrix = loaded_data['embeddings']
        self.embedding_labels = loaded_data['embedding_labels']

        if data_frame is not None:
            self.data_frame = data_frame
    
    def compare(self,query,n=5):
        queryemb = self.encoder(query)
        
        dataemb = torch.tensor(self.embedding_matrix)
        #Compute dot score between query and all document embeddings
        self.scores = torch.matmul(queryemb, dataemb.t()).tolist()[0]
        return self.scores
    
    def get_top_n_scores(self,n=5,verbose=False):

        if type(self.data_frame) != type(None): 
            df = self.data_frame
                
            sorted_indices = np.argsort(self.scores)[::-1]
            lst = []
            
            for i in sorted_indices[:n]:
                label = self.embedding_labels[i]
                score = self.scores[i]
                data = df[df[self.label_key]==self.embedding_labels[i]][self.data_key].iloc[0]
                if verbose:
                    print(f"{label}: {score:.2f}")
                    
                lst.append({
                    "label": label,
                    "score": score,
                    "data": data 
                })
        else:
            sorted_indices = np.argsort(self.scores)[::-1]
            lst = []
            
            for i in sorted_indices[:n]:
                label = self.embedding_labels[i]
                score = self.scores[i]
                if verbose:
                    print(f"{label}: {score:.2f}")
                    print('')
                    print('')
                    
                lst.append({
                    "label": label,
                    "score": score,
                    "data": '' 
                })
        return lst

    #Mean Pooling - Take attention mask into account for correct averaging
    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        
    #Default Encoder for text
    def _default_encoder(self,texts):

        # Tokenize sentences
        encoded_input = self._default_tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    
        # Compute token embeddings
        with torch.no_grad():
            model_output = self._default_model(**encoded_input)
    
        # Perform pooling
        embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
    
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
    
        return embeddings
    
    def _saveEmbeddingChunks(self, print_label):
        """
        """
        
        # Iterate over consecutive blocks of rows
        num_rows = len(self.data_frame)
        start_index = 0
        block_counter = 0
        est_total_iterations = num_rows // self.block_size
        # Initialize tqdm with the total number of iterations
        with tqdm.tqdm(total=est_total_iterations, desc=f"Saving {print_label} Embeddings in Chunks") as pbar:
        # Start your while loop
            while start_index < num_rows:
                end_index = start_index + self.block_size if start_index + self.block_size < num_rows else num_rows
                subset_df = self.data_frame.iloc[start_index:end_index]
            
                # Apply the encode function to the current block
                emb = self.encoder(list(subset_df[self.data_key]))
                emb_labels = list(subset_df[self.label_key])
                self._saveChunk(emb,emb_labels,block_counter)
                
                start_index = end_index
                block_counter += 1
                pbar.update(1)

    def _saveCombinedEmbedding(self,filename='final.pkl'):
        d = {
            "label_key":self.label_key,
            "data_key":self.data_key,
            "embedding_labels":self.embedding_labels,
            "embeddings": self.embedding_matrix
        }
            
        # Save to disk
        with open(filename, 'wb') as f:
            pickle.dump(d, f)
    
    def _saveChunk(self, embeddings, embedding_labels, block_num=0, num_padding=5, filename=None):
        """
        """
        d = {
            "block":block_num,
            "embedding_labels":embedding_labels,
            "embeddings": embeddings
        }
        block_num = d['block']
        if not filename: 
            filename = f"{block_num:0{num_padding}}_data.pkl"
            self.files.append(filename)
            
        # Save to disk
        with open(filename, 'wb') as f:
            pickle.dump(d, f)
    
    def _cleanup(self):
        # Explicit method for cleaning up resources
        for file in self.files:
            try:
                os.remove(file)
                #print(f"File {file} deleted successfully.")
            except Exception as e:
                print(f"Error deleting file {file}: {e}")
                
    def _combineEmbeddingChunks(self):
        """
        """
        # This code takes in a list of files, then loads them, 
        # extracts the embeddings and concatenates them with the other embeddings.
        list_of_embeddings = []
        list_of_labels = []
        for file in tqdm.tqdm(self.files,desc="Combining Embeddings"): 
            # Load from disk
            with open(file, 'rb') as f:
                loaded_data = pickle.load(f)
            list_of_embeddings.append(loaded_data['embeddings'])
            list_of_labels.extend(loaded_data['embedding_labels'])
        
        self.embedding_matrix = np.vstack(list_of_embeddings)
        self.embedding_labels = list_of_labels

if __name__ == "__main__":
    print('test')



    
    
    
