import time
import tqdm
import pickle
import os
import pprint
import subprocess
import requests
import sys

import tensorflow as tf
import pandas as pd
import numpy as np

from transformers import AutoTokenizer, TFAutoModel

def download(id,filename):
    """
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
            self._default_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/multi-qa-MiniLM-L6-cos-v1")
            self._default_model = TFAutoModel.from_pretrained("sentence-transformers/multi-qa-MiniLM-L6-cos-v1")
            self.encoder = self._default_encoder
        else:
            self.encoder = encoder
        
    def create_embeddings(self, df, data_key, label_key, block_size=1, filename=None):
        self.data_frame = df
        self.data_key = data_key
        self.label_key = label_key            
        self.block_size = block_size

        self._saveEmbeddingChunks()
        self._combineEmbeddingChunks()
        self._saveCombinedEmbedding(filename=filename)
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
    
    def compare(self,query,n=5,verbose=False):
        queryemb = self.encoder(query)
        
        dataemb = tf.constant(self.embedding_matrix,dtype="float32")
        #Compute dot score between query and all document embeddings
        self.scores = (queryemb @ tf.transpose(dataemb))[0].numpy().tolist()
    
    def get_top_n_scores(self,n=5,verbose=False):

        df = self.data_frame
            
        sorted_indices = np.argsort(self.scores)[::-1]
        lst = []
        
        for i in sorted_indices[:n]:
            label = self.embedding_labels[i]
            score = self.scores[i]
            data = df[df[self.label_key]==self.embedding_labels[i]][self.data_key].iloc[0]
            if verbose:
                print(f"{label}: {score:.2f}")
                print(data)
                print('')
                print('')
                
            lst.append({
                "label": label,
                "score": score,
                "data": data 
            })
        return lst

    #Mean Pooling - Take attention mask into account for correct averaging
    def _mean_pooling(self,model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = tf.cast(tf.tile(tf.expand_dims(attention_mask, -1), [1, 1, token_embeddings.shape[-1]]), tf.float32)
        return tf.math.reduce_sum(token_embeddings * input_mask_expanded, 1) / tf.math.maximum(tf.math.reduce_sum(input_mask_expanded, 1), 1e-9)
        
    #Default Encoder for text
    def _default_encoder(self,texts):

        # Tokenize sentences
        encoded_input = self._default_tokenizer(texts, padding=True, truncation=True, return_tensors='tf')
        
        # Compute token embeddings
        model_output = self._default_model(**encoded_input, return_dict=True)
        
        # Perform pooling
        embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
        
        # Normalize embeddings
        embeddings = tf.math.l2_normalize(embeddings, axis=1)
        
        return embeddings
    
    def _saveEmbeddingChunks(self):
        """
        """
        
        # Iterate over consecutive blocks of rows
        num_rows = len(self.data_frame)
        start_index = 0
        block_counter = 0
        est_total_iterations = num_rows // self.block_size
        # Initialize tqdm with the total number of iterations
        with tqdm.tqdm(total=est_total_iterations, desc="Saving Embeddings in Chunks") as pbar:
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


def example_main(example_query):
    filename = '../../metadata_with_episode_dates_and_category.tsv'
    CROP_SHOWS = 100 # for testing
    block_size = 30  # see benchmarking
    file = 'example_embeddings.pkl'
    data_key='show_description'
    label_key='show_name'
    
    try: 
        df = pd.read_csv(filename,sep='\t')
    except Exception as e1:
        #print(e)
        try: 
            print("Attempt to pull from Google Drive")
            id = "1guz7ILFUGLN2aYoXUez-K5ZCw0Xjo6m4"
            download(id,filename)
            df = pd.read_csv(filename,sep='\t')
        except Exception as e2:
            print("Fetch to Google Drive failed to download file.")
            print(e1, e2)
            exit()

    # clean the data
    df['release_date'] = pd.to_datetime(df['release_date'], format='%Y-%m-%d').reset_index(drop=True)
    df = df[~df['release_date'].isna()]
    df = df[~df['category'].isna()]
    df = df[~df['show_description'].isna()]
    df = df[~df['show_name'].isna()]
    df = df[~df['episode_description'].isna()]
    df = df[~df['episode_name'].isna()]

    df_shows = df.drop_duplicates(['show_name','show_description'])[['show_name','show_description']].reset_index(drop=True)
    df_example = df_shows.sample(CROP_SHOWS)


    # delete file before generating things to make the example valid (i.e. no stale data somewhere)
    try:
        # Check if the file exists
        if os.path.exists(file):
            # Remove the file
            os.remove(file)
            print(f"File {file} deleted successfully.")
        else:
            print(f"File {file} does not exist. Continuing on my merry way.")
    except Exception as e:
        print(f"Error deleting file {file}: {e}")

    
    print("**************************************")
    print("  Generate embeddings and then        ")
    print("  Perform a comparison with a random  ")
    print("  query.                              ")
    print("**************************************")
    start = time.time()
    emb = TextEmbeddings()
    emb.create_embeddings( df_example,
                        data_key=data_key, 
                        label_key=label_key, 
                        block_size=block_size, 
                        filename=file)
    end = time.time()
    print(f"{end-start} to Generate Embeddings.")
    emb.compare(example_query)
    print(f"Query: {example_query}")
    print(f"Top Five Results:")
    pprint.pprint(emb.get_top_n_scores(n=5))
    
    
    print("**************************************")
    print("  Now Try the same query comparison   ")
    print("  Except pull the embeddings from     ")
    print("  saved file, instead in memory data. ")
    print("**************************************")
    emb2 = TextEmbeddings()
    emb2.load(file,df_example)
    emb2.compare(example_query)
    print(f"Query: {example_query}")
    print(f"Top Five Results:")
    pprint.pprint(emb2.get_top_n_scores(n=5))

if __name__ == "__main__":
    # TODO: add the following arguments to a parseargs: 
    #       1. embedding location (if loading)
    #       2. data set location (if generating)
    #       3. flag to load data
    #       4. flag to generate embeddings 
    #
    #  Create Embeddings
    #  python embeddings.py --generate --dataset <filename> --datakey <key> --labelkey <key>
    #  
    #  Load embeddings from file and include dataset information
    #  python embeddings.py --load <filename> --dataset <filename> --datakey <key> --labelkey <key>
    #  
    #  Load embeddings from file and query 
    #  python embeddings.py --load <filename> --dataset <filename> --datakey <key> --labelkey <key> --query <some_string>
    example_main("We are all in the matrix, so just swallow the pill will you?")
    
    
    
