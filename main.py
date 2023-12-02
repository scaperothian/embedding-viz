from flask import Flask, render_template, jsonify
from os import path
import json
import pickle 
import pandas as pd
import logging

app = Flask(__name__)
root_path = path.dirname(__file__)

# Configure the Flask app logger
app.logger.setLevel(logging.DEBUG)
app.logger.addHandler(logging.StreamHandler())

from embeddings import TextEmbeddings,download

embeddings_file = "podcast_show_description_embeddings.20231126.pkl"
data_file = 'metadata_with_episode_dates_and_category.tsv'
data_key='show_description'
label_key='show_name'
    
try: 
    df = pd.read_csv(data_file,sep='\t')
except Exception as e1:
#print(e)
    try: 
        print("Attempt to pull Data File from Google Drive")
        id = "1guz7ILFUGLN2aYoXUez-K5ZCw0Xjo6m4"
        download(id,data_file)
        df = pd.read_csv(data_file,sep='\t')
    except Exception as e2:
        app.logger.debug("Fetch to Google Drive for Data File failed to download file.")
        app.logger.debug(e1, e2)
        exit()

# clean the data
df['release_date'] = pd.to_datetime(df['release_date'], format='%Y-%m-%d').reset_index(drop=True)
df = df[~df['release_date'].isna()]
df = df[~df['category'].isna()]
df = df[~df['show_description'].isna()]
df = df[~df['show_name'].isna()]

df_shows = df.drop_duplicates(['show_name','show_description'])[['show_name','show_description']].reset_index(drop=True)



emb = TextEmbeddings()
try: 
    emb.load(embeddings_file, df_shows)
except Exception as e1:
    try: 
        print("Attempt to pull Embeddings File from Google Drive")
        id = "1TNsMh9jvuTXU2xOWs0LE8FGJ-U7w9WDU"
        download(id,embeddings_file)
        emb.load(embeddings_file, df_shows)
    except Exception as e2:
        app.logger.debug("Fetch to Google Drive for Embeddings File failed to download file.")
        app.logger.debug(e1, e2)
        exit()


@app.route('/')
def index():
    return render_template('base.html')

@app.route('/get_input_text')
def get_input_text():
    """
    read in some data from the file system.
    """

    with open('input.txt', 'r') as file:
        input_text = file.read()
    return jsonify({'input_text': input_text})

@app.route('/get_top_scores')
def get_top_scores():
    """
    read in some data from the file system.
    """
    rsp = get_input_text()
    query = rsp.json['input_text']
    emb.compare(query)
    val = emb.get_top_n_scores(n=5)
    return jsonify(val)

if __name__ == '__main__':
    app.jinja_env.auto_reload = True
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(debug=True)
