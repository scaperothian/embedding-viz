from flask import Flask, render_template, jsonify
from os import path
import json
import pickle 
import pandas as pd

app = Flask(__name__)
root_path = path.dirname(__file__)

from embeddings import TextEmbeddings,download

file = "podcast_show_description_embeddings.20231126.pkl"
filename = 'metadata_with_episode_dates_and_category.tsv'
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

df_shows = df.drop_duplicates(['show_name','show_description'])[['show_name','show_description']].reset_index(drop=True)



emb = TextEmbeddings()
emb.load(file, df_shows)


@app.route('/')
def index():
    return render_template('base.html')

@app.route('/get_generator_text')
def get_generator_text():
    """
    read in some data from the file system.
    """

    with open('lorem.txt', 'r') as file:
        lorem_text = file.read()
    return jsonify({'generator_text': lorem_text})

@app.route('/get_top_artists')
def get_top_artists():
    """
    read in some data from the file system.
    """
    rsp = get_generator_text()
    query = rsp.json['generator_text']
    emb.compare(query)
    import pprint
    val = emb.get_top_n_scores(n=5)

    try:
        with open(root_path + "/top_artists.json") as fp:
            artists_data = json.load(fp)
        return jsonify(artists_data)
     
    except FileNotFoundError:
        return jsonify({'error': 'File not found'}), 404
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_top_songs')
def get_top_songs():
    """
    read in some data from the file system.
    """
    try:
        with open(root_path + "/top_artist_songs.json") as fp:
            song_data = json.load(fp)
        return jsonify(song_data)
    
    except FileNotFoundError:
        return jsonify({'error': 'File not found'}), 404
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.jinja_env.auto_reload = True
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(debug=True)
