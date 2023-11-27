from flask import Flask, render_template, jsonify
from os import path
import json
import pickle 

app = Flask(__name__)
root_path = path.dirname(__file__)

import embeddings.TextEmbedding as TextEmbedding

file = "podcast_show_description_embeddings.pkl"
with open(file, 'rb') as f:
    showemb = pickle.load(f)

emb = TextEmbedding()
emb.load(showemb)


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
    query_emb = emb.encoder(query)
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
