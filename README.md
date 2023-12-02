# A Simple Text Embedding Visualization
Flask Web Page to perform simple query and comparison with text embeddings.

## Create Conda Environment using environment.yml<br>
```conda env create -f environment.yml```

## Start environment<br>
```conda activate flask-embeddings```

## Add deps<br>
```pip install -r requirements.txt```<br><br>
Note: there is an issue with installing tensorflow using this method on mac.  if you can perform the operation "import tensorflow" in the python cli, the application *should* work.

## Run flask
```flask --app main.py run```

