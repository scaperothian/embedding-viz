from flask import Flask, render_template, jsonify

app = Flask(__name__)

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

if __name__ == '__main__':
    app.jinja_env.auto_reload = True
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(debug=True)
