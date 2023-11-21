# from flask import Flask, render_template
# app = Flask(__name__)

# @app.route("/")
# def w209():
#     file="about9.jpg"
#     return render_template("w209.html",file=file)

# if __name__ == "__main__":
#     app.run()


from flask import Flask, render_template
# from flask_bootstrap import Bootstrap
import altair as alt
import pandas as pd
import numpy as np

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('base.html')




if __name__ == '__main__':
    app.jinja_env.auto_reload = True
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(debug=True)
