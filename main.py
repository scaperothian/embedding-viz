from flask import Flask, render_template, request

app = Flask(__name__)

def find_similarity(text_input):
    return ['Hello there is a secret message going off the screen','world','dinosaur','good','morning','Vietnam']

@app.route("/", methods=["GET", "POST"])
def index():    
    if request.method == "POST":
        user_input = request.form["input_text"]
        similar_strings = find_similarity(user_input) # Replace with your actual function
        return render_template("index.html", similar_strings=similar_strings, user_input=user_input)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)