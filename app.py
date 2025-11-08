from flask import Flask, send_from_directory, render_template_string
import os

app = Flask(__name__)

HTML = """
<!doctype html>
<title>face-embed-lab dashboard</title>
<h2>face-embed-lab (No ID / No Matching)</h2>
<ul>
  <li><a href="/img/outputs/pca_plot.png">PCA Plot</a></li>
  <li><a href="/img/outputs/tsne_plot.png">t-SNE Plot</a></li>
</ul>
<p>Veriler lokal tutulur. Bu arayüz yalnızca görselleştirme içindir.</p>
"""

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/img/outputs/<path:fn>")
def outputs(fn):
    return send_from_directory("data/outputs", fn)

if __name__ == "__main__":
    os.makedirs("data/outputs", exist_ok=True)
    app.run(host="127.0.0.1", port=5000, debug=False)
