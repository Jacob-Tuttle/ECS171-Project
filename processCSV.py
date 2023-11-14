import csv
import flask
import io
import pandas as pd

app = flask.Flask(__name__)

@app.route("/upload", methods=["PUT"])
def upload():
  csv_string = io.StringIO(flask.request.files["csvfile"].read().decode("utf-8"))

  # Read the CSV string into a Pandas DataFrame
  df = pd.read_csv(csv_string)

  # Manipulate the DataFrame with Python

  # Display the DataFrame in the HTML output
  output = df.to_html()

  return flask.render_template("index.html", output=output)

if __name__ == "__main__":
  app.run(debug=True)
