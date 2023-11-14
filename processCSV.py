import csv
import pandas as pd
import flask

app = flask.Flask(__name__)

@app.route("/upload", methods=["PUT"])
def upload():
  csv_file = flask.request.files["csvfile"]
  csv_filename = csv_file.filename

  # Save the CSV file to the server
  csv_file.save(csv_filename)

  # Read the CSV file into a Pandas DataFrame
  df = pd.read_csv(csv_filename)

  # Manipulate the DataFrame with Python

  # Print the DataFrame to the home page
  output = df.to_html()

  return flask.render_template("index.html", output=output)

if __name__ == "__main__":
  app.run(debug=True)
