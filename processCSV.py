import csv
import flask

app = flask.Flask(__name__)

@app.route("/upload", methods=["POST"])
def upload():
  csvfile = flask.request.files["csvfile"]

  # Save the CSV file to the server
  csvfile.save("upload.csv")

  # Read the CSV file into a Pandas DataFrame
  df = pd.read_csv("upload.csv")

  # Manipulate the DataFrame with Python

  # Display the DataFrame in the HTML output
  output = df.to_html()

  return flask.render_template("index.html", output=output)

if __name__ == "__main__":
  app.run(debug=True)
