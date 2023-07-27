# Created to label Functional Squat Data for Threshold Creation as well as
# ML pipeline approaches
# Created by Ben Randoing during 07/2023

import csv
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

# TODO: Create Video File List and Iterate Through
VIDEO_FILE = "static/IMG_6783.mov"
VIDEO_FILE_FRONT = "static/IMG_6784.mov"
CSV_FILE = "data.csv"

@app.route('/')
def index():
    return render_template('index.html', video_url=VIDEO_FILE,
                           video_url_front=VIDEO_FILE_FRONT)

@app.route('/submit', methods=['POST'])
def submit():
    score = request.form['score']
    description = request.form['description']

    with open(CSV_FILE, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([score, description])

    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
