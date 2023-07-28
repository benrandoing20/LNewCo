# Created to label Functional Squat Data for Threshold Creation as well as
# ML pipeline approaches
# Created by Ben Randoing during 07/2023

import csv
import os
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

VID_FOLDER = 'static/'

HEADER = ['Filename Side', 'Filename Front', 'Squat Score',
          'Knee Stability', 'Core Strength', 'Asymmetry Score', 'Description']

# TODO: make user input filename, and check that is not already a filename
CSV_FILE = "data.csv"
START_INDEX = 0

# Get a list of all files and directories in the folder
all_items = os.listdir(VID_FOLDER)
all_items = sorted(all_items)

# Filter out only the filenames (excluding directories)
file_list = [os.path.join(
    VID_FOLDER, item) for item in all_items if os.path.isfile(os.path.join(
    VID_FOLDER, item))]

# Create tuples of two filenames (side view, front view)
FILENAMES = [(file_list[i], file_list[i + 1]) for i in range(0,
                                                          len(file_list), 2)]

FILES = len(FILENAMES)

with open(CSV_FILE, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(HEADER)

@app.route('/')
def home():
    return render_template('home.html', start=START_INDEX)

@app.route('/start/<int:index>', methods=['POST'])
def start(index):
    return redirect(url_for('index', index=index))

@app.route('/<int:index>')
def index(index):
    return render_template('index.html', video_url=FILENAMES[index][0],
                           video_url_front=FILENAMES[index][1], index=index)

@app.route('/submit/<int:index>', methods=['POST'])
def submit(index):
    squat_score = request.form['score']
    knee_score = request.form['knee']
    core_score = request.form['core']
    asym_score = request.form['asym']

    description = request.form['description']

    with open(CSV_FILE, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([FILENAMES[index][0], FILENAMES[index][1], squat_score,
                             knee_score, core_score, asym_score, description])
    if index >= FILES - 1:
        return redirect(url_for('end'))
    else:
        return redirect(url_for('index', index=index + 1)) # File Increment

@app.route('/end')
def end():
    return render_template('end.html')

if __name__ == '__main__':
    app.run(debug=True)
