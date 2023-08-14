# LNewCo
#### Functional Physical Assessment of a Deep Squat Summer 2023

<figure>
    <img src="Example%20Images/temp_image.jpg"
         alt="Side View" width="300" 
height="500">
    <img src="Example%20Images/temp_image_front.jpg"
         alt="Front View" width="300" 
height="500">
    <figcaption>Base of Squat, Sagittal Plane (L) / Coronal Plane 
(R)</figcaption>
</figure>




## Table of Contents

[Project Introduction](#project-introduction)

[Screenshots](#screenshots)

[Setup / Installation](#setup--installation)

[Approach](#approach)

[Status](#status)

## Project Introduction:

During the summer of 2023, LNewCo was interested in growing and automating 
their functional movement assessment tests. The existing exam includes 
grip strength, VO2 max, and single leg stance tests. The trainer team expressed
interest in the functional movement assessment (FMS) to gain a deeper sense 
into client physical fitness while maintain a short test time. 

This Functional Deep Squat analysis pipeline supports LNewCo's physical 
fitness assessment. The primary function of this repository is to provide 
a personalized squat performance report that features user facing metrics 
as well as labels to support trainer workflows when designing intervention 
plans.

The repository also enables two secondary processes. First, EDA is enabled 
with the EDA.ipynb to support in the preliminary creation of heuristic 
scoring and classifications. The repository also features a data labelling 
flask app to enable trainers to label videos for heuristic and ML validation. 

## Output Report:

<figure>
    <img src="Example%20Images/profile.jpg" width="400" 
height="550"></img>
    <figcaption>1st page with Squat Scoring and Base of Squat 
Visual</figcaption>
</figure>
<br>
<figure>
    <img src="Example%20Images/profile_heatmap.jpg" width="400" 
height="550"></img>
    <figcaption>2nd page with Muscle Heat Map</figcaption>
</figure>
<br>
<figure>
    <img src="Example%20Images/profile_indicator.jpg" width="400" 
height="550"></img>
    <figcaption>3rd page with Trainer Insights</figcaption>
</figure>


## Technologies Used:
1. Google MediaPipe Pose Estimation Python Library
2. Flask Python Framework
3. Jupyter Notebooks

## Setup / Installation:

1. Install the GitHub repository to your local machine ```git clone <ssh to 
   git repo>```
2. Crate a folder entitled ```data``` in the project root directory
   1. Create a
3. Create a virtual environment with pipenv as follows: ```python3 -m venv 
   <VENV NAME>'```
4. Install project dependencies: ```pip3 install -r requirements.txt```

### Analyzing Squat Batches

### Performing EDA

### Collecting Video Labels



## Approach:

Pose estimation is leveraged to analyze an individual's squat. Using a 
front a side view of the squat, pose estimation is performed wth Google's 
mediapipe library in Python. The specific points are used with trigonometry 
to estimate joint angles (ie: knee flexion/extension) and angles
between various body segment (ie: angle between the torso and shank segment). 

Domain knowledge from the " " book and feedback from trainers are the 
inspiration for rule-based criteria for squat scoring. Additionally, 
exploratory data analysis was conducted to identify features and 
relationships indicative of good and bad squat performance. Accordingly, 
numerous features such as knee varus/valgus at the base point (deepest 
point) in a squat demonstrated a correlation between good and bad squat 
performance. Such features are used to create a heuristic scoring platform 
as well as squat labels to support trainers. 


<figure>
    <img src="Example%20Images/EDA%20kvv.png" width="400" 
height="300"></img>
    <figcaption>Example from EDA of Knee Varus/Valgus between Good
(Green) and Bad (Red) Squat Form</figcaption>
</figure>

## Status:
The current pipeline leverages heuristic based models that wer developed 
based on insights gathered in conversations with Trainers and through EDA. 

In order to develop a more robust model, ML and RL approaches should be 
leveraged to verify the scoring. Potential future wok may include 
collecting a dataset of labeled squat data from trainers as the basis for 
a supervised learning model is combined with the current heuristic. 
Reinforcement Learning approaches may include creating a reward model based 
on trainer feedback to refine the recommendations over time.

## MIT License

Copyright 2023 Ben Randoing

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
