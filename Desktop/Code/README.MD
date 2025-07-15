# Task
Task is to make a CCTV based identification of faces, store those faces in a database, and then use this database to re-identify a person the next time.

## Tools to be used
We will be using FlowCV, face-recognition packages.
FlowCV will be for capturing video, face-recognition is self-explanatory.
We will be using sqlite3 for database purposes.

## Issues encountered
<ul>
<li> Only old version of numpy works
<li> Need to build wheel for dlib
<li> Need proper RTSP config
<li> If person side ways or wearing a cap or some obstruction it is not working
<li> CNN too slow.
<li>for combination other than hog and 0.5 resizing it is either extremely slow, feed closing with a snap. for 0.5 and hog, no face is detected, forget recognizing
</ul>

# Comparison of different models

| Model name   | Accuracy | Time         |
|------------|-----|--------------|
| HOG      | Low  | Low    |
| CNN        | Better  | Higher     |
| Deepface    | Best  | Highest       |
