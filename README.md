# Human Trajectory
This is a ROS package that classifies human poses from bayes_people_tracker_logging in mongodb into human movements or non-human movements. The classification is based on poses and velocity of poses which are applied to KNN classifier. 
This package can only be run offline retrieving human poses from perception_people 

Run this package by typing 

```
rosrun human_movement_identifier classifier.py [train_ratio] [chunk] [alpha] [beta] [accuracy (1/0)]
```

where ```[train_ratio]``` is the training data ratio between 0.0 and 1.0.
Setting 0.9 makes 0.1 of all data will be used as test data.
```[chunk]``` is the length of poses for each data. Long sequence of poses will
be divided into chunk of ```[chunk]``` before it is processed.
```[alpha]``` is the modifier for poses.
```[beta]``` is the modifier for velocity.
```[accuracy]``` is the option to choose between knowing the accuracy of this
classifier (1) or just wanting to test data from test data.
