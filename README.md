# SHREC'20 Challenge: sphericalDZ
This repository contains three notebooks to perform consequetively the steps for predicting the labels of a point cloud. We highlight that ground detection was performed using Quasi Lambda Flat zones (LFZ) using Bird Eye View (BEV) proposed by [1].

Notebooks in this repository are:
1) Spherical projection - example : Perform spherical projection of a point cloud.
2) Prediction of spherical projection : Predict labels using spherical prediction with detected ground using LFZ
3) Backprojection of 2D prediction : Perform KNN to label 3D points that are not visible in spherical projection due to resolution constraints of the image.

[1]  Hernandez,  J, Marcotegui,  B.  Point cloud segmentation towards urban ground modeling.   In:  2009 Joint Urban Remote Sensing Event. IEEE;2009, p. 1–5
