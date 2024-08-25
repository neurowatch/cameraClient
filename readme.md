# Neurowatch - CameraClient

This project contains the code for the cameraClient, it is meant to run on a raspberry pi or other single board computer.

To run it, simply execute `python neurowatch.py`

## Overview

The CameraClient overal work is very simple. The `CameraController.capture_video()` method starts the flow, a `VideoCapture` object is obtained from `cv2.VideoCapture()` depending on the passed source it can be from a file or a camera feed. After that a series of use cases are called to build the background frame, detect motion, detect objects(if motion is detected), create and upload the clip to the server. This all occurs on a loop as any cv2 code.

## Setup

Running `python neurowatch.py` will start the execution and perform initial setup. A token is required which should be geneated serverside. Check the [server](https://github.com/neurowatch/server) repo for more information on who to geneate a token and all the required setup on that side.

## TestBed

The project contains a module called `TestBed` it will run tests to the separate modules of the project and produce a report. While the default results obtained by me are below, feel free to run it and see which configuration works best on your setup.

`TestBed` will evaluate each UseCase by separate, including some that are written but not used in the current implementation. For instance this is the case of the `BuildBackgroundFrame` usecase and the `GetMOG2BackgroundSubstractorFrame` usecase, which uses the cv2 `BackgroundSubtractorMOG2`. It will also evaluate the entire flow by running the `CameraController.caputre_video()` method.
