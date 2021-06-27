# Drowsing Detector

This project was inspired by Sigmoidal's Computer Vision Master Class
and Adrian Rosebroke Article.

We use a shape predictor 68 face landmarks model to detect the
face points that represents eyes, mouth, jaw, eyebrows and nose.
With this points we were able to detect how much the eyes are open/closed and
trigger some alert depending on it's value.

The predictor is loaded using the `dlib` package.

## Usage

- Clone this repo
- Navigate to project directory
- Download the trained model
- Install the dependencies
- Set the WEBCAM index (through environment variable or code)
- Run

```shell
git clone https://github.com/virb30/drowsing_detector.git drowsing_detector
cd drowsing_detector
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
pip install -r requirements.txt
# export WEBCAM=1
set WEBCAM=1
python .
```

### Note About dlib

Installation of `dlib` on Windows can be tricky.

Follow these SO instructions if needed:

https://stackoverflow.com/questions/41912372/dlib-installation-on-windows-10

## References

For more details about the original project that inspired this one can be found on:

https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/
