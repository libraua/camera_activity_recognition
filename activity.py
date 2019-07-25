import tensorflow as tf
import tensorflow_hub as hub

tf.logging.set_verbosity(tf.logging.ERROR)

# Some modules to help with reading the UCF101 dataset.
import random
import re
import os
import tempfile
import cv2
import numpy as np

# Some modules to display an animation using imageio.
import imageio
from IPython import display

from urllib import request  # requires python3


#@title Helper functions for the UCF101 dataset

# Utilities to fetch videos from UCF101 dataset
UCF_ROOT = "http://crcv.ucf.edu/THUMOS14/UCF101/UCF101/"
_VIDEO_LIST = None
_CACHE_DIR = tempfile.mkdtemp()

def list_ucf_videos():
  """Lists videos available in UCF101 dataset."""
  global _VIDEO_LIST
  if not _VIDEO_LIST:
    index = request.urlopen(UCF_ROOT).read().decode("utf-8")
    videos = re.findall("(v_[\w_]+\.avi)", index)
    _VIDEO_LIST = sorted(set(videos))
  return list(_VIDEO_LIST)

def fetch_ucf_video(video):
  """Fetchs a video and cache into local filesystem."""
  cache_path = os.path.join(_CACHE_DIR, video)
  if not os.path.exists(cache_path):
    urlpath = request.urljoin(UCF_ROOT, video)
    print("Fetching %s => %s" % (urlpath, cache_path))
    data = request.urlopen(urlpath).read()
    open(cache_path, "wb").write(data)
  return cache_path

# Utilities to open video files using CV2
def crop_center_square(frame):
  y, x = frame.shape[0:2]
  min_dim = min(y, x)
  start_x = (x // 2) - (min_dim // 2)
  start_y = (y // 2) - (min_dim // 2)
  return frame[start_y:start_y+min_dim,start_x:start_x+min_dim]

def load_video(path, max_frames=0, resize=(224, 224)):
  cap = cv2.VideoCapture(path)
  frames = []
  try:
    while True:
      ret, frame = cap.read()
      if not ret:
        break
      frame = crop_center_square(frame)
      frame = cv2.resize(frame, resize)
      frame = frame[:, :, [2, 1, 0]]
      frames.append(frame)
      
      if len(frames) == max_frames:
        break
  finally:
    cap.release()
  return np.array(frames) / 255.0

def animate(images):
    converted_images = np.clip(images * 255, 0, 255).astype(np.uint8)
    imageio.mimsave('./animation.gif', converted_images, fps=25)
    with open('./animation.gif','rb') as f:
        display.display(display.Image(data=f.read(), height=300))


#@title Get the kinetics-400 labels
# Get the kinetics-400 action labels from the GitHub repository.
KINETICS_URL = "https://raw.githubusercontent.com/deepmind/kinetics-i3d/master/data/label_map_600.txt"
with request.urlopen(KINETICS_URL) as obj:
    labels = [line.decode("utf-8").strip() for line in obj.readlines()]
print("Found %d labels." % len(labels))

# Get the list of videos in the dataset.
ucf_videos = list_ucf_videos()
  
categories = {}
for video in ucf_videos:
    category = video[2:-12]
    if category not in categories:
        categories[category] = []
    categories[category].append(video)
print("Found %d videos in %d categories." % (len(ucf_videos), len(categories)))

for category, sequences in categories.items():
    summary = ", ".join(sequences[:2])
    print("%-20s %4d videos (%s, ...)" % (category, len(sequences), summary))


# Get a sample cricket video.
#sample_video = load_video(fetch_ucf_video("v_CricketShot_g04_c02.avi"))

while True:
    sample_video = load_video(os.environ['VIDEO_URL'], max_frames=125)
    print("sample_video is a numpy array of shape %s." % str(sample_video.shape))
    #animate(sample_video)


    # Run the i3d model on the video and print the top 5 actions.

    # First add an empty dimension to the sample video as the model takes as input
    # a batch of videos.
    model_input = np.expand_dims(sample_video, axis=0)

    # Create the i3d model and get the action probabilities.
    with tf.Graph().as_default():
        i3d = hub.Module("https://tfhub.dev/deepmind/i3d-kinetics-600/1")
        input_placeholder = tf.placeholder(shape=(None, None, 224, 224, 3), dtype=tf.float32)
        logits = i3d(input_placeholder)
        probabilities = tf.nn.softmax(logits)
        with tf.train.MonitoredSession() as session:
            [ps] = session.run(probabilities, feed_dict={input_placeholder: model_input})

    print("Top 5 actions:")
    for i in np.argsort(ps)[::-1][:5]:
        if ps[i] * 100 > 30:
            print("I thing you are " + labels[i])
        print("%-22s %.2f%%" % (labels[i], ps[i] * 100))