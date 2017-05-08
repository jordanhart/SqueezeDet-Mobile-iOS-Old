# Author: Bichen Wu (bichen@berkeley.edu) 08/25/2016

"""SqueezeDet Demo.

In image detection mode, for a given image, detect objects and draw bounding
boxes around them. In video detection mode, perform real-time detection on the
video stream.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import time
import sys
import os
import glob

import numpy as np
import tensorflow as tf

from config import *
from train import _draw_box
from nets import *

#from freeze
from tensorflow.python.framework import graph_util


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'mode', 'image', """'image' or 'video'.""")
tf.app.flags.DEFINE_string(
    'checkpoint', './data/model_checkpoints/squeezeDet/model.ckpt-87000',
    """Path to the model parameter file.""")
tf.app.flags.DEFINE_string(
    'input_path', './data/sample.png',
    """Input image or video to be detected. Can process glob input such as """
    """./data/00000*.png.""")
tf.app.flags.DEFINE_string(
    'out_dir', './data/out/', """Directory to dump output image or video.""")


def video_demo():
    """Detect videos."""

    cap = cv2.VideoCapture(FLAGS.input_path)

    # Define the codec and create VideoWriter object
    # fourcc = cv2.cv.CV_FOURCC(*'XVID')
    # fourcc = cv2.cv.CV_FOURCC(*'MJPG')
    # in_file_name = os.path.split(FLAGS.input_path)[1]
    # out_file_name = os.path.join(FLAGS.out_dir, 'out_'+in_file_name)
    # out = cv2.VideoWriter(out_file_name, fourcc, 30.0, (375,1242), True)
    # out = VideoWriter(out_file_name, frameSize=(1242, 375))
    # out.open()

    with tf.Graph().as_default():
        # Load model
        mc = kitti_squeezeDet_config()
        mc.BATCH_SIZE = 1
        # model parameters will be restored from checkpoint
        mc.LOAD_PRETRAINED_MODEL = False
        model = SqueezeDet(mc, FLAGS.gpu)

        saver = tf.train.Saver(model.model_params)

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            saver.restore(sess, FLAGS.checkpoint)

            times = {}
            count = 0
            while cap.isOpened():
                t_start = time.time()
                count += 1
                out_im_name = os.path.join(
                    FLAGS.out_dir, str(count).zfill(6) + '.jpg')

                # Load images from video and crop
                ret, frame = cap.read()
                if ret:
                    # crop frames
                    frame = frame[500:-205, 239:-439, :]
                    im_input = frame.astype(np.float32) - mc.BGR_MEANS
                else:
                    break

                t_reshape = time.time()
                times['reshape'] = t_reshape - t_start

                # Detect - Jordan. Step 1 - convert this part to mobile, then
                # re-impliment filtering. Then go figure out how to convert to
                # .pb and whats needed:
                det_boxes, det_probs, det_class = sess.run(
                    [model.det_boxes, model.det_probs, model.det_class],
                    feed_dict={model.image_input: [im_input], model.keep_prob: 1.0})

                t_detect = time.time()
                times['detect'] = t_detect - t_reshape

                # Filter. Jordan - Post processing, need to re-write for mobile
                # filtering, and put on camera screen boxes and names
                final_boxes, final_probs, final_class = model.filter_prediction(
                    det_boxes[0], det_probs[0], det_class[0])

                keep_idx = [idx for idx in range(len(final_probs))
                            if final_probs[idx] > mc.PLOT_PROB_THRESH]
                final_boxes = [final_boxes[idx] for idx in keep_idx]
                final_probs = [final_probs[idx] for idx in keep_idx]
                final_class = [final_class[idx] for idx in keep_idx]

                t_filter = time.time()
                times['filter'] = t_filter - t_detect

                # Draw boxes

                # TODO(bichen): move this color dict to configuration file
                cls2clr = {
                    'car': (255, 191, 0),
                    'cyclist': (0, 191, 255),
                    'pedestrian': (255, 0, 191)
                }
                _draw_box(
                    frame, final_boxes,
                    [mc.CLASS_NAMES[idx] + ': (%.2f)' % prob
                        for idx, prob in zip(final_class, final_probs)],
                    cdict=cls2clr
                )

                t_draw = time.time()
                times['draw'] = t_draw - t_filter

                cv2.imwrite(out_im_name, frame)
                # out.write(frame)

                times['total'] = time.time() - t_start

                # time_str = ''
                # for t in times:
                #   time_str += '{} time: {:.4f} '.format(t[0], t[1])
                # time_str += '\n'
                time_str = 'Total time: {:.4f}, detection time: {:.4f}, filter time: '\
                           '{:.4f}'. \
                    format(times['total'], times['detect'], times['filter'])

                print(time_str)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    # Release everything if job is finished
    cap.release()
    # out.release()
    cv2.destroyAllWindows()


def image_demo():
    """Detect image."""

    with tf.Graph().as_default():
        # Load model
        mc = kitti_squeezeDet_config()
        mc.BATCH_SIZE = 1
        # model parameters will be restored from checkpoint
        mc.LOAD_PRETRAINED_MODEL = False
        model = SqueezeDet(mc, FLAGS.gpu)

        saver = tf.train.Saver(model.model_params)

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            saver.restore(sess, FLAGS.checkpoint)

            for f in glob.iglob(FLAGS.input_path):
                im = cv2.imread(f)
                im = im.astype(np.float32, copy=False)
                im = cv2.resize(im, (mc.IMAGE_WIDTH, mc.IMAGE_HEIGHT))
                input_image = im - mc.BGR_MEANS

                # Detect
                det_boxes, det_probs, det_class = sess.run(
                    [model.det_boxes, model.det_probs, model.det_class],
                    feed_dict={model.image_input: [input_image], model.keep_prob: 1.0})

                # Filter
                final_boxes, final_probs, final_class = model.filter_prediction(
                    det_boxes[0], det_probs[0], det_class[0])

                keep_idx = [idx for idx in range(len(final_probs))
                            if final_probs[idx] > mc.PLOT_PROB_THRESH]
                final_boxes = [final_boxes[idx] for idx in keep_idx]
                final_probs = [final_probs[idx] for idx in keep_idx]
                final_class = [final_class[idx] for idx in keep_idx]

                # TODO(bichen): move this color dict to configuration file
                cls2clr = {
                    'car': (255, 191, 0),
                    'cyclist': (0, 191, 255),
                    'pedestrian': (255, 0, 191)
                }

                # Draw boxes
                _draw_box(
                    im, final_boxes,
                    [mc.CLASS_NAMES[idx] + ': (%.2f)' % prob
                        for idx, prob in zip(final_class, final_probs)],
                    cdict=cls2clr,
                )

                file_name = os.path.split(f)[1]
                out_file_name = os.path.join(FLAGS.out_dir, 'out_' + file_name)
                cv2.imwrite(out_file_name, im)
                print('Image detection output saved to {}'.format(out_file_name))


def freeze_graph(model_folder):
    # Load model from demo
  with tf.Graph().as_default():
    mc = kitti_squeezeDet_config()
    mc.BATCH_SIZE = 1
    # model parameters will be restored from checkpoint
    mc.LOAD_PRETRAINED_MODEL = False
    model = SqueezeDet(mc, FLAGS.gpu)


    # We retrieve our checkpoint fullpath
    input_checkpoint = '/Users/jordanhart/squeezeDet/src/data/model_checkpoints/squeezeDet/model.ckpt-87000'

    # We precise the file fullname of our freezed graph
    absolute_model_folder = "/".join(input_checkpoint.split('/')[:-1])
    output_graph = absolute_model_folder + "/frozen_model.pb"

    # Before exporting our graph, we need to precise what is our output node
    # This is how TF decides what part of the Graph he has to keep and what part it can dump
    # NOTE: this variable is plural, because you can have multiple output nodes
    output_node_names = ["probability/score", "probability/class_idx", "bbox/trimming/bbox"]

    # We clear devices to allow TensorFlow to control on which device it will load operations
    clear_devices = True

    # We import the meta graph and retrieve a Saver
    #saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

    #from demo
    saver = tf.train.Saver(model.model_params)


    # We retrieve the protobuf graph definition
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    # We start a session and restore the graph weights
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        saver.restore(sess, FLAGS.checkpoint)    
        #with tf.Session() as sess:
        #saver.restore(sess, input_checkpoint)
        #print([n.name for n in tf.get_default_graph().as_graph_def().node])
        # We use a built-in TF helper to export variables to constants
        output_graph_def = graph_util.convert_variables_to_constants(
            sess, # The session is used to retrieve the weights
            input_graph_def, # The graph_def is used to retrieve the nodes
            output_node_names#.split(",") # The output node names are used to select the usefull nodes
        )

        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))



def freeze_graph_drop_dropout(model_folder):
    # Load model from demo
  with tf.Graph().as_default():
    mc = kitti_squeezeDet_config()
    mc.BATCH_SIZE = 1
    # model parameters will be restored from checkpoint
    mc.LOAD_PRETRAINED_MODEL = False
    model = SqueezeDet(mc)


    # We retrieve our checkpoint fullpath
    input_checkpoint = '/Users/jordanhart/squeezeDet/src/data/model_checkpoints/squeezeDet/model.ckpt-87000'

    # We precise the file fullname of our freezed graph
    absolute_model_folder = "/".join(input_checkpoint.split('/')[:-1])
    output_graph = absolute_model_folder + "/frozen_model.pb"

    # Before exporting our graph, we need to precise what is our output node
    # This is how TF decides what part of the Graph he has to keep and what part it can dump
    # NOTE: this variable is plural, because you can have multiple output nodes
    output_node_names = ["probability/score", "probability/class_idx", "bbox/trimming/bbox"]

    # We clear devices to allow TensorFlow to control on which device it will load operations
    clear_devices = True

    # We import the meta graph and retrieve a Saver
    #saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

    #from demo
    saver = tf.train.Saver(model.model_params)


    # We retrieve the protobuf graph definition
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    # We start a session and restore the graph weights
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        saver.restore(sess, FLAGS.checkpoint)    
        #with tf.Session() as sess:
        #saver.restore(sess, input_checkpoint)
        #print([n.name for n in tf.get_default_graph().as_graph_def().node])
        # We use a built-in TF helper to export variables to constants
        output_graph_def = graph_util.convert_variables_to_constants(
            sess, # The session is used to retrieve the weights
            input_graph_def, # The graph_def is used to retrieve the nodes
            output_node_names#.split(",") # The output node names are used to select the usefull nodes
        )

        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))



def main(argv=None):
   if not tf.gfile.Exists(FLAGS.out_dir):
        tf.gfile.MakeDirs(FLAGS.out_dir)
#  if FLAGS.mode == 'image':
#    image_demo()
#  else:
#    video_demo()
   freeze_graph_drop_dropout('/Users/jordanhart/squeezeDet/src/data/model_checkpoints/squeezeDet/model.ckpt-87000')


if __name__ == '__main__':
    tf.app.run()
