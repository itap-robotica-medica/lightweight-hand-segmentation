# Copyright 2020 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MediaPipe solution drawing utils."""

import math
from typing import List, Mapping, Optional, Tuple, Union

import cv2
import dataclasses
import matplotlib.pyplot as plt
import numpy as np

from mediapipe.framework.formats import detection_pb2
from mediapipe.framework.formats import location_data_pb2
from mediapipe.framework.formats import landmark_pb2

_PRESENCE_THRESHOLD = 0.5
_VISIBILITY_THRESHOLD = 0.5
_RGB_CHANNELS = 3

WHITE_COLOR = (224, 224, 224)
BLACK_COLOR = (0, 0, 0)
RED_COLOR = (0, 0, 255)
GREEN_COLOR = (0, 128, 0)
BLUE_COLOR = (255, 0, 0)


@dataclasses.dataclass
class DrawingSpec:
    # Color for drawing the annotation. Default to the white color.
    color: Tuple[int, int, int] = WHITE_COLOR
    # Thickness for drawing the annotation. Default to 2 pixels.
    thickness: int = 2
    # Circle radius. Default to 2 pixels.
    circle_radius: int = 2


def _normalized_to_pixel_coordinates(
        normalized_x: float, normalized_y: float, image_width: int,
        image_height: int) -> Union[None, Tuple[int, int]]:
    """Converts normalized value pair to pixel coordinates."""

    # Checks if the float value is between 0 and 1.
    def is_valid_normalized_value(value: float) -> bool:
        return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                          math.isclose(1, value))

    if not (is_valid_normalized_value(normalized_x) and
            is_valid_normalized_value(normalized_y)):
        # TODO: Draw coordinates even if it's outside of the image bounds.
        return None
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    return x_px, y_px


def draw_detection(
        image: np.ndarray,
        detection: detection_pb2.Detection,
        keypoint_drawing_spec: DrawingSpec = DrawingSpec(color=RED_COLOR),
        bbox_drawing_spec: DrawingSpec = DrawingSpec()):
    """Draws the detction bounding box and keypoints on the image.
  Args:
    image: A three channel RGB image represented as numpy ndarray.
    detection: A detection proto message to be annotated on the image.
    keypoint_drawing_spec: A DrawingSpec object that specifies the keypoints'
      drawing settings such as color, line thickness, and circle radius.
    bbox_drawing_spec: A DrawingSpec object that specifies the bounding box's
      drawing settings such as color and line thickness.
  Raises:
    ValueError: If one of the followings:
      a) If the input image is not three channel RGB.
      b) If the location data is not relative data.
  """
    if not detection.location_data:
        return
    if image.shape[2] != _RGB_CHANNELS:
        raise ValueError('Input image must contain three channel rgb data.')
    image_rows, image_cols, _ = image.shape

    location = detection.location_data
    if location.format != location_data_pb2.LocationData.RELATIVE_BOUNDING_BOX:
        raise ValueError(
            'LocationData must be relative for this drawing funtion to work.')
    # Draws keypoints.
    for keypoint in location.relative_keypoints:
        keypoint_px = _normalized_to_pixel_coordinates(keypoint.x, keypoint.y,
                                                       image_cols, image_rows)
        cv2.circle(image, keypoint_px, keypoint_drawing_spec.circle_radius,
                   keypoint_drawing_spec.color, keypoint_drawing_spec.thickness)
    # Draws bounding box if exists.
    if not location.HasField('relative_bounding_box'):
        return
    relative_bounding_box = location.relative_bounding_box
    rect_start_point = _normalized_to_pixel_coordinates(
        relative_bounding_box.xmin, relative_bounding_box.ymin, image_cols,
        image_rows)
    rect_end_point = _normalized_to_pixel_coordinates(
        relative_bounding_box.xmin + relative_bounding_box.width,
        relative_bounding_box.ymin + +relative_bounding_box.height, image_cols,
        image_rows)
    cv2.rectangle(image, rect_start_point, rect_end_point,
                  bbox_drawing_spec.color, bbox_drawing_spec.thickness)


def draw_landmarks(
        image: np.ndarray,
        landmark_list: landmark_pb2.NormalizedLandmarkList,
        connections: Optional[List[Tuple[int, int]]] = None,
        connection_drawing_spec: Union[DrawingSpec,
                                       Mapping[Tuple[int, int],
                                               DrawingSpec]] = DrawingSpec()):
    """Draws the landmarks and the connections on the image.
  Args:
    image: A three channel RGB image represented as numpy ndarray.
    landmark_list: A normalized landmark list proto message to be annotated on
      the image.
    connections: A list of landmark index tuples that specifies how landmarks to
      be connected in the drawing.
    landmark_drawing_spec: Either a DrawingSpec object or a mapping from
      hand landmarks to the DrawingSpecs that specifies the landmarks' drawing
      settings such as color, line thickness, and circle radius.
    connection_drawing_spec: Either a DrawingSpec object or a mapping from
      hand connections to the DrawingSpecs that specifies the
      connections' drawing settings such as color and line thickness.
  Raises:
    ValueError: If one of the followings:
      a) If the input image is not three channel RGB.
      b) If any connetions contain invalid landmark index.
  """
    if not landmark_list:
        return
    # if image.shape[2] != _RGB_CHANNELS:
    #     raise ValueError('Input image must contain three channel rgb data.')
    image_rows, image_cols = image.shape[:2]
    idx_to_coordinates = {}
    for idx, landmark in enumerate(landmark_list.landmark):
        if ((landmark.HasField('visibility') and
             landmark.visibility < _VISIBILITY_THRESHOLD) or
                (landmark.HasField('presence') and
                 landmark.presence < _PRESENCE_THRESHOLD)):
            continue
        landmark_px = _normalized_to_pixel_coordinates(landmark.x, landmark.y,
                                                       image_cols, image_rows)
        if landmark_px:
            idx_to_coordinates[idx] = landmark_px
    if connections and True:
        num_landmarks = len(landmark_list.landmark)
        # Draws the connections if the start and end landmarks are both visible.
        for connection in connections:
            start_idx = connection[0]
            end_idx = connection[1]
            if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):
                raise ValueError(f'Landmark index is out of range. Invalid connection '
                                 f'from landmark #{start_idx} to landmark #{end_idx}.')
            if start_idx in idx_to_coordinates and end_idx in idx_to_coordinates:
                drawing_spec = connection_drawing_spec[connection] if isinstance(
                    connection_drawing_spec, Mapping) else connection_drawing_spec
                cv2.line(image, idx_to_coordinates[start_idx],
                         idx_to_coordinates[end_idx], drawing_spec.color,
                         drawing_spec.thickness)
    # fill palm
    try:
        coordinates = np.int32(np.array([idx_to_coordinates[idx] for idx in (0, 1, 5, 17)]).reshape(1, 4, 2))
        cv2.fillPoly(image, coordinates, _GRAY)
    except KeyError:
        pass


def draw_axis(
        image: np.ndarray,
        rotation: np.ndarray,
        translation: np.ndarray,
        focal_length: Tuple[float, float] = (1.0, 1.0),
        principal_point: Tuple[float, float] = (0.0, 0.0),
        axis_length: float = 0.1,
        axis_drawing_spec: DrawingSpec = DrawingSpec()):
    """Draws the 3D axis on the image.
  Args:
    image: A three channel RGB image represented as numpy ndarray.
    rotation: Rotation matrix from object to camera coordinate frame.
    translation: Translation vector from object to camera coordinate frame.
    focal_length: camera focal length along x and y directions.
    principal_point: camera principal point in x and y.
    axis_length: length of the axis in the drawing.
    axis_drawing_spec: A DrawingSpec object that specifies the xyz axis
      drawing settings such as line thickness.
  Raises:
    ValueError: If one of the followings:
      a) If the input image is not three channel RGB.
  """
    # if image.shape[2] != _RGB_CHANNELS:
    #     raise ValueError('Input image must contain three channel rgb data.')
    image_rows, image_cols, _ = image.shape
    # Create axis points in camera coordinate frame.
    axis_world = np.float32([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    axis_cam = np.matmul(rotation, axis_length * axis_world.T).T + translation
    x = axis_cam[..., 0]
    y = axis_cam[..., 1]
    z = axis_cam[..., 2]
    # Project 3D points to NDC space.
    fx, fy = focal_length
    px, py = principal_point
    x_ndc = np.clip(-fx * x / (z + 1e-5) + px, -1., 1.)
    y_ndc = np.clip(-fy * y / (z + 1e-5) + py, -1., 1.)
    # Convert from NDC space to image space.
    x_im = np.int32((1 + x_ndc) * 0.5 * image_cols)
    y_im = np.int32((1 - y_ndc) * 0.5 * image_rows)
    # Draw xyz axis on the image.
    origin = (x_im[0], y_im[0])
    x_axis = (x_im[1], y_im[1])
    y_axis = (x_im[2], y_im[2])
    z_axis = (x_im[3], y_im[3])
    cv2.arrowedLine(image, origin, x_axis, RED_COLOR, axis_drawing_spec.thickness)
    cv2.arrowedLine(image, origin, y_axis, GREEN_COLOR,
                    axis_drawing_spec.thickness)
    cv2.arrowedLine(image, origin, z_axis, BLUE_COLOR,
                    axis_drawing_spec.thickness)


def _normalize_color(color):
    return tuple(v / 255. for v in color)

import hands_connections

"""_RADIUS = 5
_RED = (48, 48, 255)
_GREEN = (48, 255, 48)
_BLUE = (192, 101, 21)
_YELLOW = (0, 204, 255)
_GRAY = (128, 128, 128)
_PURPLE = (128, 64, 128)
_PEACH = (180, 229, 255)
_WHITE = (224, 224, 224)"""

_RADIUS = 5
_GREEN = 35
_BLUE = 35*2
_YELLOW = 35*3
_GRAY = 35*4
_PURPLE = 35*5
_PEACH = 35*6

# Hands
_THICKNESS_WRIST_MCP = 7
_THICKNESS_FINGER = 7

_HAND_CONNECTION_STYLE = {
    hands_connections.HAND_PALM_CONNECTIONS:
        DrawingSpec(color=_GRAY, thickness=_THICKNESS_WRIST_MCP),
    hands_connections.HAND_THUMB_CONNECTIONS:
        DrawingSpec(color=_PEACH, thickness=_THICKNESS_FINGER),
    hands_connections.HAND_INDEX_FINGER_CONNECTIONS:
        DrawingSpec(color=_PURPLE, thickness=_THICKNESS_FINGER),
    hands_connections.HAND_MIDDLE_FINGER_CONNECTIONS:
        DrawingSpec(color=_YELLOW, thickness=_THICKNESS_FINGER),
    hands_connections.HAND_RING_FINGER_CONNECTIONS:
        DrawingSpec(color=_GREEN, thickness=_THICKNESS_FINGER),
    hands_connections.HAND_PINKY_FINGER_CONNECTIONS:
        DrawingSpec(color=_BLUE, thickness=_THICKNESS_FINGER)
}


def get_custom_hand_connections_style(
) -> Mapping[Tuple[int, int], DrawingSpec]:
  """Returns the default hand connections drawing style.
  Returns:
      A mapping from each hand connection to its default drawing spec.
  """
  hand_connection_style = {}
  for k, v in _HAND_CONNECTION_STYLE.items():
    for connection in k:
      hand_connection_style[connection] = v
  return hand_connection_style