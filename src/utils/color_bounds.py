import numpy as np

cielab_color_bounds = {
  'red': [np.array([0,220,200]),np.array([170,255,255])],
  'blue': [np.array([0,158,0]),np.array([100,217,15])],
  'green' : [np.array([30,0,148]),np.array([250,108,168])],
  'yellow': [np.array([120,130,200]),np.array([255,150,230])],
  'brown' : [np.array([25,115,150]),np.array([60,130,250])],
  'white' : [np.array([150,127,127]),np.array([255,129,129])],
  'gray' : [np.array([31,127,127]),np.array([254,129,129])],
  'grey' : [np.array([31,127,127]),np.array([254,129,129])],
  'black' : [np.array([0,127,127]),np.array([70,129,129])]
}

hsv_color_bounds = {
    'red': [np.array([0,70,70]), np.array([4.4,255,255])],
    'red2': [np.array([175,110,140]), np.array([179,255,255])],
    'green': [np.array([47,70,117.3]), np.array([65,255,255])],
    'blue': [np.array([108,70,70]), np.array([117,255,255])],
    'cyan': [np.array([88.5,70,76]), np.array([95,255,255])],
    'orange': [np.array([12,70,100]), np.array([19,255,255])],
    'purple': [np.array([137,70,84]), np.array([142,255,255])],
    'pink': [np.array([157,70,100]), np.array([167,200,255])],
    'brown': [np.array([15,40,30]), np.array([19,255,190])],
    'yellow': [np.array([28.5,70,100]), np.array([32,255,255])],
    'black': [np.array([0,0,0]), np.array([179,255,70])],
    'white': [np.array([0,0,195]), np.array([179,3,255])],
    'gray': [np.array([0,0,40]), np.array([179,3,190])],
    'grey': [np.array([0,0,40]), np.array([179,3,190])]
}