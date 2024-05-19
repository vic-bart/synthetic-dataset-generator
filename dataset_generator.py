import cv2 as cv
from cv2.typing import MatLike
import numpy as np
import math
import random

def generate_filename(folder:str, id:int, file_extension:str='.png'):
  """
  Generates a filename for a given folder and image id (assumes 4-digit naming convention).
  """
  if id >= 1000:
    filename:str = folder + '/' + str(id) + file_extension
  elif id >= 100:
    filename:str = folder + '/0' + str(id) + file_extension
  elif id >= 10:
    filename:str = folder + '/00' + str(id) + file_extension
  else:
    filename:str = folder + '/000' + str(id) + file_extension
  return filename

def combine_images(id:int, background:MatLike, foreground:MatLike, size:float, offset: tuple[int, int], angle:float, motion:int) -> MatLike:
  """
  Combines two images, accounting for their transparent qualities.

  Credit to Mala for the alpha compositing: 
  https://stackoverflow.com/questions/40895785/using-opencv-to-overlay-transparent-image-onto-another-image
  """
  # Check if images are valid
  if background is None:
    raise Exception('Background not found')
  if foreground is None:
    raise Exception('Foreground not found')

  # Compress the foreground (i.e. bottle) to fit into the background
  foreground = cv.resize(foreground, None, fx=size, fy=size)

  # Apply rotations to the foreground
  foreground = rotate_image(foreground, angle)
  
  # Denormalise translation offsets from 0-1, to 0-width and 0-height
  w_offset:int = int(offset[0] * len(background[0]))
  h_offset:int = int(offset[1] * len(background))

  # Cap the offsets at the background's edges
  if h_offset + len(foreground) > len(background):
    h_offset = len(background) - len(foreground)
  if w_offset + len(foreground[0]) > len(background[0]):
    w_offset = len(background[0]) - len(foreground[0])
  if h_offset < 0 or w_offset < 0:
    raise Exception('Foreground does not fit in background')
  
  # Generate bounding box
  generate_bounding_box(
    id=id,
    background=background, 
    foreground=foreground, 
    class_id=0, 
    w_offset=w_offset / len(background[0]), 
    h_offset=h_offset / len(background)
    )

  # Make sure both images have alpha channels
  if not (has_alpha_channel(background) and has_alpha_channel(foreground)):
    background = cv.cvtColor(background, cv.COLOR_BGR2BGRA)
    foreground = cv.cvtColor(foreground, cv.COLOR_BGR2BGRA)

  # Normalise alpha channels from 0-255 to 0-1
  alpha_background = background[:,:,3] / 255
  alpha_foreground = foreground[:,:,3] / 255

  # Set adjusted colors
  for i in range(len(alpha_foreground)):
    for j in range(len(alpha_foreground[i])):
      for k in range(0, 3):
        background[i+h_offset,j+w_offset,k] = \
          (alpha_foreground[i,j] * foreground[i,j,k]) + \
            (alpha_background[i+h_offset,j+w_offset] * background[i+h_offset,j+w_offset,k] * \
             (1 - alpha_foreground[i,j]))

  # Set adjusted alpha and denormalise back to 0-255
  for i in range(len(alpha_foreground)):
    for j in range(len(alpha_foreground[i])):
      background[i+h_offset,j+w_offset,3] = (1 - (1 - alpha_foreground[i,j]) * (1 - alpha_background[i+h_offset,j+w_offset])) * 255

  # Apply motion blur to the foreground
  if motion > 0:
    background = blur_image(background, motion)

  # Return the composite image
  return background

def blur_image(img:MatLike, motion:int):
  """
  Applies horizontal motion blur to an image.

  Credit to GeeksforGeeks:
  https://www.geeksforgeeks.org/opencv-motion-blur-in-python/
  """
  blur_mat = np.zeros((motion, motion))

  # Fill the middle row with ones
  blur_mat[(motion - 1)//2, :] = np.ones(motion)

  # Normalise
  blur_mat /= motion
  
  # Apply the horizontal blur matrix
  img = cv.filter2D(img, -1, blur_mat)
  return img

def has_alpha_channel(img:MatLike) -> bool:
  """
  Checks if the image has an alpha channel (.png) or doesn't (.jpg).
  """
  if len(img[0,0]) == 3:
    return False
  return True

def rotate_image(img:MatLike, angle:float) -> MatLike:
  """
  Rotates an image.

  Credit to Alex Rodrigues for the original answer, and JTIM for fixing boundary cropping:
  https://stackoverflow.com/questions/9041681/opencv-python-rotate-image-by-x-degrees-around-specific-point
  """
  h, w = img.shape[:2]
  img_c = (w / 2, h / 2)
  rot_mat = cv.getRotationMatrix2D(img_c, angle, 1)
  # Resizes image to prevent boundary cropping from rotation
  rad = math.radians(angle)
  sin = math.sin(rad)
  cos = math.cos(rad)
  b_w = int((h * abs(sin)) + (w * abs(cos)))
  b_h = int((h * abs(cos)) + (w * abs(sin)))
  rot_mat[0, 2] += ((b_w / 2) - img_c[0])
  rot_mat[1, 2] += ((b_h / 2) - img_c[1])

  img = cv.warpAffine(img, rot_mat, (b_w, b_h), flags=cv.INTER_LINEAR)
  return img

def generate_bounding_box(id:int, background:MatLike, foreground:MatLike, class_id:int, w_offset:float, h_offset:float) -> None:
  width:float = len(foreground[0]) / len(background[0])
  center_x:float = w_offset + width/2
  height:float = len(foreground) / len(background)
  center_y:float = h_offset + height/2
  bounding_box_filename:str = generate_filename('synthetic_dataset', id, '.txt')
  with open(bounding_box_filename, 'w') as bounding_box:
    bounding_box.write(f'{class_id} {center_x} {center_y} {width} {height}')

def main() -> None:
  """
  Generates a single image.
  """
  background_filename:str = generate_filename(folder='background', id=0)
  foreground_filename:str = generate_filename(folder='hammer', id=14)
  background:MatLike = cv.imread(background_filename, cv.IMREAD_UNCHANGED)
  foreground:MatLike = cv.imread(foreground_filename, cv.IMREAD_UNCHANGED)

  composite:MatLike = combine_images(
    id=0,
    background=background, 
    foreground=foreground, 
    size=0.05, # Normalised to foreground resolution
    offset=(0.4, 0.6), # Normalised to background width and height
    angle=0,
    motion=10
    )
  composite_filename:str = generate_filename(folder='synthetic_dataset', id=0)
  cv.imwrite(composite_filename, composite)

def generate_images(object_folders:list=['bottle'], background_variants:int=1, foreground_variants:int=1, x_variants:int=1, y_variants:int=1, a_variants:int=1, m_variants:int=1) -> None:
  """
  Generates o*b*f*x*y*a*m unique images to use as a synthetic dataset for YOLO model training.
  """
  id:int = 0
  # Get background
  for b in range(background_variants):
    background_filename:str = generate_filename(folder='background', id=b)
    background:MatLike = cv.imread(background_filename, cv.IMREAD_UNCHANGED)
  # Get object
    for o in object_folders:
  # Get foreground
      for f in range(foreground_variants):
        foreground_filename:str = generate_filename(folder=o, id=f)
        foreground:MatLike = cv.imread(foreground_filename, cv.IMREAD_UNCHANGED)
  # Get height/y offset
        for y in range(y_variants):
          min_h_offset = 0.2 # Minimum height offset from the top (normalised to background height), for the foreground
          y:float = random.uniform(min_h_offset, 1)
  # Get width/x offset
          for x in range(x_variants):
            x:float = random.uniform(0, 1)
  # Get angle/a offset
            for a in range(a_variants):
              a:float = random.uniform(0, 360)
  # Get angle/a offset
              for m in range(m_variants):
                max_blur:int = 30
                m = random.randint(0, max_blur)
  # Generate composite image
                composite:MatLike = combine_images(
                  id=id,
                  background=background, 
                  foreground=foreground, 
                  size=0.1, # Normalised to foreground resolution
                  offset=(x, y), # Normalised to background width and height
                  angle=a,
                  motion=m
                  )
                composite_filename:str = generate_filename(folder='synthetic_dataset', id=id)
                cv.imwrite(composite_filename, composite)
                id += 1

if __name__=="__main__":
  main()
  # random.seed(3)
  # generate_images(
  #   object_folders=['bottle', 'hammer'],
  #   background_variants=2,
  #   foreground_variants=2,
  #   x_variants=2,
  #   y_variants=1,
  #   a_variants=1,
  #   m_variants=1
  # )