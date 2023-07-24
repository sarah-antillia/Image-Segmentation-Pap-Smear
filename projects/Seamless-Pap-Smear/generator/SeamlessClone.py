# Copyright 2023 antillia.com Toshiyuki Arai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# SeamlessClone.py
# 2023/07/23 to-arai

import cv2
from PIL import Image, ImageDraw

import numpy as np
import traceback

class SeamlessClone:
  def __init__(self):
    pass

  def toOpenCV(self, pil_image):
    # Convert PIL image to OpenCV Image
    image = np.array(pil_image, dtype=np.uint8)
    if image.ndim == 3:
      channel = image.shape[2]
      if channel == 3:
        image = image[:, :, ::-1]
      if channel == 4:
        image = image[:, :, [2, 1, 0, 3]]
    return image

  def toPIL(self, cv_image):
    # Convert OpenCV image to PIL Image
    image = cv_image.copy()
    if image.ndim == 3:
      channel = image.shape[2]
      if channel == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      if channel == 4: 
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    image = Image.fromarray(image)
    return image

  def create(self, src_file, target_size):
    
    src = Image.open(src_file)
    w, h = src.size
    (W, H) = target_size
    # Target W and H must be greater than w and h of src_image size.
    pixel = src.getpixel((w-2, h-2))

    # 1 Create background to paste src 
    background = Image.new("RGB", (W, H), pixel)
    if W >=w:
      x = (W-w)//2
    else:
      raise Exception("Invalid target_size")
    if H >= h:
      y = (H-h)//2
    else:
      raise Exception("Invalid target_size")

    # 2 Paste src on to backgroud
    background.paste(src, (x, y))
    background.show()
    target  = Image.new("RGB", (W, H), pixel)
    # 3 Create mask 
    mask    = Image.new("L", (W, H))
    draw    = ImageDraw.Draw(mask)

    # 4 Draw a white rectangle on the black mask 
    draw.rectangle((x, y, x+w, y+h), fill="white")
    mask.show()
    target  = Image.new("RGB", (W, H), pixel)

    clone = self.seamlessClone(background, target, mask)
    return clone

  def seamlessClone(self, src, target, mask):
    ws, hs = src.size
    print(" ws:{} hs:{}".format(ws, hs))
    w, h   = target.size
    print(" w: {} h: {}".format(w, h))
    wm, hm = mask.size
    print(" wm:{} hm:{}".format(wm, hm))

    src    = self.toOpenCV(src)
    target = self.toOpenCV(target)
    mask   = self.toOpenCV(mask)
    center = (w//2, h//2)

    cloned = cv2.seamlessClone(src, target, mask, center, cv2.NORMAL_CLONE)
    cloned = self.toPIL(cloned)
    return cloned


if __name__ == "__main__":
  try:
    src_file = "./149143469-149143479-002.BMP"
    clone = SeamlessClone()
    target_size = (312, 312)
    image = clone.create(src_file, target_size)
    image.show()

  except:
    traceback.print_exc()



