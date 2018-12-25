import sys
from PIL import Image
import logging
import os
from directory_utils import Folder_Utils
from threading import Thread, Lock, current_thread

def slice_image(dest_folder, image_file, height, width, step):
    dir_helper = Folder_Utils()
    lock = Lock()
    dir_helper.createEmptyFolder(dest_folder)
    im = Image.open(image_file)
    imgwidth, imgheight = im.size;
    count1 = [0]
    count2 = [0]
    if height > imgheight or width > imgwidth:
        try:
            im.save(os.path.join(dest_folder, "sliced.png"))
        except Exception as e:
            logging.exception(e.__str__())
        finally:
            return
    x_stop = imgwidth - width
    print('slicing')
    y_stop = imgheight - height
    half_y_stop = int(round(y_stop / 2))
    t1 = Thread(name="t1", target=crop_image, args=(im, count1, dest_folder, width, height, x_stop, half_y_stop, step, 0, lock))
    t2 = Thread(name="t2", target=crop_image, args=(im, count2, dest_folder, width, height, x_stop, y_stop, step, half_y_stop, lock))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    print('slicing finished, total {0}'.format(count1[0] + count2[0]))

def _crop_image_helper(im, box, dest_folder, count, name = "0"):
    try:
        count_value = count[0]     
        a = im.crop(box)
        count_value =count_value + 1            
        a.save(os.path.join(dest_folder, "sliced-IMG-{0}-{1}.png".format(name, count_value)))
        count[0] = count_value
    except Exception as e:
        logging.exception(e.__str__())
    
def crop_image(im, count, dest_folder, target_width, target_height, x_stop, y_stop, step, start_y = 0, lock = None):
    for j in range(start_y, y_stop, step):
        for i in range(0, x_stop, step):        
            box = (i, j, target_width+i, target_height+j)
            if lock is not None:
                with lock:
                    _crop_image_helper(im, box, dest_folder, count, current_thread().getName())
            else:
                _crop_image_helper(im, box, dest_folder, count)
# slice_image("test", "etcanada.com/news/299494/canadian-tennis-star-eugenie-bouchard-goes-topless-in-sports-illustrated-swimsuit-2018-issue/26.png", 100, 100, 50)  