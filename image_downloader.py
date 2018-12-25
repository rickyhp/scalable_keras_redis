from bs4 import BeautifulSoup
import os, errno
import threading
import logging
import requests
import mimetypes
import sys
from directory_utils import Folder_Utils

class Download_Image(object):
    def __init__(self, html_data, folderName, create_folder=False):
        self.folderName = folderName
        self.html_data = html_data
        self.create_folder = create_folder
    
    def _saveImages(self, url_folderName, folder, links, index):
        try:
            for link in links:
                image = link.get("src")
                if 'http' not in image or 'https' not in image:
                    image = 'http://' + url_folderName + '/' + image
                print('image url = ' + image)
                r2 = requests.get(image)
                content_type = r2.headers['content-type']
                extension = mimetypes.guess_extension(content_type)
                if (extension is None):
                    extension = ''
                with open(folder + '/' + str(index) + extension, "wb") as f:
                    f.write(r2.content)
                    index = index + 1
        except Exception as e:
            logging.exception(e.__str__())
            pass
    
    def downloadImages(self):
        soup = BeautifulSoup(self.html_data, "html.parser")
        index=1
        try:
            dir_path = os.path.dirname(os.path.realpath(__file__))
            folder = os.path.join(dir_path, self.folderName)
            if self.create_folder:
                dir_helper = Folder_Utils()
                dir_helper.createEmptyFolder(folder)
        except OSError as e:
            logging.exception('OSError ' + e.__traceback__.__str__())
        link_array = soup.find_all('img');
        count = len(link_array)
        first_links = []
        second_links = []
        thrid_links = []
        forth_links = []
        if count >= 4:
            quater_num = int(round(count/4))
            first_links = link_array[:quater_num]
            second_links = link_array[quater_num : 2*quater_num]
            thrid_links = link_array[2*quater_num : 3 * quater_num]
            forth_links = link_array[3 * quater_num :]
            t1 = threading.Thread(target=self._saveImages, args=(self.folderName, folder, first_links, 1))
            t2 = threading.Thread(target=self._saveImages, args=(self.folderName, folder, second_links, quater_num + 1))
            t3 = threading.Thread(target=self._saveImages, args=(self.folderName, folder, thrid_links, 2 * quater_num + 1))
            t4 = threading.Thread(target=self._saveImages, args =(self.folderName, folder, forth_links, 3 * quater_num + 1))
            
            t1.start()
            t2.start()
            t3.start()
            t4.start()
            
            t1.join()
            t2.join()
            t3.join()
            t4.join()    
        else:
            self._saveImages(link_array, index)