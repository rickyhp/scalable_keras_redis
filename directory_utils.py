import os
import logging
import shutil

class Folder_Utils(object):

    def createEmptyFolder(self,directory):
        try:
            if os.path.exists(directory):
                shutil.rmtree(directory, ignore_errors=True)
            os.makedirs(directory)
        except OSError as e:
            logging.exception('OSError ' + e.__str__())