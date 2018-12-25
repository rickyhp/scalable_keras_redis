import time
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
import selenium.webdriver.support.expected_conditions as EC
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import requests
import mimetypes
import os, errno
import argparse
from selenium.webdriver.support.wait import WebDriverWait
# from pip.index import Link
import threading
from PIL import Image
import logging
from image_downloader import Download_Image
import image_utils
from multiprocessing import Process 
from directory_utils import Folder_Utils
from sys import platform

class Scraping_Image(object):
    def __init__(self, url, dest_folder=''):
        self.url = url
        self.dest_folder = dest_folder
    
    def _download(self, data, folderName):
        image_downloader = Download_Image(data, folderName) 
        image_downloader.downloadImages()        
    
    def run(self):
        i = 0
        folderName = self.url if self.dest_folder == '' else self.dest_folder
        if folderName is not None or folderName != '':            
            if '//' in self.url:
                folderName = self.url.split('//')[1].replace('/','')
                self.dest_folder = folderName if self.dest_folder == '' else self.dest_folder
        ###browser = webdriver.PhantomJS("C:/Program Files/Python36/phantomjs-2.1.1-windows/bin/phantomjs.exe")
    #     browser = webdriver.Chrome('./chromedriver.exe')
        phantomjs_path = ''
        if platform == "linux" or platform == "linux2":
            phantomjs_path = os.path.abspath('/usr/local/bin/phantomjs')
        if platform == "win32":        
            phantomjs_path = os.path.abspath('./tools/windows/phantomjs.exe')
        browser = webdriver.PhantomJS(executable_path=phantomjs_path)
        browser.get(self.url)
        print(browser.title)
        browser.maximize_window()
        pause = 10
        
        lastHeight = browser.execute_script("return document.body.scrollHeight")
        
        print(lastHeight)
    #     browser.get_screenshot_as_file("test03_1_" + str(i) + ".png")
        next_height = 400
        for i in range(0, lastHeight, 400):
            next_height = i * next_height
            scroll_text = "window.scrollTo(0, {0});".format(next_height)
            browser.execute_script(scroll_text)
        browser.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(pause)
        data = browser.page_source
        dir_path = os.path.dirname(os.path.realpath(__file__))
        folder = os.path.join(dir_path, folderName)
        dir_helper = Folder_Utils()
        dir_helper.createEmptyFolder(folder)
        screenshot_name = folderName.replace('.', '_') + ".png"
        browser.get_screenshot_as_file(screenshot_name)
        t1 = threading.Thread(target=self._download, args=(data, folderName))
        t2 = threading.Thread(target=image_utils.slice_image, args=(os.path.join(folderName, "slices"), screenshot_name, 300,300,150))
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        browser.quit()
            
#scraping = Scraping_Image('http://www.bbc.com/')
#scraping.run()
