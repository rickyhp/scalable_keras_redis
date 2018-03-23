# USAGE
# Start the server:
# 	python run_keras_server.py
# Submit a request via cURL:
# 	curl -X POST -F image=@jemma.png 'http://localhost:5000/predict'
# Submit a request via Python:
#	python simple_request.py 
# Submit a request via browser example
#       http://127.0.0.1:5000/predicturl?url=etcanada.com/news/299494/canadian-tennis-star-eugenie-bouchard-goes-topless-in-sports-illustrated-swimsuit-2018-issue

# import the necessary packages
from keras.applications import ResNet50
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from threading import Thread
from PIL import Image
import numpy as np
import base64
import flask
import redis
import uuid
import time
import json
import sys
import io
from bs4 import BeautifulSoup
import os, errno
import requests
import mimetypes
import glob
import caffe
from io import BytesIO

# initialize constants used to control image spatial dimensions and
# data type
#IMAGE_WIDTH = 224
#IMAGE_HEIGHT = 224
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256
IMAGE_CHANS = 3
IMAGE_DTYPE = "float32"

# initialize constants used for server queuing
IMAGE_QUEUE = "image_queue"
BATCH_SIZE = 32
SERVER_SLEEP = 0.25
CLIENT_SLEEP = 0.25

# initialize our Flask application, Redis server, and Keras model
app = flask.Flask(__name__)
db = redis.StrictRedis(host="localhost", port=6379, db=0)
model = None

def base64_encode_image(a):
	# base64 encode the input NumPy array
	return base64.b64encode(a).decode("utf-8")

def base64_decode_image(a, dtype, shape):
	# if this is Python 3, we need the extra step of encoding the
	# serialized NumPy string as a byte object
	if sys.version_info.major == 3:
		a = bytes(a, encoding="utf-8")

	# convert the string to a NumPy array using the supplied data
	# type and target shape
	a = np.frombuffer(base64.decodestring(a), dtype=dtype)
	a = a.reshape(shape)

	# return the decoded image
	return a

def prepare_image(image, target):
	# if the image mode is not RGB, convert it
	if image.mode != "RGB":
		image = image.convert("RGB")

	# resize the input image and preprocess it
	image = image.resize(target)
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)
	image = imagenet_utils.preprocess_input(image)

	# return the processed image
	return image

def classify_process():
	model = 'open_nsfw'

	# load the pre-trained Keras model (here we are using a model
	# pre-trained on ImageNet and provided by Keras, but you can
	# substitute in your own networks just as easily)
	print("* Loading model...")
	if(model == 'resnet50'):
		model = ResNet50(weights="imagenet")
		print("* ResNet50 imagenet model loaded")
	elif(model == 'open_nsfw'):
		# Pre-load caffe model.
		nsfw_net = caffe.Net('nsfw_model/deploy.prototxt',  # pylint: disable=invalid-name
					'nsfw_model/resnet_50_1by2_nsfw.caffemodel', caffe.TEST)
		print("* open_nsfw caffe model loaded")
		# Load transformer
		# Note that the parameters are hard-coded for best results
		caffe_transformer = caffe.io.Transformer({'data': nsfw_net.blobs['data'].data.shape})
		caffe_transformer.set_transpose('data', (2, 0, 1))  # move image channels to outermost
		caffe_transformer.set_mean('data', np.array([104, 117, 123]))  # subtract the dataset-mean value in each channel
		caffe_transformer.set_raw_scale('data', 255)  # rescale from [0, 1] to [0, 255]
		caffe_transformer.set_channel_swap('data', (2, 1, 0))  # swap channels from RGB to BGR

	# continually pool for new images to classify
	while True:
		# attempt to grab a batch of images from the database, then
		# initialize the image IDs and batch of images themselves
		queue = db.lrange(IMAGE_QUEUE, 0, BATCH_SIZE - 1)
		imageIDs = []
		batch = None

		# loop over the queue
		for q in queue:
			# deserialize the object and obtain the input image
			q = json.loads(q.decode("utf-8"))
			#image = base64_decode_image(q["image"], IMAGE_DTYPE,
			#	(1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANS))
			image = open(q["image"], 'rb').read()
			# check to see if the batch list is None
			if batch is None:
				batch = image

			# otherwise, stack the data
			else:
				batch = np.vstack([batch, image])
			
			print("Load from redis : ",q["id"])
			# update the list of image IDs
			imageIDs.append(q["id"])

		# check to see if we need to process the batch
		if len(imageIDs) > 0:
			if(model == 'resnet50'):
				# classify the batch
				print("* Batch size: {}".format(batch.shape))
				preds = model.predict(batch)
				results = imagenet_utils.decode_predictions(preds)

				# loop over the image IDs and their corresponding set of
				# results from our model
				for (imageID, resultSet) in zip(imageIDs, results):
					# initialize the list of output predictions
					output = []

					# loop over the results and add them to the list of
					# output predictions
					for (imagenetID, label, prob) in resultSet:
						r = {"label": label, "probability": float(prob)}
						output.append(r)

					# store the output predictions in the database, using
					# the image ID as the key so we can fetch the results
					db.set(imageID, json.dumps(output))

				# remove the set of images from our queue
				db.ltrim(IMAGE_QUEUE, len(imageIDs), -1)
			elif(model == 'open_nsfw'):
				# classify the batch
				#print("* Batch size: {}".format(batch.shape))
				print("Image : ",imageIDs[len(imageIDs)-1])
				# Classify
				scores = caffe_preprocess_and_compute(batch, caffe_transformer=caffe_transformer, caffe_net=nsfw_net,
								  output_layers=['prob'])

				# Scores is the array containing SFW / NSFW image probabilities
				# scores[1] indicates the NSFW probability
				print("NSFW score: %s " % scores[1])
				r = {"label": imageIDs[len(imageIDs)-1], "probability": float(scores[1])}
				output = []
				output.append(r)
				db.set(imageIDs[len(imageIDs)-1],json.dumps(output))
				# remove the set of images from our queue
				db.ltrim(IMAGE_QUEUE, len(imageIDs), -1)

		# sleep for a small amount
		time.sleep(SERVER_SLEEP)

def img_scraper(website):
	r = requests.get(website)
	data = r.text
	soup = BeautifulSoup(data, "lxml")
	folderName = website.split('//')[1]
	i = 1
	for link in soup.find_all('img'):
	        image = link.get("src")
	        if 'http' not in image or 'https' not in image:
	            image = 'http://' + folderName + '/' + image
	        print('image url = ' + image)
	        r2 = requests.get(image)
	        content_type = r2.headers['content-type']
	        extension = mimetypes.guess_extension(content_type)

	        try:
	            os.makedirs(folderName)
	        except OSError as e:
	            if e.errno != errno.EEXIST:
	                raise
	        if(extension is None):
	            extension = ''
	        with open(folderName+'/'+str(i)+extension, "wb") as f:
	            f.write(r2.content)
	            i = i + 1
	return folderName

@app.route("/predicturl", methods=["GET"])
def predicturl():
	# /predicturl?url=http://xxx.com/abcnews
	website = flask.request.args.get('url')
	print('scraping images at: ', website)
	folderName = img_scraper(website)
	#folderName = "etcanada.com"
	data = {"success": False}
	for filename in glob.glob(folderName+"/**/*.*", recursive=True):
		print("filename : ",filename)
		#image = Image.open(filename)
		#image = prepare_image(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
		#image = image.copy(order="C")
		k = str(uuid.uuid4())
		d = {"id": k, "image": filename}	
		#d = {"id": k, "image": base64_encode_image(image)}
		db.rpush(IMAGE_QUEUE, json.dumps(d))
		while True:
				# attempt to grab the output predictions
				output = db.get(k)

				# check to see if our model has classified the input
				# image
				if output is not None:
 					# add the output predictions to our data
 					# dictionary so we can return it to the client
					output = output.decode("utf-8")
					data[filename] = json.loads(output)

					# delete the result from the database and break
					# from the polling loop
					db.delete(k)
					break

				# sleep for a small amount to give the model a chance
				# to classify the input image
				time.sleep(CLIENT_SLEEP)
	data["success"] = True
	return flask.jsonify(data)

@app.route("/predict", methods=["POST"])
def predict():
	# initialize the data dictionary that will be returned from the
	# view
	data = {"success": False}

	# ensure an image was properly uploaded to our endpoint
	if flask.request.method == "POST":
		if flask.request.files.get("image"):
			# read the image in PIL format and prepare it for
			# classification
			image = flask.request.files["image"].read()
			image = Image.open(io.BytesIO(image))
			image = prepare_image(image, (IMAGE_WIDTH, IMAGE_HEIGHT))

			# ensure our NumPy array is C-contiguous as well,
			# otherwise we won't be able to serialize it
			image = image.copy(order="C")

			# generate an ID for the classification then add the
			# classification ID + image to the queue
			k = str(uuid.uuid4())
			d = {"id": k, "image": base64_encode_image(image)}
			db.rpush(IMAGE_QUEUE, json.dumps(d))

			# keep looping until our model server returns the output
			# predictions
			while True:
				# attempt to grab the output predictions
				output = db.get(k)

				# check to see if our model has classified the input
				# image
				if output is not None:
 					# add the output predictions to our data
 					# dictionary so we can return it to the client
					output = output.decode("utf-8")
					data["predictions"] = json.loads(output)

					# delete the result from the database and break
					# from the polling loop
					db.delete(k)
					break

				# sleep for a small amount to give the model a chance
				# to classify the input image
				time.sleep(CLIENT_SLEEP)

			# indicate that the request was a success
			data["success"] = True

	# return the data dictionary as a JSON response
	return flask.jsonify(data)

# open_nsfw caffe model image preparation
def resize_image(data, sz=(256, 256)):
    """
    Resize image. Please use this resize logic for best results instead of the 
    caffe, since it was used to generate training dataset 
    :param byte data:
        The image data
    :param sz tuple:
        The resized image dimensions
    :returns bytearray:
        A byte array with the resized image
    """
    im = Image.open(BytesIO(data))
    
    if im.mode != "RGB":
        im = im.convert('RGB')
    imr = im.resize(sz, resample=Image.BILINEAR)
    fh_im = BytesIO()
    imr.save(fh_im, format='JPEG')
    fh_im.seek(0)
    return fh_im

# open_nsfw caffe model
def caffe_preprocess_and_compute(pimg, caffe_transformer=None, caffe_net=None,
                                 output_layers=None):
    """
    Run a Caffe network on an input image after preprocessing it to prepare
    it for Caffe.
    :param PIL.Image pimg:
        PIL image to be input into Caffe.
    :param caffe.Net caffe_net:
    :param list output_layers:
        A list of the names of the layers from caffe_net whose outputs are to
        to be returned.  If this is None, the default outputs for the network
        are returned.
    :return:
        Returns the requested outputs from the Caffe net.
    """
    if caffe_net is not None:

        # Grab the default output names if none were requested specifically.
        if output_layers is None:
            output_layers = caffe_net.outputs

        img_bytes = resize_image(pimg, sz=(256, 256))
        image = caffe.io.load_image(img_bytes)

        H, W, _ = image.shape
        _, _, h, w = caffe_net.blobs['data'].data.shape
        h_off = max((H - h) / 2, 0)
        w_off = max((W - w) / 2, 0)
        crop = image[int(h_off):int(h_off + h), int(w_off):int(w_off + w), :]
        transformed_image = caffe_transformer.preprocess('data', crop)
        transformed_image.shape = (1,) + transformed_image.shape

        input_name = caffe_net.inputs[0]
        all_outputs = caffe_net.forward_all(blobs=output_layers,
                                            **{input_name: transformed_image})

        outputs = all_outputs[output_layers[0]][0].astype(float)
        return outputs
    else:
        return []

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
	# load the function used to classify input images in a *separate*
	# thread than the one used for main classification
	print("* Starting model service...")
	t = Thread(target=classify_process, args=())
	t.daemon = True
	t.start()

	# start the web server
	print("* Starting web service...")
	app.run()
