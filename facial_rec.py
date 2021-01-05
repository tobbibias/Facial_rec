### imports 
##importing os for file manipulation
import os 
##importing Facial_recognition a python library for a pre trained neural network to do the facial recognitioning.
import face_recognition as fr 
##importing opencv a deeplearning framwork that we will use to locate faces in pictures to do the facial recognition.
import cv2
## importing pyplot to display the finished recognized images.
from matplotlib import pyplot

#setting up our directories in this case known and unknown faces directories, the known face directory will contain multiple personalities in sub directories.
FACE_KNOWN = "known_face"
FACE_UNKNOWN = "unknown_face"
# arrays used to hold the images of people
face_known_array= []
name_known_array = []




#now we load the known images, for this we must set up som parameters that we will use with our deeplearning framework
MODEL = 'cnn' # The cnn model in facialrecognition framework will by default supply a 
ERROR_TOLERANCE =0.51 # the "distance" between the facial encodings of known and unknown image that is required for 
print("+loading images")
#looping over each subdirectory in known face directory in order to append the feature vector of each image to the array. 
for name in os.listdir(f'{FACE_KNOWN}'):
	print(f'-person {name}')
	for filename in os.listdir(f'{FACE_KNOWN}/{name}'):
		print(f"--image {filename}")
		image = fr.load_image_file(f'{FACE_KNOWN}/{name}/{filename}')
		# before we can get the feature vektor of the images we must locate the faces.
		locations = fr.face_locations(image, model=MODEL)
		encodings = fr.face_encodings(image,locations)[0]
		#now we have that we have the encodings we can append them into our arrays
		face_known_array.append(encodings)
		name_known_array.append(name)



# now we can iterate and encode the unknown faces.


print("+loading unknown faces...")


for filename in os.listdir(f'{FACE_UNKNOWN}'):
	
	print(f"--image {filename}")
	image = fr.load_image_file(f'{FACE_UNKNOWN}/{filename}')
	# before we can get the feature vektor of the images we must locate the faces.
	locations = fr.face_locations(image, model=MODEL)
	# note that unlike when we are locating faces in known images these images can contain multiple faces of multiple people, so we are dropping [0] and are running recognition on all found faces.
	encodings = fr.face_encodings(image,locations)
	
	#Now we iterate over face detected in the image and
	for face_encoding, face_location in zip(encodings, locations):


		recognition = fr.compare_faces(face_known_array,face_encoding,ERROR_TOLERANCE)
		Detected = None

		if True in recognition:
			Detected = name_known_array[recognition.index(True)]
			print(f'Match found : {Detected}')
			# now we want to illustrate our findings , with opencv we can draw recangles bordering the faces and even label the faces.

			#first we must set the 4 corners of the rectangle , these are the the 4 floats represented by the location variable.
			top_left = (face_location[3], face_location[0])
			bottom_right = (face_location[1], face_location[2])

			# use the cv2 function to paint the border on to the image.
			color =[255,0,0]
			cv2.rectangle(image, top_left, bottom_right,color,2)

			# we also want to label the frames with the person in the image so we make a nother rectangle that we paint the name on top of.
			top_left = (face_location[3], face_location[2])
			bottom_right = (face_location[1], face_location[2] + 22)

			# Paint frame
			cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
			# write the name of the person onto the frame
			cv2.putText(image, Detected, (face_location[3] + 10, face_location[2] + 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 200, 200),5)
			
			# we had issues using the cv2.imshow() so we opted for pyplot´s show and save functions instead.
			pyplot.imshow(image)
			#the pictures containing know faces will be saved in the directory afterwards.
			pyplot.savefig(f'{Detected}{filename}')

		else:
			print(f'-No match')

