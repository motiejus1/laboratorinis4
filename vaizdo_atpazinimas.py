##importuojame packageus reikalingus
## naudojama 3.6 python verisja


## komandos instaliuojant su pip
## pip install numpy
## pip install pip install opencv-python
## pip install imutils
## pip install pytesseract
## pip install matplotlib


import numpy as np
import cv2
from imutils.object_detection import non_max_suppression
import pytesseract
from matplotlib import pyplot as plt

args = {"min_confidence":0.0001, "width":320, "height" : 320}



## butina instaliuoti Teseract programa ir nurodyti jos kelia
## kaip idiegti https://stackoverflow.com/questions/50951955/pytesseract-tesseractnotfound-error-tesseract-is-not-installed-or-its-not-i
## parsisiuntimo nuoroda https://github.com/UB-Mannheim/tesseract/wiki
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


## apsirasome paveikslelio argumenta
#args['image']="image/test.png"
#args['image']="image/test1.png"
#args['image']="image/Screenshot_2.jpg"
#args['image']="image/IMG_20200102_142109.jpg"
#args['image']="image/IMG_20200102_142537.jpg"
#args['image']="image/IMG_20200102_142552.jpg"
args['image']="image/IMG_20200102_143620.jpg"


image = cv2.imread(args['image'])

#Originalios kopija
orig = image.copy()
(origH, origW) = image.shape[:2]

# Nustatome paveikslelio auksti 320x320
(newW, newH) = (args["width"], args["height"])

# skaiciuojame koks mastelis tarp originalios nuotraukos ir kopijos. Geriausia kad originali nuotrauka butu 32 kartotinis
#pagal tai bus apibreziams rastas tekstas
rW = origW / float(newW)
rH = origH / float(newH)

# keiciame paveiksliuko dydi
image = cv2.resize(image, (newW, newH))
(H, W) = image.shape[:2]

# perduodam paveiksleli i EAST modeli
blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
	(123.68, 116.78, 103.94), swapRB=True, crop=False)

# pakrauname is anksto treaniruota EAST atpazinimo modeli
net = cv2.dnn.readNet('frozen_east_text_detection.pb')

# is east modelio gauname du duomenis: tikimybe, kad pasirinktame plote yra tekstas, teksto geometrija - dezutes kurioje yra tekstas koordinates

# pasiimame siuos du east modelio sluoksnius
layerNames = [
	"feature_fusion/Conv_7/Sigmoid",
	"feature_fusion/concat_3"]

# perduodama informacija i east modeli feedforward budu
net.setInput(blob)
(scores, geometry) = net.forward(layerNames)

#funkcija tikimybems skaiciuoti
def predictions(prob_score, geo):
	(numR, numC) = prob_score.shape[2:4]
	boxes = []
	confidence_val = []

	# apsibreziam kad tekstas nuotraukoje
	for y in range(0, numR):
		scoresData = prob_score[0, 0, y]
		x0 = geo[0, 0, y]
		x1 = geo[0, 1, y]
		x2 = geo[0, 2, y]
		x3 = geo[0, 3, y]
		anglesData = geo[0, 4, y]

		# einame per stulpelius
		for i in range(0, numC):
			if scoresData[i] < args["min_confidence"]:
				continue

			(offX, offY) = (i * 4.0, y * 4.0)

			# skaiciuojame sin ir cos
			angle = anglesData[i]
			cos = np.cos(angle)
			sin = np.sin(angle)

			# gauname apibrezimo dezutes dydi
			h = x0[i] + x2[i]
			w = x1[i] + x3[i]

			# skaiciuojame apibrezimo dezutes pradzia ir pabagiga
			endX = int(offX + (cos * x1[i]) + (sin * x2[i]))
			endY = int(offY - (sin * x1[i]) + (cos * x2[i]))
			startX = int(endX - w)
			startY = int(endY - h)

			boxes.append((startX, startY, endX, endY))
			confidence_val.append(scoresData[i])

	# grazinamas dezutes dydis ir patikimumo reiksme
	return (boxes, confidence_val)

(boxes, confidence_val) = predictions(scores, geometry)
boxes = non_max_suppression(np.array(boxes), probs=confidence_val)


# rezultatu masyvas
results = []

# randamos visos apibrezimo sritys
for (startX, startY, endX, endY) in boxes:
	#pagal masteli originaliame paveiksliuke nubreziame apibrezimo sritis
	startX = int(startX * rW)
	startY = int(startY * rH)
	endX = int(endX * rW)
	endY = int(endY * rH)

	#reikiama sritis teksto nuskaitymui
	r = orig[startY:endY, startX:endX]

	#nustatymai. -l eng - anglu kalbam --oem 1 - naudojamas LSTM tinkas, psm - paveiksliuko skirstymo metodai 8 - Treat the image as a single word.
	#pilna informacija parinktys.txt faile
	configuration = ("-l eng --oem 1 --psm 8")
    ##atpazinimas teksto apibrezimo srity
	text = pytesseract.image_to_string(r, config=configuration)

	# pridedamas apibrezimo sriies koordinates
	results.append(((startX, startY, endX, endY), text))

orig_image = orig.copy()

# apibrezimo sriciu atvaizdavimas ekrane
for ((start_X, start_Y, end_X, end_Y), text) in results:
    # display the text detected by Tesseract
    print("{}\n".format(text))


    text = "".join([x if ord(x) < 128 else "" for x in text]).strip()
    cv2.rectangle(orig_image, (start_X, start_Y), (end_X, end_Y),
                  (0, 0, 255), 2)
    cv2.putText(orig_image, text, (start_X, start_Y - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

plt.imshow(orig_image)
plt.title('Output')
plt.show()