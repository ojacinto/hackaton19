from PIL import Image
import numpy as np

"""
O funcion abrir imagen 

O funcion para q te la matriz

- funcion para q te el vector (reshape)

O funcion que te recorte la imagen (se le pasa los parametros para recortar)
		crop = im.crop((221, 83, 463, 338))
O function para resize ()
"""

#Se abre la imagen 
#Params: Str [Nombre de archivo], Bool [Si se quiere imagen en escala de grises]
#Return: Imagen como objeto
def open_image(filename, grayscale = True):
	#Lee la imagen a color y la convierte a escala de grises
	if grayscale:
	    image = Image.open(filename).convert('L')
    #Lee la imagen a color
	else:
	    image = Image.open(filename)
	return image

#Devuelve el vector de la imagen
#Params: Obj [Imagen a obtener datos]
#Return: Matriz multidimensional si la imagen está en RGB, 
#		 una matriz de una dimensión si la imagen está en escala de grises
def array_image(image):
	array = np.array(image.getdata())
	return array

#Devuelve una sección cortada de la imagen
#Params: Obj [Imagen a cortar], Tuple [X_Min, X_Max, Y_Min, Y_Max]
#Returns: Imagen cortada como objeto
def crop_image(image, dim):
	return image.crop(dim)

#Devuelve una versión reescalada de la imagen
#Params: Obj [Imagen a cortar], Tuple [Widht, Height]
#Returns: Imagen reescalada como objeto
def resize_image(image, dim):
	return image.resize(dim)


def process_image_file(filename):
	im = open_image(filename)
	im = resize_image(im, [416, 416])
	return array_image(im)
