#from funciones import translateRotation
import cv2
import numpy as np
from scipy import ndimage
from tkinter import *
from skimage.measure import label
import math
import pandas as pd
import PIL.Image, PIL.ImageTk

def segmentador_de_histograma(frame,p_umbral):
	#De RGB a escala de grises
	I = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
	#Umbral de intensidad a partir de p
	umbral = np.max(I)*(p_umbral/100)
	#Con mascara con condicionante obtenemos una mascara binarizada
	mascara = np.uint8((I>umbral)*255)
	return mascara

def relleno(frame): return ndimage.binary_fill_holes(frame).astype(np.uint8)*255

def ruido_gaussiano(frame,p_noise_shape):
	#Gaussian_kernel_shape
	gks = int((p_noise_shape/100)*frame.shape[0])
	#Gks debe ser impar
	if gks%2 == 0: gks += 1
	#Gaussian_Blur
	frame_blur = cv2.GaussianBlur(frame,(gks,gks),0)
	return frame_blur

def enmarcar_objetos(frame,mascara,dibujo):
	#Encontramos los contornos
	contours, hierarchy	= cv2.findContours(mascara,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
	#Encerramos los contornos en rectangulos
	cajas_originales = np.empty((1, 4),dtype = np.int16)
	cajas_redimensionadas = np.empty((1, 4),dtype = np.int16)
	for i in contours:
			rect = cv2.boundingRect(i)
			x,y,w,h = rect
			#Dibujamos los rectangulos en el frame ingresado en un area mayor para contener el emisor y sus bordes
			#El area se incrementa en 1/3 de las dimensiones del lado del rectangulo original
			x1,y1,w1,h1 = int(x-(w/3)),int(y-(h/3)),int(5*w/3),int(5*h/3)
			#Agregamos el contorno a la lista
			cajas_originales = np.append(cajas_originales,[[x,y,w,h]],axis=0)
			cajas_redimensionadas = np.append(cajas_redimensionadas,[[x1,y1,w1,h1]],axis=0)
			if dibujo != 0:
				if len(str(dibujo)) >= 3 and str(dibujo)[0]==str(1):
					#Contornos
					cv2.drawContours(frame, contours, -1, (255, 0, 0), 2, cv2.LINE_AA)
				if len(str(dibujo)) >= 2 and str(dibujo)[1]==str(1):
					#Original
					cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
				if len(str(dibujo)) >= 1 and str(dibujo)[2]==str(1):
					#Redimensionado
					cv2.rectangle(frame,(x1,y1),(x1+w1,y1+h1),(0,0,255),2)
	cajas_originales[0] = [0,0,10,10]
	cajas_redimensionadas[0] = [0,0,10,10]
	return frame,cajas_originales,cajas_redimensionadas,contours

def recorte(frame,x,y,w,h): return frame[y:y+h,x:x+w]

def pinta_categorias(frame,categoria,x,y,w,h):
	if categoria == "Emisor":
		cv2.rectangle(frame ,(x,y),(x+w,y+h),(0,255,0),2)
	if categoria == "No_Emisor":
		cv2.rectangle(frame ,(x,y),(x+w,y+h),(0,0,255),2)

def generador_de_elpises(frame,contorno,dibujo):
	if len(contorno) >= 5:
		ellipse = cv2.fitEllipse(contorno)
		xcentroEllipse = ellipse[0][0]
		ycentroEllipse = ellipse[0][1]
		anchoEllipse = ellipse[1][0]
		altoEllipse = ellipse[1][1]
		rotationEllipse = ellipse[2]
		eccetricity = anchoEllipse/altoEllipse
		rotation = translateRotation(rotationEllipse,anchoEllipse,altoEllipse)
		if dibujo == 1:
			if (math.isnan(xcentroEllipse)): print("Nan")
			elif (math.isnan(ycentroEllipse)): print("Nan")
			elif (math.isnan(altoEllipse)): print("Nan")
			elif (math.isnan(anchoEllipse)): print("Nan")
			elif (math.isnan(rotationEllipse)): print("Nan")
			else:
				cv2.ellipse(frame, ellipse, (0,255,0), 3)
		return xcentroEllipse,ycentroEllipse,anchoEllipse,altoEllipse,rotationEllipse,rotation,eccetricity
	else: print("Contorno no cumple con el minimo de puntos")

def momentos(contorno):
	M = cv2.moments(contorno)
	print(type(M))
	return M


def caracterizador(frame,contorno,dibujo,recorte,p_umbral):
	#Elipse
	caracteristicas = {}
	nombres = []
	numerico = []
	xcentroEllipse,ycentroEllipse,anchoEllipse,altoEllipse,rotationEllipse,rotation,eccetricity = generador_de_elpises(frame,contorno,dibujo)
	nombres = np.append(nombres,('xcentroEllipse','ycentroEllipse','anchoEllipse','altoEllipse','rotationEllipse','rotation','eccetricity'))
	numerico = np.append(numerico,(xcentroEllipse,ycentroEllipse,anchoEllipse,altoEllipse,rotationEllipse,rotation,eccetricity))
	#Recorte
	#Caracterisitca 1 ============== Relación de Aspecto
	alto = recorte.shape[0]
	ancho = recorte.shape[1]
	if (alto or ancho)==0:
		alto+=1
		ancho+=1
	rel_aspecto = alto/ancho
	#Caracterisitca 2 ============== Relación lados con lados frame
	alto_frame = frame.shape[0]
	ancho_frame = frame.shape[1]
	rel_frame = (alto/alto_frame)+(ancho/ancho_frame)
	#Caracteristica 3 ============== Area
	nombres = np.append(nombres,('rel_aspecto','rel_frame'))
	numerico = np.append(numerico,(rel_aspecto,rel_frame))
	#Estadisticas por color
	#Colores
	gray = cv2.cvtColor(recorte,cv2.COLOR_RGB2GRAY)
	rojo = recorte[:,:,0]
	verde = recorte[:,:,1]
	azul = recorte[:,:,2]
	#Median
	median_recorte = np.median(recorte); median_rojo = np.median(rojo);	median_verde = np.median(verde); median_azul = np.median(azul);	median_gray = np.median(gray)
	#Average
	average_recorte = np.average(recorte); average_rojo = np.average(rojo); average_verde = np.average(verde); average_azul = np.average(azul); average_gray = np.average(gray)
	#Mean
	mean_recorte = np.mean(recorte); mean_rojo = np.mean(rojo); mean_verde = np.mean(verde); mean_azul = np.mean(azul); mean_gray = np.mean(gray)
	#Standart Desviation
	std_recorte = np.std(recorte); std_rojo = np.std(rojo); std_verde = np.std(verde); std_azul = np.std(azul); std_gray = np.std(gray)
	#Variance
	var_recorte = np.var(recorte);	var_rojo = np.var(rojo);	var_verde = np.var(verde);	var_azul = np.var(azul);	var_gray = np.var(gray)
	#Min
	min_recorte = np.min(recorte);	min_rojo = np.min(rojo);	min_verde = np.min(verde);	min_azul = np.min(azul);	min_gray = np.min(gray)
	#Max
	max_recorte = np.max(recorte);	max_rojo = np.max(rojo);	max_verde = np.max(verde);	max_azul = np.max(azul);	max_gray = np.max(gray)
	#Range of values
	ptp_recorte = np.ptp(recorte);	ptp_rojo = np.ptp(rojo);	ptp_verde = np.ptp(verde);	ptp_azul = np.ptp(azul);	ptp_gray = np.ptp(gray)
	#Percentile
	percentile_recorte = np.percentile(recorte,p_umbral-5);	percentile_rojo = np.percentile(rojo,p_umbral-5);	percentile_verde = np.percentile(verde,p_umbral-5);	percentile_azul = np.percentile(azul,p_umbral-5);	percentile_gray = np.percentile(gray,p_umbral-5)
	nombres = np.append(nombres,('median_recorte','median_rojo','median_verde','median_azul','median_gray','average_recorte','average_rojo','average_verde','average_azul','average_gray','mean_recorte','mean_rojo','average_verde','average_azul','mean_gray','std_recorte','std_rojo','std_verde','std_azul','std_gray','var_recorte','var_rojo','var_verde','var_azul','var_gray','min_recorte','min_rojo','min_verde','min_azul','min_gray','max_recorte','max_rojo','max_verde','max_azul','max_gray','ptp_recorte','ptp_rojo','ptp_verde','ptp_azul','ptp_gray','percentile_recorte','percentile_rojo','percentile_verde','percentile_azul','percentile_gray'))
	parametros = np.append(numerico,(median_recorte,median_rojo,median_verde,median_azul,median_gray,average_recorte,average_rojo,average_verde,average_azul,average_gray,mean_recorte,mean_rojo,average_verde,average_azul,mean_gray,std_recorte,std_rojo,std_verde,std_azul,std_gray,var_recorte,var_rojo,var_verde,var_azul,var_gray,min_recorte,min_rojo,min_verde,min_azul,min_gray,max_recorte,max_rojo,max_verde,max_azul,max_gray,ptp_recorte,ptp_rojo,ptp_verde,ptp_azul,ptp_gray,percentile_recorte,percentile_rojo,percentile_verde,percentile_azul,percentile_gray))
	#Agregar a caracteristicas
	a = 0
	for i in nombres:
		caracteristicas[i] = parametros[a]
		a+=1
	#Momentos
	M = momentos(contorno)
	caracteristicas.update(M)
	return caracteristicas

def toRGB(imagen):
	if len(imagen.shape) == 2:
		imagen = cv2.cvtColor(imagen,cv2.COLOR_GRAY2RGB)
	if len(imagen.shape) == 3:
		imagen = cv2.cvtColor(imagen,cv2.COLOR_BGR2RGB)
	return imagen

def encontrarDirectorio():
    c = __file__[len(__file__)-1]
    contador = 2
    while(c != "/"):
        ultimo = __file__[len(__file__)-contador]
        directorio = __file__[0:len(__file__)-contador+1]
        c = ultimo
        contador += 1
    #directorio.replace(" ","\ ")
    return directorio

def ventana_categoria(img,img2):
	def emisor():
	    cat.set("Emisor"); estado.set(True)
	    ventana_principal.config(background = "green"); ventana_principal.destroy()

	def no_emisor():
	    cat.set("No_Emisor"); estado.set(True)
	    ventana_principal.config(background = "red"); ventana_principal.destroy()
	
	def listo():
		print(cat.get)
		if cat.get() == "":
			print("No ha ingresado valores.")
		else:	
			estado.set(True)
			ventana_principal.destroy()

	def omitir():
		estado.set(False)
		ventana_principal.destroy()

	def finalizacion():
		final_ventana.set(True)
		ventana_principal.destroy()

	ventana_principal = Tk()

	ventana_principal.geometry("1000x1000")
	ventana_principal.config(background = "black", bd = 15)

	cat = StringVar()
	estado = BooleanVar()
	final_ventana = BooleanVar()

	#state="disabled"
	img = toRGB(img)
	img2 = toRGB(img2)
	img = cv2.resize(img,(0,0),fx = 0.3,fy = 0.3)
	img2 = cv2.resize(img2,(200,200))
	height, width, no_channels = img.shape
	height2, width2, no_channels2 = img2.shape
	#img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
	#img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)
	canvas = Canvas(ventana_principal, width = width, height = height)
	canvas2 = Canvas(ventana_principal, width = width2, height = height2)
	canvas.pack()
	canvas2.pack()
	img = PIL.Image.fromarray(img)
	img2 = PIL.Image.fromarray(img2)
	img = PIL.ImageTk.PhotoImage(img)
	img2 = PIL.ImageTk.PhotoImage(img2)
	canvas.create_image(0, 0, image=img, anchor=NW)
	canvas2.create_image(0, 0, image=img2, anchor=NW)

	Label(ventana_principal, text="Resultado").pack()
	Entry(ventana_principal, justify="center", textvariable=cat,state="disabled").pack()
	Label(ventana_principal, text="").pack()
	em = Button(ventana_principal, text="Emisor", command=emisor, height = 50, width = 20,background = "green")
	no_em = Button(ventana_principal, text="No_Emisor", command=no_emisor, height = 50, width = 20,background = "red")
	Button(ventana_principal, text="Listo", command=listo, height = 5, width = 5,background = "blue").pack(side="bottom")
	Button(ventana_principal, text="Omitir", command=omitir, height = 3, width = 5,background = "orange").pack(side="top")
	Button(ventana_principal, text="Finalizar", command=finalizacion, height = 3, width = 5,background = "black").pack(side="top")

	em.pack(side="left")
	no_em.pack(side="left")

	ventana_principal.mainloop()
	categoria = cat.get()
	est = estado.get()
	final = final_ventana.get()
	print("resultado : "+categoria)
	return categoria,est,final

#Funciones copiadas de internet:
def translateRotation(rotation, width, height):
    if (width < height):
        rotation = -1 * (rotation - 90)
    if (rotation > 90):
        rotation = -1 * (rotation - 180)
    rotation *= -1
    return round(rotation)

