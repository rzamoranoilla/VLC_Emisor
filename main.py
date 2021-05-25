import cv2
import numpy as np
import funciones as fn
from tkinter import *
import datetime
import pandas as pd

today = datetime.date.today()

#Le indicara en que porcentaje superior del histograma de intensidad luminosa debe cortar.
p_umbral		= 99
#Le indicara el tamaño del kernel utilizado para la aplicacion de ruido gaussiano
p_noise_shape	= 2
#Es un indicador de tolerancia, es el porcentajo minimo del frame que necesita el elemento reconocido para ser considerado en la muestra.
p_corte			= 3

ruta = fn.encontrarDirectorio()
capeta_ubicacion_imagenes = ruta + "./Muestras/COCO/emisores/"
#capeta_ubicacion_imagenes = "C:/Users/raul/Desktop/8.- Tesis/5.- Redes Neuronales/Muestras/Muestras/COCO/"
n_max_imagenes = 5000

#Inicializamos el dataframe que contrendra la informacion del entrenamiento
datos_entrenamiento = pd.DataFrame()

#Recorremos las imagenes una por una.
numeros = np.arange(0,n_max_imagenes,1)
for i in numeros:
	nombre = str(i)+".jpg"
	while len(nombre) != 16:
		nombre = "0"+nombre
	img = cv2.imread(capeta_ubicacion_imagenes+nombre)
	#Si es que la imagen existe dentro de la muestra
	if isinstance(img,np.ndarray):
		#Hacemos copias del frame principal
		final = img.copy()
		enmarcado = img.copy()
		caracterizado = img.copy()
		#Segmentamos los valores del histograma superiores al umbral, aplicamos ruido gaussiano de acuerdo al p_noise_shape, rellenamos los elementos reconocidos y volvemos a aplicar ruido gaussiano de acuerdo al p_noise_shape.
		mascara = fn.segmentador_de_histograma(img, p_umbral)
		mascara = fn.ruido_gaussiano(mascara,p_noise_shape)
		mascara = fn.relleno(mascara)
		mascara = fn.ruido_gaussiano(mascara,p_noise_shape)
		#A partir de esta mascara es sobre la que trabajaremos para obtener las caracteristicas de manera automatica y enlistarlas junto al recorte de la zona reconocida.
		#La informacion se guardara de la siguiente manera en una carpeta llamada "Informacion":
		#Se creará un archivo con el formato 000000001.jpg
		#Se enlistara en un csv las caracteristicas obtenidas del objeto a partir de los contornos
		#La obtencion de cajas que rodearan al area de interes tiene propositos utilitarios
		reconocimiento,cajas_originales,cajas_redimensionadas,contornos = fn.enmarcar_objetos(enmarcado,mascara,111)
		#Para cada una de las cajas propuestas se recorta el area para almacenarla como imagen.
		#Iniciamos un contador que nos va a servir para leer los contornos paralelamente
		a = 0
		final_ventana = False
		terminar = False
		for i in cajas_redimensionadas:
			x,y,w,h = i[0],i[1],i[2],i[3]
			x1,y1,w1,h1 = cajas_originales[a][0],cajas_originales[a][1],cajas_originales[a][2],cajas_originales[a][3]
			#Condicionantes para que se pueda recortar y redimencionar correctamente si es que se sale de los margenes de la imagen.
			if x<=0: x=0
			if y<=0: y=0	
			if x+w>=img.shape[1]: w=img.shape[1]-x
			if y+h>=img.shape[0]: h=img.shape[0]-y
			#Se realizan los recortes correspondientes a 
			recorte = fn.recorte(img,x,y,w,h)
			recorte_a_guardar = fn.recorte(img,x1,y1,w1,h1)
			while(True):
				#Condicionante de tamaño de area
				if w<=((img.shape[0])*(p_corte/100)) or h<=((img.shape[1])*(p_corte/100)): a+=1; break
				#Hacemos una copia para rallar y dibujamos la caja y en contorno que vamos a analizar
				imagen = img.copy()
				cv2.drawContours(imagen, contornos[a-1], -1, (255, 255, 0), 2, cv2.LINE_AA)
				cv2.rectangle(imagen ,(x,y),(x+w,y+h),(0,0,255),2)
				#cv2.imshow("ROI_ENMARCADO",imagen)
				#Para mostrar el area de manera legible independiente del tamaño que tenga originalmente se redimensiona en 200x200. De esta manera si son pequeños los recortes, se puede tomar la desicion correcta. La condicionante solo actua en el caso de que p_corte = 0, ya que a veces w o h se vuelven 0.
				if not(recorte.shape[0] == 0 or recorte.shape[1] == 0):
					resized = cv2.resize(recorte, (200,200), interpolation = cv2.INTER_AREA)
				#Ahora aplicatemos la funcion que caracterizara a el ROI de todas las maneras estudiadas
				# ========================
				caracteristicas = fn.caracterizador(caracterizado,contornos[a-1],1,recorte,p_umbral)
				# ========================
				#Llegamos al punto en donde el usuario tendra a disposicion todas las pantallas y la información que hemos procesado hasta el momento y debera catalogar la zona que se esta mostrando como ROI.
				categoria,estado,final_ventana = fn.ventana_categoria(imagen, recorte)
				#categoria,estado,final_ventana = "No_Emisor",True,False
				#Pintamos en el frame que mostraremos en cada imagen catalogada, con diferentes colores, los tipos de emisor que se han catalogado.
				fn.pinta_categorias(final,categoria,x,y,w,h)
				if estado == True:
					#Guardamos el recorte como imagen
					nombre_recorte = "./Recortes/recorte_"+str(a)+"_"+nombre
					cv2.imwrite(nombre_recorte,recorte)
					#Almacenamos todas las caracteristicas obtenidas, los contornos, los puntos de las cajas, la categoria que el usuario ha dado al emisor y el nombre de la imagen desde la que provienen en el dataframe.
					#Tambien almacenamos el recorte del ROI original y el recorte del ROI redimensionado.
					# ========================
					diccionario_caracteristicas={}
					diccionario_caracteristicas['Categoria']=categoria
					#diccionario_caracteristicas['Nombre_Archivo_Original']=nombre
					#diccionario_caracteristicas['Nombre_Archivo_Recorte']=nombre_recorte
					diccionario_caracteristicas.update(caracteristicas)
					datos_entrenamiento = datos_entrenamiento.append(diccionario_caracteristicas, ignore_index=True)
					# ========================
					a+=1
					break
				if estado == False:
					a += 1
					break
				if final_ventana == True:
					break
			if final_ventana == True:
				break
		while(True):
			#imagen = cv2.resize(img, None, fx=0.8, fy=0.8)
			#cv2.imshow("Imagen",img)
			#cv2.imshow("Mascara",mascara)
			#cv2.imshow("segmentador_de_histograma",img)
			cv2.imshow("Final",final)
			if cv2.waitKey(1) & 0xFF == ord('s'):
				cv2.destroyAllWindows()
				print(datos_entrenamiento)
				break
			if cv2.waitKey(1) & 0xFF == ord('o'):
				terminar = True
				break
		if terminar == True:
			break
datos_entrenamiento.to_csv('datos_entrenamiento.csv')
