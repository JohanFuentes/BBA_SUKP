import random
import math
import numpy as np
import pandas as pd

#Funcion para poder leer cada archivo (txt), donde cada uno corresponde a una instancia.
def leerArchivo(txt):
  fichero = open('Benchmarks\\'+txt)
  lineas = fichero.readlines()
  for linea in lineas:
      if linea == "\n":
        lineas.remove("\n")
  num_items = int(lineas[0])           #numero de items (entero)
  num_elementos = int(lineas[1])       #numero de elementos (entero)
  capacidad = int(lineas[2])           #capacidad de la mochila (entero)
  list_ganancia_items = (lineas[3].lstrip()).split(sep = " ") #lista de ganancia para cada item
  list_pesos_elementos = (lineas[4].lstrip()).split(sep = " ") #lista de peso para cada elemento
  
  try:
    list_pesos_elementos.remove("\n")
  except:
    print("")
  try:
    list_ganancia_items.remove("\n")
  except:
    print("")

  ganancia_items = np.array(list_ganancia_items)
  pesos_elementos = np.array(list_pesos_elementos)
  matrizItemElementos = np.empty((num_items,num_elementos)) #matriz que asocia los items con los elementos

  for j in range(num_items):
    line = lineas[j+5].split(sep = " ")
    
    for i in line:
      if i == "\n":
        line.remove("\n")
    matrizItemElementos[j:j+1] = line #Llenando la matriz
  
  mit = matrizItemElementos.copy()
  return ganancia_items,pesos_elementos,mit,capacidad


#Funcion para generar las posiciones iniciales de cada murcielago, de manera random, dandole mas prioridad al valor 0, para evitar infactibilidad
def generar_soluciones(cantidad, largo):
    vectores = []
    for _ in range(cantidad):
        vector = [random.choice([0,0,0,0,0,0,0,1]) for _ in range(largo)]
        vectores.append(vector)
    return np.array(vectores)

#Funcion de trannsferencia, a partir de la velocidad de un murcielago retorna un valor correspondiente a una probabilidad ([0,1]).
def funcionTransferencia(velocidad):
    phi = math.pi
    return abs(2*math.atan(phi*velocidad/2)/phi)    

#Funcion que cambia el valor de un bit, de acuerdo a la velocidad, mientras mas alta la velocidad, mas probabilidad tiene de cambiarse.
def actualizarPosicion(velocidad, posicion):
    rand = random.random()
    prob = funcionTransferencia(velocidad)
    if(rand < prob):
        if(posicion):
            posicion = 0
        else:
          #Se agrega esta probabilidad para que se menos la probabilidad de pasar de 0 a 1 (temas de posible infactibilidad)
          if(random.random()>0.6):
            posicion = 1
    return posicion

#Sirve llevar el valor de la velocidad a un rango entre 0 y 5, ya que si no daria valores demasiado grandes
def normalizarVelocidad(velocidad,limiteSuperior):
    return velocidad*5/limiteSuperior #velocidad de 0 a 5

#Actualiza la velocidad, a partir de la frecuencia [0,1], y la diferencia entre el valor de la funcion objetivo de lo mejor encontrado menos el de la actual
def actualizarVelocidad(mejor,actual,limiteSuperior):
    #fmin = 0
    #fmax = 1
    #mejor y actual es el valor de la funcion objetivo
    velocidad = random.random()*(mejor-actual)
    velocidad = normalizarVelocidad(velocidad,limiteSuperior)
    return velocidad

#Utilizada para calcular un valor para normalizar la velocidad
def sumar_elementos(arreglo):
    suma = np.sum(arreglo)
    return int(suma)

#A partir de una lista entrega el valor de la funcion objetivo (no arregla solucion si es infactible)
def sumaTotal(aux,matrizItemElementos,list_ganancia_items): 
  suma_ga = 0
  indices = np.where(aux == 1)[0]
  # Convertir los índices en una lista
  lista_indices = indices.tolist()
  for i in lista_indices:
    suma_ga = suma_ga + int(list_ganancia_items[int(i)])
  return suma_ga

#Retorna el peso total de una solucion
def pesoTotal(aux,matrizItemElementos,list_pesos_elementos):
  sumaPeso = 0
  setIndicesPesos = set()
  indices = np.where(aux == 1)[0]

  # Convertir los índices en una lista
  lista_indices = indices.tolist()

  for i in lista_indices:
    indicesElementos = np.where(matrizItemElementos[int(i)] == 1)[0]
    for j in indicesElementos:
      setIndicesPesos.add(j)

  for k in setIndicesPesos:
    sumaPeso = sumaPeso + int(list_pesos_elementos[int(k)])
  return sumaPeso

#Inicializa las velocidades para cada solucion, con valores entre 0 y 5
def inicializarVelocidades(largo):
    array = np.random.uniform(0, 5, largo)
    return array

#Inicializa el valor de las amplitudes (a) o el de la tasa de emision (r)
def inicializar_r_a(valor, largo):
    array = np.full(largo, valor)
    return array

#Dentro de un conjunto de soluciones, devuelve el que tenga mejor valor de funcion objetivo (arreglando las soluciones infactibles)
def mejor(listaAux,matrizItemElementos,list_ganancia_items,list_pesos_elementos,capacidad):
  mejor = np.zeros(len(list_ganancia_items))
  mayorGanancia = 0
  for solucion in listaAux:
    peso = pesoTotal(solucion,matrizItemElementos,list_pesos_elementos)
    if(peso>capacidad):
      solucion = arreglarSolucion(solucion,matrizItemElementos,list_pesos_elementos,capacidad)
    ganancia = sumaTotal(solucion,matrizItemElementos,list_ganancia_items)
    if(ganancia>mayorGanancia):
      mayorGanancia = ganancia
      mejor = solucion
  return mejor,mayorGanancia

#A partir de una solucion, entrega el valor de la funcion objetivo (arreglando las soluciones infactibles)
def funcionObjetivo(solucion,matrizItemElementos,list_ganancia_items,list_pesos_elementos,capacidad):
  peso = pesoTotal(solucion,matrizItemElementos,list_pesos_elementos)
  if(peso>capacidad):
    #Funcion arreglar
    solucion = arreglarSolucion(solucion,matrizItemElementos,list_pesos_elementos,capacidad)
  ganancia = sumaTotal(solucion,matrizItemElementos,list_ganancia_items)
  return ganancia 

#A partir de una solucion infactible, devuelve una solucion factible (quitando items aleatoriamente, hasta que sea factible)
def arreglarSolucion(solucion,matrizItemElementos,list_pesos_elementos,capacidad):
  while(pesoTotal(solucion,matrizItemElementos,list_pesos_elementos) > capacidad):
    indices = np.where(solucion == 1)[0]

    if len(indices) > 0:
      # Seleccionar aleatoriamente un índice con valor de 1
      indice_seleccionado = np.random.choice(indices)
      # Cambiar el valor del índice seleccionado a 0
      solucion[indice_seleccionado] = 0
  return solucion

#A partir de una probabilidad (a), cambia el valor de un bit de la solucion actual, por el de la mejor solucion
def randomWalk(actual,mejorPosicion,prob):
  for indice in range(len(actual)):
    if(random.random()<prob):
      #Cambiar bit por el del mejor
      actual[indice] = mejorPosicion[indice]
  return actual

#A partir de una solucion, se le cambian aleatoriamente algunos bits (acorde a una probabilidad), dando mas prioridad a cambiar 1s por 0s
def aleatorio(actual):
  for indice in range(len(actual)):
    if(random.random()<0.1):
      #Cambiar bit por el valor contrario
      if(actual[indice]):
        actual[indice] = 0
      else:
        if(random.random()>0.6):
          actual[indice] = 1
  return actual