import math
import numpy as np
import random
import pandas as pd
from FUNCIONES_BBA_SUKP import *
import time

listaTxt = list([
"85_100_0.10_0.75.txt"
,"85_100_0.15_0.85.txt"
,"100_100_0.10_0.75.txt"
,"100_100_0.15_0.85.txt"
,"100_85_0.10_0.75.txt"
,"100_85_0.15_0.85.txt"
,"185_200_0.10_0.75.txt"
,"185_200_0.15_0.85.txt"
,"200_185_0.10_0.75.txt"
,"200_185_0.15_0.85.txt"
,"200_200_0.10_0.75.txt"
,"200_200_0.15_0.85.txt"
,"285_300_0.10_0.75.txt"
,"285_300_0.15_0.85.txt"
,"300_285_0.10_0.75.txt"
,"300_285_0.15_0.85.txt"
,"300_300_0.10_0.75.txt"
,"300_300_0.15_0.85.txt"
,"385_400_0.10_0.75.txt"
,"385_400_0.15_0.85.txt"
,"400_385_0.10_0.75.txt"
,"400_385_0.15_0.85.txt"
,"400_400_0.10_0.75.txt"
,"400_400_0.15_0.85.txt"
,"485_500_0.10_0.75.txt"
,"485_500_0.15_0.85.txt"
,"500_485_0.10_0.75.txt"
,"500_485_0.15_0.85.txt"
,"500_500_0.10_0.75.txt"
,"500_500_0.15_0.85.txt"])

"""
PSEUDOCODIGO BBA

Initialize the bat population: Xi (i = 1, 2, ... , n)=rand(0 or 1) and Vi=0
Define pulse frequency Fi
Initialize pulse rates ri and the loudness Ai
while (t < Max number of iterations)
    Adjusting frequency and updating velocities
    Calculate transfer function value using equation (9)
    Update positions using equation (10)
if (rand > ri)
    Select a solution (Gbest) among the best solutions randomly
    Change some of the dimensions of position vector with some of the dimensions of Gbest
end if
    Generate a new solution by flying randomly
if (rand < Ai & f(xi) < f(Gbest))
    Accept the new solutions
    Increase ri and reduce Ai
end if
    Rank the bats and find the current Gbest
end while 
"""
listaIter = list() #Lista para guardar las ultima iteracion donde hubo mejora, de cada instancia 
listaGananciaTotal = list() #Lista para guardar el ultimo valor mejorarado, de cada instancia
listaTiempos = list() #Guarda los tiempos de ejecucion en segundos, para cada instancia
listaTodasIteraciones = list() #Guarda todas las iteraciones donde mejoro la solucion, para cada instancia
listaTodasGanancias = list() #Guarda los valores que fueron mejorando, para cada instancia

for txt in listaTxt:
    inicio = time.time()
    print("#######################################",txt,"#########################################")
    ganancia_items,pesos_elementos,matriz,capacidad = leerArchivo(txt) #Leer archivo txt
    ganancia_items = ganancia_items.astype(int)
    pesos_elementos = pesos_elementos.astype(int)

    largo_poblacion = int(len(ganancia_items)/2) #Tamaño de poblacion dinamico
    largo_poblacion = 10 #Tamaño de poblacion fijo
    posiciones_iniciales = generar_soluciones(largo_poblacion, len(ganancia_items)) #Posiciones iniciales
    limiteSuperior = int(sumar_elementos(ganancia_items)/(len(ganancia_items)/40)) #Necesario para normalizar las velocidades
    velocidades = inicializarVelocidades(largo_poblacion) # np array con las velocidades iniciales de manera random entre 0 y 5
    r = 0.001 #tasa de emision
    a = 0.9 #amplitud
    array_r = inicializar_r_a(r, largo_poblacion)
    array_a = inicializar_r_a(a, largo_poblacion)
    alfa = 0.9
    gama = 0.02
    iteracionesMax = 3000
    iteracion = 0

    mejorPosicion,mayorGanancia = mejor(posiciones_iniciales,matriz.copy(),ganancia_items.copy(),pesos_elementos.copy(),capacidad)
    mejorPosicionGlobal, mayorGananciaGlobal = mejorPosicion,mayorGanancia

    ultimaIteracion = 0 #Guarda el valor de la iteracion donde por ultima vez se mejoro el valor de la funcion objetivo
    listaIt = list() #Guarda las iteraciones donde se mejoro el valor de la funcion objetivo
    listaGa = list() #Guarda los valores donde se mejoro el valor de la funcion objetivo
    while(iteracion<iteracionesMax):
        band = False #Permite saber si mejoro el valor de la funcion objetivo en la iteracion actual
        #Recorriendo todos los muercielagos
        for indice in range(len(posiciones_iniciales)):
            copiaPosicion = posiciones_iniciales[indice].copy()
            if(random.random()<array_r[indice]):
                #Random Walk
                #Cambiando los bits de la solucion actual con cierta probabilidad a, por los bits de la mejor solucion
                copiaPosicion = randomWalk(copiaPosicion,mejorPosicionGlobal,array_a[indice])
            #Cambio Aleatorio
            copiaPosicion = aleatorio(copiaPosicion)
            #Ver si se aceptan las nuevas soluciones (si mejoran) y si se cumple rand > a_i
            actual = funcionObjetivo(posiciones_iniciales[indice],matriz.copy(),ganancia_items.copy(),pesos_elementos.copy(),capacidad)
            modificado = funcionObjetivo(copiaPosicion,matriz.copy(),ganancia_items.copy(),pesos_elementos.copy(),capacidad)

            if((random.random() > array_a[indice]) and (actual < modificado)):
                #Se acepta la nueva solucion
                posiciones_iniciales[indice] = copiaPosicion
                #Se actualiza r y a
                array_r[indice] = 1-math.exp(-gama*(iteracion+0.001))
                array_a[indice] = array_a[indice]*alfa
            #else: #Opcional, por si se quiere aceptar una solucion que no cumpla con la condicion anterior (podria ayudar en la exploracion)
                #if(random.random()<0.1):
                    #posiciones_iniciales[indice] = copiaPosicion
        
        #Calcular el mejor
        mejorPosicion,mayorGanancia = mejor(posiciones_iniciales.copy(),matriz.copy(),ganancia_items.copy(),pesos_elementos.copy(),capacidad)
        #Ver si alguno supera al mejor global
        if(mayorGanancia>mayorGananciaGlobal):
            mejorPosicionGlobal = mejorPosicion
            mayorGananciaGlobal = mayorGanancia
            ultimaIteracion = iteracion
            band = True

        #Actualizar velocidad y posicion para cada solucion
        for indice in range(len(posiciones_iniciales)):
            gananciaActual = funcionObjetivo(posiciones_iniciales[indice],matriz.copy(),ganancia_items.copy(),pesos_elementos.copy(),capacidad)
            copiaPi = posiciones_iniciales[indice].copy()
            for i in range(len(ganancia_items)):
                velocidad = actualizarVelocidad(mayorGananciaGlobal,gananciaActual,limiteSuperior)
                copiaPi[i] = actualizarPosicion(velocidad, copiaPi[i])
            gan2Pos = funcionObjetivo(copiaPi,matriz.copy(),ganancia_items.copy(),pesos_elementos.copy(),capacidad)
            #Aceptarla el movimiento si mejora el valor de la funcion objetivo
            if(gan2Pos > gananciaActual):
                posiciones_iniciales[indice] = copiaPi
            #else: #Opcional, por si se quiere aceptar una solucion que no cumpla con la condicion anterior (podria ayudar en la exploracion)
                #if(random.random()<0.1):
                    #posiciones_iniciales[indice] = copiaPi

        #Calcular el mejor
        mejorPosicion,mayorGanancia = mejor(posiciones_iniciales.copy(),matriz.copy(),ganancia_items.copy(),pesos_elementos.copy(),capacidad)
        #Ver si alguno supera al mejor global
        if(mayorGanancia>mayorGananciaGlobal):
            mejorPosicionGlobal = mejorPosicion
            mayorGananciaGlobal = mayorGanancia
            ultimaIteracion = iteracion
            band = True
        
        if(band):
            listaIt.append(iteracion)
            listaGa.append(mayorGananciaGlobal)

        iteracion = iteracion + 1

    fin = time.time()
    tiempo_transcurrido = fin - inicio #tiempo en segundos
    print(mayorGananciaGlobal,mejorPosicionGlobal)
    listaIter.append(ultimaIteracion)
    listaGananciaTotal.append(mayorGananciaGlobal)
    listaTiempos.append(tiempo_transcurrido)
    listaTodasIteraciones.append(listaIt)
    listaTodasGanancias.append(listaGa)

datosExportar = pd.DataFrame({'Txt' : listaTxt,'Iteracion Final' : listaIter,'Ganancia Total': listaGananciaTotal, 'tiempo(segundos)':listaTiempos})
result = pd.ExcelWriter('ResultadosBat1.xlsx')  
datosExportar.to_excel(result)
result.save()

datosExportar = pd.DataFrame({'Txt' : listaTxt,'Iteraciones' : listaTodasIteraciones,'Ganancias': listaTodasGanancias})
result = pd.ExcelWriter('ResultadosBat1Convergencia.xlsx')  
datosExportar.to_excel(result)
result.save()