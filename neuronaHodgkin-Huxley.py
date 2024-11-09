
## Alan Yair Cortes Lopez y Rodrigo García Núñez

# Importamos las bibliotecas necesarias
import matplotlib.pyplot as plt  # Para graficar resultados
import numpy as np  # Para operaciones numéricas

class Neurona:
    """El modelo HHModel rastrea las conductancias de 3 canales para calcular Vm (voltaje de membrana)"""

    # Constantes del modelo
    Cm = 1  # Capacitancia de membrana
    #ENa, EK, EKleak = 115, -12, 10.6  # Potenciales de reversión de iones
    ENa, EK, EKleak = 50, -77, -54.4 
    gNa, gK, gKleak = 120, 36, 0.3  # Conductancias de canales
    m, n, h = 0.05, 0.6, 0.32  # Puertas para los canales (n es de k, m es de Na y h es la probabilidad de que el canal de Na esté abierto)
    Vm=-65 #voltaje inicial de la membrana

    def alfasyBetas(self, Vm):
        nalfa = (0.01 * (Vm+55)) / (1-np.exp(-(Vm+55) / 10))
        nbeta = 0.125 * np.exp(-(Vm+65) / 80)
        malfa = (0.1 * (Vm+40)) / (1- np.exp(-(Vm+40) / 10))
        mbeta = 4 * np.exp(-(Vm+65) / 18)
        halfa = 0.07 * np.exp(-(Vm+65) / 20)
        hbeta = 1 / (1+ np.exp(-(Vm+35) / 10))
        return nalfa, nbeta, malfa, mbeta, halfa, hbeta

    def corrientesNaKL(self):
        """Calculamos las corrientes de los canales utilizando las últimas constantes de tiempo de las puertas"""
        # Corrientes de sodio (INa), potasio (IK) y fuga (IKleak)
        INa = np.power(self.m, 3) * self.gNa * self.h * (self.Vm - self.ENa)
        IK = np.power(self.n, 4) * self.gK * (self.Vm - self.EK)
        IKleak = self.gKleak * (self.Vm - self.EKleak)
        return INa, IK, IKleak

    def actualizarCompuertas(self, delta):
        # Actualizamos el estado de la puerta en función de las constantes alpha y beta
        resultado = self.alfasyBetas(self.Vm)
        nalfa, nbeta, malfa, mbeta, halfa, hbeta = resultado[0], resultado[1], resultado[2], resultado[3], resultado[4], resultado[5] 
        self.n +=  delta*(nalfa * (1 - self.n) - nbeta * (self.n))
        self.m +=  delta*(malfa * (1 - self.m) - mbeta * (self.m))
        self.h +=  delta*(halfa * (1 - self.h) - hbeta * (self.h))
        #return n, m, h
    
    def voltajes(self, estimulo, tmp):
        resultado = self.corrientesNaKL()
        INa, IK, IKleak = resultado[0], resultado[1], resultado[2]
        self.Vm += tmp*((estimulo - INa - IK - IKleak)/self.Cm)
        #return V
        
if __name__ == "__main__":
    # Creamos una instancia de la neuronita
    neuronita = Neurona()
    
    # Configuramos parámetros para la simulación
    pointCount = 100000
    voltajes = np.empty(pointCount)
    deltaT = 0.001
    times = np.arange(pointCount) * deltaT
    stim = np.zeros(pointCount)
    
    n = np.empty(pointCount)
    m = np.empty(pointCount)
    h = np.empty(pointCount)

    voltajes[0] = neuronita.Vm

    n[0] = neuronita.n
    m[0] = neuronita.m
    h[0] = neuronita.h
    stim[30000:50000] = 30  # Creamos un pulso cuadrado de estimulación

    # Realizamos la simulación en un bucle a lo largo del tiempo
    for i in range(1,len(times)):
        
        neuronita.actualizarCompuertas(deltaT)
        neuronita.voltajes(stim[i], deltaT)
        
        voltajes[i] = neuronita.Vm
        n[i] = neuronita.n
        m[i] = neuronita.m
        h[i] = neuronita.h
        print(voltajes[i], n[i], m[i], h[i])
        #input("Presiona Enter para continuar...")
    # Creamos gráficos para visualizar los resultados
    f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(8, 5), gridspec_kw={'height_ratios': [3, 3, 3]})

    ax1.plot(times, voltajes, 'b')
    ax1.set_ylabel("v[mV]")
    ax1.set_title("Modelo de Neurona de Hodgkin-Huxley", fontsize=16)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.tick_params(bottom=False)

    ax2.plot(times, stim, 'r')
    ax2.set_ylabel("I[micro A]")
    ax2.set_xlabel("t[ms]")
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)

    ax3.plot(times, n, 'b', label="n")
    ax3.plot(times, m, 'g', label="m")
    ax3.plot(times, h, 'r', label="h")
    plt.legend(loc='upper right')
    ax3.set_ylabel("act./inact")
    ax3.spines['right'].set_visible(False)
    ax3.spines['top'].set_visible(False)
    ax3.spines['bottom'].set_visible(False)
    ax3.tick_params(bottom=False)

    plt.margins(0, 0.1)
    plt.tight_layout()
    plt.show()
