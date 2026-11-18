# -*- coding: utf-8 -*-
"""
celda_calibrada.py
==================
Funciones comunes para la celda de carga PSD-S1-2000kg + NAU7802 (Adafruit)
en la Raspberry Pi 4B.

Lo usan:
  * calibrar_celda_completa.py  -> genera calibracion_celda.json
  * tensile_pro.py              -> usa la calibración durante los ensayos

IMPORTANTE: el factor de calibración solo es válido con la MISMA configuración
del ADC (ganancia, LDO, velocidad). Por eso la configuración vive aquí, en un
único lugar, y se guarda dentro del JSON para verificarla al cargarlo.

Uso en tensile_pro.py:
    from celda_calibrada import configurar_nau7802, CeldaCalibrada
    nau = configurar_nau7802()
    celda = CeldaCalibrada(nau)
    celda.tarar()                                   # mordazas montadas, sin probeta
    F = celda.fuerza_N(celda.leer_cuentas(5))       # fuerza en N (tracción = +)
"""
import json
import math
import os
import statistics
import time

import board
from cedargrove_nau7802 import NAU7802

try:
    CARPETA = os.path.dirname(os.path.abspath(__file__))
except NameError:                      # por si se pega en la consola
    CARPETA = os.getcwd()
RUTA_CAL = os.path.join(CARPETA, "calibracion_celda.json")

# ----------------------------- Configuración del ADC -----------------------------
DIR_I2C = 0x2A
GANANCIA = 128
LDO = "3V0"            # excitación de la celda = 3.0 V (misma referencia del ADC)
SPS = 10               # muestras/s (10 SPS = menor ruido). Solo se aplica si la
                       # versión de la librería expone 'conversion_rate'.
FONDO_ESCALA_CUENTAS = 2 ** 23   # 24 bits con signo


# ----------------------------- Bajo nivel -----------------------------
def disponible(nau):
    """Compatible con 'available' como método o como propiedad."""
    a = nau.available
    return a() if callable(a) else bool(a)


def configurar_nau7802(ganancia=GANANCIA, ldo=LDO, sps=SPS, verbose=True):
    """Crea y configura el NAU7802 SIEMPRE igual (calibración y uso)."""
    i2c = board.I2C()
    nau = NAU7802(i2c, address=DIR_I2C, active_channels=1)
    nau.enable(True)
    try:
        nau.ldo_voltage = ldo
    except Exception as e:
        print("  [AVISO] No se pudo fijar el LDO a %s: %s" % (ldo, e))
    nau.gain = ganancia
    if hasattr(nau, "conversion_rate"):
        try:
            nau.conversion_rate = sps
        except Exception as e:
            print("  [AVISO] No se pudo fijar %d SPS: %s" % (sps, e))
    elif verbose:
        print("  [INFO] Esta versión de la librería no expone 'conversion_rate'; "
              "se usa la velocidad por defecto.")
    time.sleep(0.5)
    ok = calibrar_interno(nau)
    if verbose:
        print("  NAU7802: dir 0x%02X | ganancia %d | LDO %s | calibración interna: %s"
              % (DIR_I2C, ganancia, ldo, "OK" if ok else "FALLÓ"))
    return nau


def calibrar_interno(nau):
    """Calibración interna del AFE (entradas cortocircuitadas internamente).
    NO usar 'OFFSET' ni 'GAIN': esas toman la señal real de la celda."""
    try:
        ok = nau.calibrate("INTERNAL")
    except Exception as e:
        print("  [AVISO] calibrate('INTERNAL') lanzó: %s" % e)
        return False
    time.sleep(0.3)
    return bool(ok) if ok is not None else True


def leer_muestras(nau, segundos):
    """Lee durante 'segundos'. Devuelve lista de (t_relativo_s, cuentas)."""
    datos = []
    t0 = time.monotonic()
    t_ult = t0
    while True:
        ahora = time.monotonic()
        if ahora - t0 >= segundos:
            break
        if disponible(nau):
            datos.append((ahora - t0, nau.read()))
            t_ult = ahora
        else:
            if ahora - t_ult > 5.0:
                raise RuntimeError("El NAU7802 no entrega datos (revisa I2C/cableado).")
            time.sleep(0.002)
    return datos


def leer_cuentas(nau, n=5, timeout=5.0):
    """Promedio de n muestras consecutivas (cuentas)."""
    acum, cont = 0, 0
    t_ult = time.monotonic()
    while cont < n:
        if disponible(nau):
            acum += nau.read()
            cont += 1
            t_ult = time.monotonic()
        elif time.monotonic() - t_ult > timeout:
            raise RuntimeError("El NAU7802 no entrega datos (revisa I2C/cableado).")
        else:
            time.sleep(0.002)
    return acum / n


def estadisticas(datos, descartar_s=0.0, subbloques=6):
    """Estadística robusta de un bloque de lecturas.

    * Rechazo de atípicos: |x - mediana| > 5*MAD*1.4826.
    * u (incertidumbre tipo A) = max(error estándar, desviación de las medias de
      sub-bloques / sqrt(m)). Lo segundo es conservador cuando hay correlación
      entre muestras (filtro digital del ADC, deriva lenta).
    * pend = pendiente (cuentas/s) por mínimos cuadrados -> deriva.
    """
    pts = [(t, v) for (t, v) in datos if t >= descartar_s]
    if len(pts) < 5:
        raise RuntimeError("Muy pocas muestras (%d). Revisa el I2C/NAU7802." % len(pts))
    vals = [v for _, v in pts]
    med = statistics.median(vals)
    mad = 1.4826 * statistics.median([abs(v - med) for v in vals])
    lim = max(5.0 * mad, 5.0)
    ok = [(t, v) for (t, v) in pts if abs(v - med) <= lim]
    rech = len(pts) - len(ok)
    ts = [t for t, _ in ok]
    vs = [v for _, v in ok]
    n = len(vs)
    media = sum(vs) / n
    std = statistics.stdev(vs) if n > 1 else 0.0
    sem = std / math.sqrt(n)
    tm = sum(ts) / n
    den = sum((t - tm) ** 2 for t in ts)
    pend = (sum((t - tm) * (v - media) for t, v in zip(ts, vs)) / den) if den > 0 else 0.0
    u = sem
    if n >= subbloques * 5:
        tam = n // subbloques
        medias = [sum(vs[i * tam:(i + 1) * tam]) / tam for i in range(subbloques)]
        u = max(sem, statistics.stdev(medias) / math.sqrt(subbloques))
    return {"n": n, "rechazadas": rech, "media": media, "mediana": med,
            "std": std, "sem": sem, "u": u, "pend": pend,
            "min": min(vs), "max": max(vs)}


# ----------------------------- Uso de la calibración -----------------------------
class CeldaCalibrada:
    """Convierte cuentas del NAU7802 a fuerza (N). Tracción = positivo.

    fuerza_N      : usa la tara operativa (la que haces antes de cada ensayo).
    fuerza_cal_N  : usa el cero de la calibración (celda libre). Es solo una
                    referencia: tras reiniciar el ADC el cero puede moverse
                    algunas decenas de cuentas; para ensayos usa SIEMPRE tarar().
    """

    def __init__(self, nau, ruta=RUTA_CAL):
        with open(ruta, "r", encoding="utf-8") as f:
            self.cal = json.load(f)
        cfg = self.cal.get("config_adc", {})
        if (cfg.get("ganancia") != GANANCIA or cfg.get("ldo") != LDO):
            print("[AVISO] La configuración actual del ADC (ganancia %s, LDO %s) NO "
                  "coincide con la de la calibración (%s, %s). Recalibra."
                  % (GANANCIA, LDO, cfg.get("ganancia"), cfg.get("ldo")))
        self.nau = nau
        self.k = float(self.cal["cuentas_por_N"])          # con signo
        self.offset_cal = float(self.cal["offset_cuentas"])
        self.tara = self.offset_cal

    def leer_cuentas(self, n=5):
        return leer_cuentas(self.nau, n)

    def tarar(self, segundos=5.0):
        """Tara operativa: promedia 'segundos' con la carga muerta montada."""
        est = estadisticas(leer_muestras(self.nau, segundos))
        self.tara = est["media"]
        return est

    def fuerza_N(self, cuentas):
        return (cuentas - self.tara) / self.k

    def fuerza_cal_N(self, cuentas):
        return (cuentas - self.offset_cal) / self.k