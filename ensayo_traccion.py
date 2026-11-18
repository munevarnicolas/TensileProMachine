# -*- coding: utf-8 -*-
"""
ensayo_traccion.py
==================
Ensayo de tracción (ASTM D638) en la Raspberry Pi 4B:

  * FUERZA      : celda PSD-S1-2000kg + NAU7802, con calibracion_celda.json
  * DEFORMACIÓN : extensómetro de video (Arducam IMX477 + picamera2 + OpenCV).
                  Sigue DOS marcas sobre la zona calibrada con correlación
                  cruzada normalizada y precisión sub-píxel.
  * ACTUADOR    : Bevor + BTS7960 (pines de tensile_pro.py), con rampa suave.

Guarda en  ensayos/<fecha>_<nombre>/ :
    datos.csv          (t, fuerza, deformación, posiciones de marcas, estado...)
    fuerza_cruda.csv   (fuerza a la velocidad nativa del NAU7802)
    metadatos.json     (probeta, calibración, cámara, parámetros)
    inicio.jpg / fin.jpg

Luego ejecuta  curva_esfuerzo_deformacion.py  para obtener la curva y las propiedades.

Requiere en la misma carpeta: celda_calibrada.py y calibracion_celda.json
Paquetes (apt):  python3-picamera2  python3-opencv  python3-numpy  python3-gpiozero

DEFORMACIÓN INGENIERIL  e = (L - L0) / L0   con L = distancia entre las dos marcas.
Es independiente de la escala (mm/px); la escala solo se usa para mostrar mm/min.
"""
import csv
import json
import math
import os
import threading
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np
from gpiozero import Button, OutputDevice, PWMOutputDevice
from picamera2 import Picamera2

from celda_calibrada import (CARPETA, RUTA_CAL, CeldaCalibrada,
                             configurar_nau7802, disponible)

# ----------------------------- Pines (BCM) de tensile_pro.py -----------------------------
PIN_RPWM = 18          # retraer
PIN_LPWM = 13          # extender
PIN_R_EN = 23
PIN_L_EN = 24
PIN_STOP = 22          # pulsador de paro
USAR_BOTON_STOP = True
FREC_PWM = 1000        # Hz: usa el MISMO valor que en tu código del actuador que ya funciona

# ----------------------------- Cámara -----------------------------
CAM_ANCHO, CAM_ALTO, CAM_FPS = 2028, 1520, 30   # modo 2x2 del IMX477 (hasta 50 fps)
EXPOSICION_US = None   # None = congela lo que calculó la auto-exposición al inicio
CAM_GANANCIA = None    # idem
POSICION_LENTE = None  # solo si tu lente es de foco motorizado (dioptrías)
ESCALA_VISTA = 0.5     # tamaño de la vista previa
VISTA_CADA_N = 6       # refrescar la ventana cada N cuadros

# ----------------------------- Seguimiento de marcas -----------------------------
BUSQUEDA_PX = 60       # radio de búsqueda alrededor de la última posición
MIN_SCORE = 0.50       # correlación mínima aceptable
REFRESCAR_SCORE = 0.80 # si baja de esto, se refresca la plantilla (sub-píxel)
BLUR_SIGMA = 1.0
N_REPOSO = 60          # cuadros en reposo para medir L0 y el ruido
MAX_PERDIDOS = 15      # cuadros seguidos sin seguimiento -> detener
ROI_MAX_PX = 240       # lado máximo del cuadro de una marca (px, resolución completa)
ROI_SEP_MIN_PX = 150   # separación mínima entre los centros de las dos marcas (px)

# ----------------------------- Ensayo -----------------------------
RAMPA_S = 0.5          # arranque suave (0 = arranque directo al valor de VELOCIDAD)
# SIN límite de fuerza de ensayo: el ensayo sigue hasta la ROTURA (o STOP / q / estancamiento).
# Solo queda una protección de la CELDA (no es un límite del ensayo): 120 % de la capacidad,
# por debajo de su sobrecarga segura (150 %). Una probeta de PLA/PETG rompe a ~2-3 kN, muy lejos.
PROTEGER_CELDA = True
F_PROTECCION_CELDA_N = 1.2 * 2000.0 * 9.80665
DEF_MAX = 10.0         # 1000 %: prácticamente sin límite
SIGNO_TRACCION = +1    # en esta máquina la tracción de la probeta da fuerza POSITIVA (verificado)
UMBRAL_FUERZA_DIRECCION_N = 30.0   # fuerza opuesta a la tracción mayor que esto -> dirección incorrecta
PARAR_SI_SE_PIERDE_SEGUIMIENTO = False   # False: sigue registrando fuerza hasta romper
DETECTAR_ESTANCAMIENTO = True      # fin de carrera / bloqueo: ni fuerza ni deformación cambian
VENTANA_ESTANC_S = 5.0
UMBRAL_ESTANC_DEF = 0.0005         # 0.05 %
FRAC_ROTURA = 0.5      # rotura: la fuerza cae por debajo de este % del máximo...
F_MIN_ROTURA_N = 30.0  # ...y el máximo superó este valor
POST_ROTURA_S = 1.0
T_MAX_S = 1800.0
COMPROBAR_ACERCAMIENTO = False  # True = parar si la deformación baja de UMBRAL (por video); por defecto solo avisa
UMBRAL_ACERCAMIENTO = -0.01     # deformación (fracción) por debajo de la cual se considera acercamiento
EDAD_FUERZA_MAX_S = 2.0


# ----------------------------- Utilidades -----------------------------
def pedir_float(msg, defecto, minimo, maximo):
    while True:
        txt = input("%s [%s]: " % (msg, defecto)).strip().replace(",", ".")
        if txt == "":
            return float(defecto)
        try:
            v = float(txt)
        except ValueError:
            print("  Valor no válido.")
            continue
        if minimo <= v <= maximo:
            return v
        print("  Debe estar entre %s y %s." % (minimo, maximo))


def pedir_texto(msg, defecto):
    t = input("%s [%s]: " % (msg, defecto)).strip()
    return t if t else defecto


def pedir_opcion(msg, validas):
    while True:
        t = input("%s: " % msg).strip().upper()
        if t in validas:
            return t
        print("  Escribe una de: %s" % ", ".join(validas))


# ----------------------------- Cámara -----------------------------
class Camara:
    def __init__(self):
        self.cam = Picamera2()
        dur = int(1e6 / CAM_FPS)
        cfg = self.cam.create_video_configuration(
            main={"size": (CAM_ANCHO, CAM_ALTO), "format": "YUV420"},
            controls={"FrameDurationLimits": (dur, dur)},
            buffer_count=4)
        self.cam.configure(cfg)
        self.cam.start()
        time.sleep(1.5)                       # asienta la auto-exposición
        self.exposicion_us = None
        self.ganancia = None
        self._fijar_controles()

    def _fijar_controles(self):
        """Congela exposición/ganancia/balance: el brillo debe ser constante
        durante el ensayo o la correlación se degrada."""
        try:
            md = self.cam.capture_metadata()
            self.exposicion_us = int(EXPOSICION_US or md["ExposureTime"])
            self.ganancia = float(CAM_GANANCIA or md["AnalogueGain"])
            ctrl = {"AeEnable": False, "AwbEnable": False,
                    "ExposureTime": self.exposicion_us, "AnalogueGain": self.ganancia}
            if "ColourGains" in md:
                ctrl["ColourGains"] = md["ColourGains"]
            self.cam.set_controls(ctrl)
        except Exception as e:
            print("  [AVISO] No se pudo congelar la exposición: %s" % e)
        if POSICION_LENTE is not None:
            try:
                self.cam.set_controls({"AfMode": 0, "LensPosition": POSICION_LENTE})
            except Exception as e:
                print("  [AVISO] No se pudo fijar el foco: %s" % e)

    def leer_gris(self):
        """Plano Y (luminancia) de YUV420: gris sin conversión de color."""
        arr = self.cam.capture_array("main")
        return arr[:CAM_ALTO, :CAM_ANCHO]

    def cerrar(self):
        try:
            self.cam.stop()
            self.cam.close()
        except Exception:
            pass


# ----------------------------- Seguimiento de una marca -----------------------------
class Seguidor:
    """Correlación cruzada normalizada con refinamiento parabólico (sub-píxel)."""

    def __init__(self, gris, roi):
        x, y, w, h = [int(v) for v in roi]
        self.w, self.h = w, h
        self.cx = x + (w - 1) / 2.0          # centro de la plantilla (px, sub-píxel)
        self.cy = y + (h - 1) / 2.0
        self.plantilla = self._extraer(gris, self.cx, self.cy)
        self.refrescos = 0
        self.score = 1.0
        self.perdido = False

    def _extraer(self, gris, cx, cy):
        p = cv2.getRectSubPix(gris, (self.w, self.h), (cx, cy))
        return cv2.GaussianBlur(p, (0, 0), BLUR_SIGMA)

    def seguir(self, gris):
        H, W = gris.shape[:2]
        px = int(round(self.cx - (self.w - 1) / 2.0))
        py = int(round(self.cy - (self.h - 1) / 2.0))
        x0 = max(0, px - BUSQUEDA_PX)
        y0 = max(0, py - BUSQUEDA_PX)
        x1 = min(W, px + self.w + BUSQUEDA_PX)
        y1 = min(H, py + self.h + BUSQUEDA_PX)
        ventana = cv2.GaussianBlur(gris[y0:y1, x0:x1], (0, 0), BLUR_SIGMA)
        if ventana.shape[0] < self.h or ventana.shape[1] < self.w:
            self.perdido = True
            return self.cx, self.cy, 0.0
        res = cv2.matchTemplate(ventana, self.plantilla, cv2.TM_CCOEFF_NORMED)
        _, maxv, _, (mx, my) = cv2.minMaxLoc(res)
        dx = dy = 0.0
        if 0 < mx < res.shape[1] - 1:
            a, b, c = res[my, mx - 1], res[my, mx], res[my, mx + 1]
            d = a - 2 * b + c
            if abs(d) > 1e-9:
                dx = float(np.clip(0.5 * (a - c) / d, -0.5, 0.5))
        if 0 < my < res.shape[0] - 1:
            a, b, c = res[my - 1, mx], res[my, mx], res[my + 1, mx]
            d = a - 2 * b + c
            if abs(d) > 1e-9:
                dy = float(np.clip(0.5 * (a - c) / d, -0.5, 0.5))
        borde = (mx == 0 or my == 0 or mx == res.shape[1] - 1 or my == res.shape[0] - 1)
        self.cx = x0 + mx + dx + (self.w - 1) / 2.0
        self.cy = y0 + my + dy + (self.h - 1) / 2.0
        self.score = float(maxv)
        self.perdido = bool(self.score < MIN_SCORE or borde)
        if MIN_SCORE <= self.score < REFRESCAR_SCORE and not self.perdido:
            self.plantilla = self._extraer(gris, self.cx, self.cy)
            self.refrescos += 1
        return self.cx, self.cy, self.score


# ----------------------------- Fuerza (hilo) -----------------------------
class LectorFuerza(threading.Thread):
    """Lee el NAU7802 sin parar. Solo este hilo usa el I2C durante el ensayo."""

    def __init__(self, nau, celda):
        super().__init__(daemon=True)
        self.nau, self.celda = nau, celda
        self.parar_ev = threading.Event()
        self.ultimo = None          # (t_monotonic, F_N, cuentas)
        self.registro = []          # (t_monotonic, cuentas, F_N)
        self.error = None

    def run(self):
        try:
            while not self.parar_ev.is_set():
                if disponible(self.nau):
                    c = self.nau.read()
                    t = time.monotonic()
                    f = self.celda.fuerza_N(c)
                    self.ultimo = (t, f, c)
                    self.registro.append((t, c, f))
                else:
                    time.sleep(0.002)
        except Exception as e:
            self.error = e


# ----------------------------- Actuador -----------------------------
class Actuador:
    def __init__(self):
        self.rpwm = PWMOutputDevice(PIN_RPWM, frequency=FREC_PWM)
        self.lpwm = PWMOutputDevice(PIN_LPWM, frequency=FREC_PWM)
        self.r_en = OutputDevice(PIN_R_EN)
        self.l_en = OutputDevice(PIN_L_EN)
        self.sentido = None
        self.duty = 0.0
        self.actual = 0.0

    def iniciar(self, sentido, duty):
        self.sentido, self.duty = sentido, duty

    def actualizar(self, t):
        """Rampa lineal de 0 a duty en RAMPA_S segundos."""
        d = self.duty * min(1.0, max(0.0, t / RAMPA_S))
        self.r_en.on()
        self.l_en.on()
        if self.sentido == "E":
            self.rpwm.value = 0.0
            self.lpwm.value = d
        else:
            self.lpwm.value = 0.0
            self.rpwm.value = d
        self.actual = d

    def parar(self):
        for dev in (self.rpwm, self.lpwm):
            try:
                dev.value = 0.0
            except Exception:
                pass
        for dev in (self.r_en, self.l_en):
            try:
                dev.off()
            except Exception:
                pass
        self.actual = 0.0

    def cerrar(self):
        self.parar()
        for dev in (self.rpwm, self.lpwm, self.r_en, self.l_en):
            try:
                dev.close()
            except Exception:
                pass


# ----------------------------- Interfaz de selección de marcas -----------------------------
HAY_GUI = True


def mostrar(titulo, img):
    global HAY_GUI
    if not HAY_GUI:
        return -1
    try:
        cv2.imshow(titulo, img)
        return cv2.waitKey(1) & 0xFF
    except cv2.error:
        HAY_GUI = False
        print("  [AVISO] No hay ventana gráfica disponible (¿VNC/pantalla?).")
        return -1


def elegir_rois(cam):
    print("\nVista previa: encuadra la zona calibrada. ESPACIO = congelar, q = salir.")
    while True:
        gris = cam.leer_gris()
        vista = cv2.resize(gris, None, fx=ESCALA_VISTA, fy=ESCALA_VISTA)
        k = mostrar("Vista previa (ESPACIO congela)", vista)
        if k == ord(" "):
            break
        if k == ord("q"):
            return None, None
    base = gris.copy()
    vista = cv2.resize(base, None, fx=ESCALA_VISTA, fy=ESCALA_VISTA)
    vista_c = cv2.cvtColor(vista, cv2.COLOR_GRAY2BGR)
    rois = []
    i = 1
    while i <= 2:
        print("Selecciona la MARCA %d: cuadro SOLO alrededor del punto (unos 3 diámetros) "
              "y pulsa ENTER (c = cancelar)." % i)
        r = cv2.selectROI("Marca %d" % i, vista_c, False)
        if r[2] < 8 or r[3] < 8:
            print("  Selección cancelada o demasiado pequeña.")
            return None, None
        roi = tuple(int(v / ESCALA_VISTA) for v in r)
        if max(roi[2], roi[3]) > ROI_MAX_PX:
            print("  [!] Cuadro demasiado grande (%d x %d px, máx. %d). Rodea SOLO el punto, sin bordes "
                  "de mordaza ni de la probeta. Repite la MARCA %d." % (roi[2], roi[3], ROI_MAX_PX, i))
            continue
        if i == 2:
            c1 = (rois[0][0] + rois[0][2] / 2.0, rois[0][1] + rois[0][3] / 2.0)
            c2 = (roi[0] + roi[2] / 2.0, roi[1] + roi[3] / 2.0)
            if math.hypot(c2[0] - c1[0], c2[1] - c1[1]) < ROI_SEP_MIN_PX:
                print("  [!] Las dos marcas quedaron casi en el mismo sitio (%.0f px). Selecciona la OTRA "
                      "marca (la de abajo si la 1 fue la de arriba). Repite la MARCA 2."
                      % math.hypot(c2[0] - c1[0], c2[1] - c1[1]))
                continue
        cv2.rectangle(vista_c, (r[0], r[1]), (r[0] + r[2], r[1] + r[3]), (0, 255, 0), 1)
        rois.append(roi)
        i += 1
    cv2.destroyAllWindows()
    return base, rois


def dibujar(gris, seg1, seg2, texto):
    vista = cv2.resize(gris, None, fx=ESCALA_VISTA, fy=ESCALA_VISTA)
    vista = cv2.cvtColor(vista, cv2.COLOR_GRAY2BGR)
    for s, col in ((seg1, (0, 255, 0)), (seg2, (0, 200, 255))):
        c = (int(s.cx * ESCALA_VISTA), int(s.cy * ESCALA_VISTA))
        m = (int(s.w * ESCALA_VISTA / 2), int(s.h * ESCALA_VISTA / 2))
        cv2.rectangle(vista, (c[0] - m[0], c[1] - m[1]), (c[0] + m[0], c[1] + m[1]),
                      (0, 0, 255) if s.perdido else col, 1)
    cv2.line(vista, (int(seg1.cx * ESCALA_VISTA), int(seg1.cy * ESCALA_VISTA)),
             (int(seg2.cx * ESCALA_VISTA), int(seg2.cy * ESCALA_VISTA)), (255, 255, 0), 1)
    cv2.putText(vista, texto, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return vista


# ----------------------------- Programa -----------------------------
def main():
    print("=" * 70)
    print(" ENSAYO DE TRACCIÓN: fuerza (celda calibrada) + deformación (video)")
    print("=" * 70)
    print("Preparación de la probeta y la cámara:")
    print(" * Dibuja DOS marcas finas perpendiculares al eje, dentro de la sección")
    print("   estrecha, separadas la longitud calibrada (D638 tipo I: 50 mm).")
    print("   Marcador blanco sobre probeta oscura, negro sobre clara.")
    print(" * Cámara perpendicular a la probeta, probeta enfocada, luz difusa y fija.")
    print(" * Llena el encuadre con la zona calibrada (más px/mm = más resolución).\n")

    nombre = pedir_texto("Nombre del ensayo", "probeta_1")
    ancho = pedir_float("Ancho de la sección estrecha, medido (mm)", 13.0, 1.0, 100.0)
    espesor = pedir_float("Espesor, medido (mm)", 3.2, 0.1, 50.0)
    l0_mm = pedir_float("Distancia REAL entre las dos marcas (mm)", 25.0, 5.0, 500.0)
    sentido = pedir_opcion("¿Qué comando del actuador ESTIRA la probeta? [E=extender / R=retraer]",
                           ["E", "R"])
    duty = pedir_float("Velocidad del actuador 0.1 a 1.0 (1.0 = 100 %)", 1.0, 0.1, 1.0)
    f_max = F_PROTECCION_CELDA_N if PROTEGER_CELDA else float("inf")
    def_max = DEF_MAX
    area = ancho * espesor
    print("\nÁrea inicial A0 = %.2f mm2" % area)

    if not os.path.exists(RUTA_CAL):
        raise RuntimeError("No existe calibracion_celda.json en %s" % CARPETA)

    carpeta_ens = os.path.join(CARPETA, "ensayos",
                               "%s_%s" % (datetime.now().strftime("%Y%m%d_%H%M%S"), nombre))
    os.makedirs(carpeta_ens, exist_ok=True)

    boton = None
    if USAR_BOTON_STOP:
        boton = Button(PIN_STOP, pull_up=True)
        if boton.is_pressed:
            raise RuntimeError("El pulsador de paro (GPIO %d) aparece presionado al inicio. "
                               "Revisa el cableado o pon USAR_BOTON_STOP=False." % PIN_STOP)

    print("\nConfigurando celda de carga (misma configuración que la calibración)...")
    nau = configurar_nau7802()
    celda = CeldaCalibrada(nau)

    print("\nIniciando cámara...")
    cam = Camara()
    act = Actuador()
    lector = None
    try:
        base, rois = elegir_rois(cam)
        if rois is None:
            print("Cancelado.")
            return
        seg1, seg2 = Seguidor(base, rois[0]), Seguidor(base, rois[1])

        input("\nProbeta sujeta SIN tensión. ENTER para tarar la celda (5 s, no toques nada)... ")
        est = celda.tarar(5.0)
        print("  Tara: %.1f cuentas | ruido %.2f N" % (celda.tara, est["std"] / abs(celda.k)))

        print("Midiendo L0 en reposo (%d cuadros)..." % N_REPOSO)
        ls = []
        for _ in range(N_REPOSO):
            g = cam.leer_gris()
            a = seg1.seguir(g)
            b = seg2.seguir(g)
            ls.append(math.hypot(b[0] - a[0], b[1] - a[1]))
        l0_px = float(np.mean(ls))
        ruido_def = float(np.std(ls)) / l0_px * 100.0
        mm_px = l0_mm / l0_px
        print("  L0 = %.2f px  (%.4f mm/px = %.1f px/mm)" % (l0_px, mm_px, 1.0 / mm_px))
        print("  Ruido de deformación en reposo: %.4f %% (1 sigma)" % ruido_def)
        print("  Scores: %.3f / %.3f" % (seg1.score, seg2.score))
        if min(seg1.score, seg2.score) < 0.8:
            print("  [AVISO] Correlación baja: mejora marcas, foco o iluminación.")
        if ruido_def > 0.02:
            print("  [AVISO] Ruido alto para medir módulo (>0.02 %): acerca la cámara o mejora el contraste.")
        cv2.imwrite(os.path.join(carpeta_ens, "inicio.jpg"), dibujar(g, seg1, seg2, "inicio"))
        if not (5.0 <= l0_px / l0_mm <= 200.0) or ruido_def > 0.5:
            raise RuntimeError(
                "Medición de L0 no creíble: L0 = %.1f px para %.1f mm (%.1f px/mm), ruido %.3f %%. "
                "Las dos marcas deben ser dos PUNTOS distintos (uno arriba y otro abajo), cada uno con un "
                "cuadro pequeño. Repite la selección." % (l0_px, l0_mm, l0_px / l0_mm, ruido_def))

        lector = LectorFuerza(nau, celda)
        lector.start()
        time.sleep(0.5)
        input("\nENTER para INICIAR. Para detener: q en la ventana, botón STOP o Ctrl+C... ")

        ruta_csv = os.path.join(carpeta_ens, "datos.csv")
        fcsv = open(ruta_csv, "w", newline="", encoding="utf-8")
        w = csv.writer(fcsv)
        w.writerow(["t_s", "cuadro", "F_cel_N", "edad_F_ms", "L_px", "deformacion",
                    "x1_px", "y1_px", "x2_px", "y2_px", "score1", "score2", "duty", "estado"])

        hist = deque()
        hist_est = deque()
        aviso_perdido = aviso_acerca = False
        f_pico = 0.0
        estado = "ensayo"
        t_rotura = None
        perdidos = 0
        cuadro = 0
        motivo = ""
        t0 = time.monotonic()
        act.iniciar(sentido, duty)
        print("\n>>> ENSAYO EN CURSO <<<")
        ult_vista = None
        try:
            while True:
                gris = cam.leer_gris()
                t = time.monotonic() - t0
                if estado == "ensayo":
                    act.actualizar(t)
                a = seg1.seguir(gris)
                b = seg2.seguir(gris)
                l_px = math.hypot(b[0] - a[0], b[1] - a[1])
                eps = (l_px - l0_px) / l0_px

                u = lector.ultimo
                if lector.error is not None:
                    motivo = "error de lectura de la celda: %s" % lector.error
                    break
                if u is None:
                    f, edad = float("nan"), 999.0
                else:
                    f, edad = u[1], time.monotonic() - u[0]

                w.writerow(["%.4f" % t, cuadro, "%.3f" % f, "%.0f" % (edad * 1000), "%.3f" % l_px,
                            "%.6f" % eps, "%.3f" % a[0], "%.3f" % a[1], "%.3f" % b[0], "%.3f" % b[1],
                            "%.3f" % a[2], "%.3f" % b[2], "%.3f" % act.actual, estado])
                if cuadro % 30 == 0:
                    fcsv.flush()
                cuadro += 1

                # velocidad (ventana de 1 s)
                hist.append((t, eps))
                while hist and t - hist[0][0] > 1.0:
                    hist.popleft()
                vel_pct_min = vel_mm_min = 0.0
                if len(hist) > 5 and t - hist[0][0] > 0.3:
                    de = (eps - hist[0][1]) / (t - hist[0][0])
                    vel_pct_min = de * 100.0 * 60.0
                    vel_mm_min = de * l0_mm * 60.0

                # ---- condiciones de parada ----
                if estado == "ensayo":
                    if a[2] < MIN_SCORE or b[2] < MIN_SCORE or seg1.perdido or seg2.perdido:
                        perdidos += 1
                    else:
                        perdidos = 0
                    if not math.isnan(f):
                        f_pico = max(f_pico, abs(f))
                    if perdidos > MAX_PERDIDOS:
                        if PARAR_SI_SE_PIERDE_SEGUIMIENTO:
                            motivo = "se perdió el seguimiento de las marcas"
                            break
                        if not aviso_perdido:
                            print("  [AVISO] Seguimiento de marcas perdido: la fuerza se sigue registrando "
                                  "pero la deformación de ese tramo no será fiable.")
                            aviso_perdido = True
                    if t > 1.0 and SIGNO_TRACCION * f < -UMBRAL_FUERZA_DIRECCION_N:
                        motivo = ("la celda está en COMPRESIÓN (%.0f N): dirección del actuador incorrecta" % f)
                        break
                    if abs(f) > f_max:
                        motivo = ("protección de la celda (%.0f N): revisa la probeta y el recorrido" % f_max)
                        break
                    if eps > def_max:
                        motivo = "deformación máxima (%.0f %%)" % (def_max * 100)
                        break
                    if (not COMPROBAR_ACERCAMIENTO and t > 3.0 and eps < UMBRAL_ACERCAMIENTO
                            and not aviso_acerca):
                        print("  [AVISO] Por video las marcas parecen acercarse (%.2f %%) mientras la fuerza es "
                              "%+.0f N: revisa el seguimiento en fin.jpg." % (eps * 100.0, f))
                        aviso_acerca = True
                    if COMPROBAR_ACERCAMIENTO and t > 3.0 and eps < UMBRAL_ACERCAMIENTO:
                        if abs(f) > 20.0:
                            motivo = ("las marcas se ACERCAN y la fuerza crece (%.0f N): "
                                      "dirección del actuador incorrecta" % f)
                        else:
                            motivo = ("las marcas parecen acercarse pero la fuerza no cambia "
                                      "(%.1f N): falló el SEGUIMIENTO (marcas, tamaño del cuadro, enfoque)" % f)
                        break
                    if edad > EDAD_FUERZA_MAX_S:
                        motivo = "la celda dejó de entregar datos"
                        break
                    if t > T_MAX_S:
                        motivo = "tiempo máximo"
                        break
                    if boton is not None and boton.is_pressed:
                        motivo = "botón STOP"
                        break
                    if DETECTAR_ESTANCAMIENTO and not math.isnan(f):
                        hist_est.append((t, eps, f))
                        while hist_est and t - hist_est[0][0] > VENTANA_ESTANC_S:
                            hist_est.popleft()
                        if (t > RAMPA_S + VENTANA_ESTANC_S + 2.0 and
                                t - hist_est[0][0] > 0.95 * VENTANA_ESTANC_S and
                                abs(eps - hist_est[0][1]) < UMBRAL_ESTANC_DEF and
                                abs(f - hist_est[0][2]) < max(5.0, 0.01 * f_pico)):
                            motivo = ("el actuador dejó de avanzar o la probeta no cambia (fin de carrera, "
                                      "bloqueo o deslizamiento): fuerza y deformación estancadas %.0f s"
                                      % VENTANA_ESTANC_S)
                            break
                    if f_pico > F_MIN_ROTURA_N and abs(f) < FRAC_ROTURA * f_pico:
                        estado = "rotura"
                        t_rotura = t
                        act.parar()
                        print("  >>> ROTURA detectada: F = %.1f N (máx %.1f N)" % (abs(f), f_pico))
                elif estado == "rotura" and t - t_rotura > POST_ROTURA_S:
                    motivo = "rotura de la probeta"
                    break

                if cuadro % VISTA_CADA_N == 0:
                    txt = "F=%+.1f N  sig=%.2f MPa  e=%.3f %%  marcas=%.2f mm/min" % (
                        f, f / area, eps * 100.0, vel_mm_min)
                    ult_vista = dibujar(gris, seg1, seg2, txt)
                    k = mostrar("ENSAYO (q = detener)", ult_vista)
                    if k in (ord("q"), 27):
                        motivo = "tecla q"
                        break
                if cuadro % 30 == 0:
                    print("  t=%6.1f s | F=%+9.1f N | sig=%7.2f MPa | e=%7.3f %% | marcas=%6.2f mm/min | %s"
                          % (t, f, f / area, eps * 100.0, vel_mm_min, estado))
        except KeyboardInterrupt:
            motivo = "usuario (Ctrl+C)"
        finally:
            act.parar()
            fcsv.flush()
            fcsv.close()
            if lector is not None:
                lector.parar_ev.set()
        t_fin = time.monotonic() - t0
        print("\nENSAYO TERMINADO: %s" % motivo)

        try:
            cv2.imwrite(os.path.join(carpeta_ens, "fin.jpg"), dibujar(gris, seg1, seg2, "fin: " + motivo))
        except Exception:
            pass
        with open(os.path.join(carpeta_ens, "fuerza_cruda.csv"), "w", newline="", encoding="utf-8") as f2:
            w2 = csv.writer(f2)
            w2.writerow(["t_s", "cuentas", "F_cel_N"])
            for (tt, c, ff) in lector.registro:
                w2.writerow(["%.4f" % (tt - t0), c, "%.3f" % ff])
        meta = {
            "nombre": nombre, "fecha": datetime.now().isoformat(timespec="seconds"),
            "probeta": {"ancho_mm": ancho, "espesor_mm": espesor, "area_mm2": area,
                        "L0_mm": l0_mm},
            "video": {"L0_px": l0_px, "mm_por_px": mm_px, "ruido_deformacion_pct": ruido_def,
                      "resolucion": [CAM_ANCHO, CAM_ALTO], "fps_objetivo": CAM_FPS,
                      "exposicion_us": cam.exposicion_us, "ganancia": cam.ganancia,
                      "roi1": rois[0], "roi2": rois[1],
                      "refrescos_plantilla": [seg1.refrescos, seg2.refrescos]},
            "celda": {"archivo_calibracion": os.path.basename(RUTA_CAL),
                      "fecha_calibracion": celda.cal.get("fecha"),
                      "cuentas_por_N": celda.k, "tara_cuentas": celda.tara,
                      "convencion": "traccion de la celda = positivo (signo de la calibración)"},
            "actuador": {"sentido_que_estira": sentido, "velocidad": duty, "duty": duty, "frecuencia_pwm_hz": FREC_PWM,
                         "rampa_s": RAMPA_S},
            "limites": {"F_proteccion_celda_N": (None if math.isinf(f_max) else f_max), "def_max": def_max,
                        "sin_limite_de_fuerza_de_ensayo": True},
            "resultado": {"motivo_fin": motivo, "duracion_s": t_fin, "cuadros": cuadro,
                          "fps_real": cuadro / t_fin if t_fin > 0 else None,
                          "F_pico_abs_N": f_pico, "rotura_t_s": t_rotura},
        }
        with open(os.path.join(carpeta_ens, "metadatos.json"), "w", encoding="utf-8") as fj:
            json.dump(meta, fj, indent=2, ensure_ascii=False)
        print("Fuerza máxima: %.1f N (%.2f MPa) | %d cuadros (%.1f fps)"
              % (f_pico, f_pico / area, cuadro, cuadro / t_fin if t_fin > 0 else 0))
        n_f = len(lector.registro)
        print("Lecturas de fuerza registradas: %d en %.1f s (%.1f por segundo)" % (n_f, t_fin, n_f / max(t_fin, 1e-9)))
        if n_f < 30:
            print("  [AVISO] Muy pocas lecturas de fuerza para una curva válida (módulo y fluencia). "
                  "Con esta velocidad la prueba es demasiado rápida: baja VELOCIDAD (0.1 a 0.3).")
        print("Datos en: %s" % carpeta_ens)
        print("Siguiente paso: ejecuta curva_esfuerzo_deformacion.py")
    finally:
        act.cerrar()
        if lector is not None:
            lector.parar_ev.set()
        cam.cerrar()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError) as e:
        print("\n[ERROR] %s" % e)