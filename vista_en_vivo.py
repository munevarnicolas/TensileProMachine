# -*- coding: utf-8 -*-
"""
vista_en_vivo.py
================
Vista previa SIN LÍMITE DE TIEMPO para encuadrar la probeta y enfocar las marcas,
con la misma configuración de cámara que ensayo_traccion.py (2028x1520, 30 fps).

Ábrelo desde el escritorio de la Raspberry (RealVNC) para ver la ventana.

Teclas (con la ventana seleccionada):
  q / ESC  salir
  CLIC     mueve la zona de enfoque (mide la nitidez ahí; pincha sobre una marca)
  z        ventana de zoom 1:1 (x2) de la zona de enfoque: la mejor ayuda para enfocar
  g        rejilla de tercios + cruz central (alinea el eje de la probeta con la línea vertical)
  r        reinicia el máximo de nitidez
  l        bloquea / libera la exposición (bloquéala al terminar de encuadrar)
  s        guarda una foto a resolución completa

Enfoque: gira el anillo del lente hasta que la NITIDEZ sea lo más alta posible
(compárala con el 'máx.' mostrado). Las zonas ROJAS están saturadas: reduce luz.

Si no hay ventana gráfica disponible, imprime la nitidez en la terminal sin parar (Ctrl+C / Stop).
"""
import time
from datetime import datetime

import cv2
import numpy as np
from picamera2 import Picamera2

ANCHO, ALTO, FPS = 2028, 1520, 30
ESC = 0.5                  # escala de la vista
ROI = 400                  # lado (px, resolución completa) de la zona de enfoque
VENT = "Vista en vivo (q = salir)"
VENT_Z = "Zoom de la zona de enfoque (x2)"

estado = {"cx": ANCHO // 2, "cy": ALTO // 2}


def al_clic(evento, x, y, flags, param):
    if evento == cv2.EVENT_LBUTTONDOWN:
        estado["cx"] = int(np.clip(x / ESC, ROI // 2, ANCHO - ROI // 2))
        estado["cy"] = int(np.clip(y / ESC, ROI // 2, ALTO - ROI // 2))


def texto(img, s, y):
    cv2.putText(img, s, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4)
    cv2.putText(img, s, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 1)


def nitidez(gris, cx, cy):
    z = gris[cy - ROI // 2: cy + ROI // 2, cx - ROI // 2: cx + ROI // 2]
    return float(cv2.Laplacian(z, cv2.CV_64F).var())


def main():
    if not Picamera2.global_camera_info():
        print("[ERROR] No se detecta ninguna cámara (mira 'rpicam-hello --list-cameras').")
        return
    cam = Picamera2()
    dur = int(1e6 / FPS)
    cfg = cam.create_video_configuration(main={"size": (ANCHO, ALTO), "format": "YUV420"},
                                         controls={"FrameDurationLimits": (dur, dur)}, buffer_count=4)
    cam.configure(cfg)
    cam.start()
    time.sleep(1.0)

    gui = True
    try:
        cv2.namedWindow(VENT, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(VENT, al_clic)
    except cv2.error:
        gui = False
        print("[AVISO] No hay ventana gráfica: modo terminal (Ctrl+C / Stop para salir).")
        print("        Para ver la imagen ejecuta este programa desde el escritorio (VNC).")

    mostrar_rejilla, mostrar_zoom, ae_bloqueada = True, False, False
    maximo, fps_s, t_ant = 0.0, float(FPS), time.monotonic()
    print("Vista en vivo iniciada. Sin límite de tiempo.")
    try:
        while True:
            gris = cam.capture_array("main")[:ALTO, :ANCHO]
            ahora = time.monotonic()
            fps_s = 0.9 * fps_s + 0.1 / max(ahora - t_ant, 1e-6)
            t_ant = ahora
            cx, cy = estado["cx"], estado["cy"]
            nit = nitidez(gris, cx, cy)
            maximo = max(maximo, nit)
            sat = 100.0 * float(np.mean(gris >= 250))

            if not gui:
                print("nitidez %8.1f (máx %8.1f) | brillo %5.1f | saturado %5.2f %%"
                      % (nit, maximo, gris.mean(), sat))
                time.sleep(0.4)
                continue

            v = cv2.cvtColor(cv2.resize(gris, None, fx=ESC, fy=ESC), cv2.COLOR_GRAY2BGR)
            v[v[:, :, 0] >= 250] = (0, 0, 255)                      # saturación en rojo
            h, w = v.shape[:2]
            if mostrar_rejilla:
                for f in (1 / 3, 2 / 3):
                    cv2.line(v, (int(w * f), 0), (int(w * f), h), (200, 200, 200), 1)
                    cv2.line(v, (0, int(h * f)), (w, int(h * f)), (200, 200, 200), 1)
                cv2.line(v, (w // 2, 0), (w // 2, h), (255, 255, 0), 1)
                cv2.line(v, (0, h // 2), (w, h // 2), (255, 255, 0), 1)
            r = int(ROI * ESC / 2)
            c = (int(cx * ESC), int(cy * ESC))
            cv2.rectangle(v, (c[0] - r, c[1] - r), (c[0] + r, c[1] + r), (0, 255, 0), 1)
            texto(v, "NITIDEZ %8.1f   max %8.1f   (zona verde)" % (nit, maximo), 28)
            barra = int(min(nit / max(maximo, 1e-6), 1.0) * 300)
            cv2.rectangle(v, (10, 38), (10 + barra, 50), (0, 255, 0), -1)
            cv2.rectangle(v, (10, 38), (310, 50), (255, 255, 255), 1)
            texto(v, "brillo %.0f | saturado %.2f %% | %.1f fps | AE %s"
                  % (gris.mean(), sat, fps_s, "BLOQUEADA" if ae_bloqueada else "auto"), 72)
            cv2.imshow(VENT, v)

            if mostrar_zoom:
                zc = gris[cy - 120: cy + 120, cx - 160: cx + 160]
                zc = cv2.resize(zc, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
                cv2.imshow(VENT_Z, zc)

            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break
            elif k == ord("g"):
                mostrar_rejilla = not mostrar_rejilla
            elif k == ord("z"):
                mostrar_zoom = not mostrar_zoom
                if not mostrar_zoom:
                    cv2.destroyWindow(VENT_Z)
            elif k == ord("r"):
                maximo = 0.0
            elif k == ord("l"):
                try:
                    if not ae_bloqueada:
                        md = cam.capture_metadata()
                        cam.set_controls({"AeEnable": False, "ExposureTime": int(md["ExposureTime"]),
                                          "AnalogueGain": float(md["AnalogueGain"])})
                    else:
                        cam.set_controls({"AeEnable": True})
                    ae_bloqueada = not ae_bloqueada
                except Exception as e:
                    print("No se pudo cambiar la exposición: %s" % e)
            elif k == ord("s"):
                nombre = "foto_%s.jpg" % datetime.now().strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(nombre, gris)
                print("Guardada: %s" % nombre)
    except KeyboardInterrupt:
        pass
    finally:
        cam.stop()
        cam.close()
        if gui:
            cv2.destroyAllWindows()
        print("Vista en vivo cerrada.")


if __name__ == "__main__":
    main()