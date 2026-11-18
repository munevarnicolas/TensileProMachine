# -*- coding: utf-8 -*-
"""
curva_esfuerzo_deformacion.py
=============================
Convierte los datos de ensayo_traccion.py en la curva esfuerzo-deformación
(ingenieril y verdadera) y calcula propiedades tipo ASTM D638:

  * Módulo de elasticidad E : regresión lineal con los puntos entre 10 % y 40 %
                              de la resistencia máxima (parte ascendente).
  * Corrección de pie (toe) : se desplaza la deformación para que la recta de E
                              pase por el origen (ASTM D638, Anexo A1).
  * Resistencia máxima, deformación en el máximo.
  * Fluencia por desplazamiento 0.2 % (si existe).
  * Deformación y esfuerzo en la rotura.

Uso: ejecútalo en Thonny. Toma el ensayo más reciente de la carpeta 'ensayos'
(o pega la ruta de otro cuando te la pida).

Esfuerzo = F / A0 (N/mm2 = MPa). Deformación = (L - L0)/L0 medida por video.
El signo de la fuerza se ajusta solo: el esfuerzo de tracción sale positivo.
"""
import csv
import glob
import json
import math
import os

import numpy as np

try:
    CARPETA = os.path.dirname(os.path.abspath(__file__))
except NameError:
    CARPETA = os.getcwd()

# ----------------------------- Parámetros -----------------------------
FRAC_MODULO = (0.10, 0.40)    # tramo de E, como fracción de la resistencia máxima
OFFSET_FLUENCIA = 0.002       # 0.2 %
VENT_MEDIANA = 3              # muestras (se aplican a esfuerzo y deformación)
VENT_PROMEDIO = 3
RETRASO_CAMARA_S = 0.0        # latencia de la cámara (s): si la imagen llega ~50 ms tarde, pon 0.05


# ----------------------------- Utilidades -----------------------------
def suavizar(y, n_med=VENT_MEDIANA, n_prom=VENT_PROMEDIO):
    y = np.asarray(y, float)
    if len(y) < max(n_med, n_prom) + 2:
        return y.copy()
    h = n_med // 2
    pad = np.pad(y, h, mode="edge")
    med = np.array([np.median(pad[i:i + n_med]) for i in range(len(y))])
    k = np.ones(n_prom) / n_prom
    pad2 = np.pad(med, n_prom // 2, mode="edge")
    return np.convolve(pad2, k, mode="valid")


def elegir_carpeta():
    base = os.path.join(CARPETA, "ensayos")
    todas = sorted(glob.glob(os.path.join(base, "*", "datos.csv")))
    if not todas:
        raise RuntimeError("No hay ensayos en %s" % base)
    ult = os.path.dirname(todas[-1])
    r = input("Ensayo [%s] (ENTER = ese, o pega otra ruta): " % os.path.basename(ult)).strip().strip('"')
    return r if r else ult


def cargar(carpeta):
    with open(os.path.join(carpeta, "metadatos.json"), encoding="utf-8") as f:
        meta = json.load(f)
    filas = []
    with open(os.path.join(carpeta, "datos.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            filas.append(r)
    col = lambda n: np.array([float(r[n]) for r in filas])
    tF = FF = None
    ruta_f = os.path.join(carpeta, "fuerza_cruda.csv")
    if os.path.exists(ruta_f):
        with open(ruta_f, encoding="utf-8") as f:
            rf = list(csv.DictReader(f))
        if len(rf) > 5:
            tF = np.array([float(r["t_s"]) for r in rf])
            FF = np.array([float(r["F_cel_N"]) for r in rf])
    d = {"tF": tF, "FF": FF,
         "t": col("t_s"), "F": col("F_cel_N"), "eps": col("deformacion"),
         "estado": [r["estado"] for r in filas], "edad": col("edad_F_ms"),
         "score1": col("score1"), "score2": col("score2")}
    return meta, d


def ajuste_lineal(x, y):
    p = np.polyfit(x, y, 1)
    pred = np.polyval(p, x)
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return p[0], p[1], (1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def analizar(meta, d):
    area = meta["probeta"]["area_mm2"]
    if d.get("tF") is not None:
        # Cada punto de la curva es una LECTURA REAL de fuerza; la deformación (30 cuadros/s)
        # se interpola a ese instante. Evita repetir valores de fuerza entre cuadros.
        m = (d["tF"] >= d["t"][0]) & (d["tF"] <= d["t"][-1])
        t, F = d["tF"][m], d["FF"][m]
        eps = np.interp(t + RETRASO_CAMARA_S, d["t"], d["eps"])
        res_fuente = "lecturas nativas de fuerza (%d) con deformación interpolada" % len(t)
    else:
        F = d["F"].copy()
        ok = ~np.isnan(F)
        F, eps, t = F[ok], d["eps"][ok], d["t"][ok]
        res_fuente = "fuerza del último dato disponible en cada cuadro"
    # signo: la fuerza de mayor valor absoluto es de tracción
    signo = 1.0 if F[np.argmax(np.abs(F))] >= 0 else -1.0
    F = signo * F
    sig_cruda = F / area
    sig = suavizar(sig_cruda)
    eps_s = suavizar(eps)

    # el máximo se toma de la lectura SIN suavizar: suavizar aplana el pico cuando hay pocos puntos
    ipk = int(np.argmax(sig_cruda))
    sig_max = float(sig_cruda[ipk])
    res = {"fuente_datos": res_fuente, "signo_aplicado": signo, "area_mm2": area, "sigma_max_MPa": sig_max,
           "F_max_N": float(F[ipk]), "deformacion_en_max_pct": float(eps_s[ipk] * 100.0)}

    # ---- módulo ----
    lo, hi = FRAC_MODULO
    m = (sig[:ipk + 1] >= lo * sig_max) & (sig[:ipk + 1] <= hi * sig_max)
    if int(np.sum(m)) >= 10:
        E, b, r2 = ajuste_lineal(eps_s[:ipk + 1][m], sig[:ipk + 1][m])
        eps0_recta = -b / E if E != 0 else 0.0
        # Corrección de pie (toe) SOLO si la recta de E corta el eje de deformación en positivo
        # (curva con 'pie'). Si corta en negativo la curva es cóncava hacia abajo: no se corrige.
        eps0 = max(eps0_recta, 0.0)
        res.update({"E_MPa": float(E), "E_GPa": float(E / 1000.0), "E_R2": float(r2),
                    "E_n_puntos": int(np.sum(m)), "toe_deformacion_pct": float(eps0 * 100.0),
                    "toe_recta_pct": float(eps0_recta * 100.0)})
    else:
        E, b, r2, eps0 = None, None, None, 0.0
        res["E_MPa"] = None
        res["aviso_E"] = ("Muy pocos puntos entre %d %% y %d %% de la resistencia máxima: la prueba fue "
                          "demasiado rápida para la frecuencia de lectura de la celda. Baja la velocidad."
                          % (lo * 100, hi * 100))
    eps_c = eps_s - eps0                      # deformación con corrección de pie

    # ---- módulo tangente máximo: mayor pendiente local (ventana de 12 puntos) antes del 60 % ----
    res["E_tangente_MPa"] = None
    W = 12
    mejor = 0.0
    for i in range(0, max(ipk - W, 0)):
        xs, ys = eps_s[i:i + W], sig[i:i + W]
        if len(xs) < W or ys[-1] > 0.6 * sig_max:
            break
        if (xs[-1] - xs[0]) < 5e-4 or (ys[-1] - ys[0]) < 1.0:
            continue
        Ei, bi, r2i = ajuste_lineal(xs, ys)
        if r2i > 0.98 and Ei > mejor:
            mejor = Ei
            res["E_tangente_MPa"] = float(Ei)
            res["E_tangente_ventana_pct"] = [float(xs[0] * 100.0), float(xs[-1] * 100.0)]

    # ---- fluencia por desplazamiento 0.2 % ----
    res["fluencia_offset_MPa"] = None
    if E:
        f = sig - E * (eps_c - OFFSET_FLUENCIA)
        idx = np.where((eps_c > OFFSET_FLUENCIA) & (f < 0))[0]
        if len(idx) and idx[0] > 0:
            i = idx[0]
            f0, f1 = f[i - 1], f[i]
            fr = f0 / (f0 - f1) if f0 != f1 else 0.0
            res["fluencia_offset_MPa"] = float(sig[i - 1] + fr * (sig[i] - sig[i - 1]))
            res["fluencia_offset_def_pct"] = float((eps_c[i - 1] + fr * (eps_c[i] - eps_c[i - 1])) * 100.0)

    # ---- rotura: caída de fuerza más brusca después del máximo ----
    ibr = None
    if len(sig) - ipk > 3:
        dF = np.diff(F[ipk:])
        j = int(np.argmin(dF))
        if F[ipk + j] > 0 and dF[j] < -0.2 * F[ipk]:
            ibr = ipk + j
    if ibr is not None:
        res["rotura_def_pct"] = float(eps_c[ibr] * 100.0)
        res["rotura_sigma_MPa"] = float(F[ibr] / area)   # valor sin suavizar: justo antes de la caída
    else:
        res["aviso_rotura"] = "No se detectó una caída brusca de fuerza (ensayo detenido antes de romper)."

    # ---- fluencia por pico: máximo seguido de una caída antes de la rotura (materiales dúctiles) ----
    fin = ibr if ibr is not None else len(sig) - 1
    if fin > ipk and float(np.min(sig[ipk:fin + 1])) <= 0.98 * sig_max:
        res["fluencia_pico_MPa"] = sig_max
        res["fluencia_pico_def_pct"] = float(eps_c[ipk] * 100.0)

    # ---- calidad de datos ----
    res["calidad"] = {"score_min": float(min(np.min(d["score1"]), np.min(d["score2"]))),
                      "edad_F_max_ms": float(np.max(d["edad"])),
                      "muestras": int(len(F))}
    curva = {"eps_c": eps_c, "sig": sig, "eps_raw": eps, "F": F, "t": t,
             "ipk": ipk, "ibr": ibr, "E": E, "b": b}
    return res, curva


def guardar_y_graficar(carpeta, meta, res, c):
    eps_pct = c["eps_c"] * 100.0
    # verdadera (válida hasta antes de la estricción)
    sig_v = c["sig"] * (1.0 + c["eps_c"])
    eps_v = np.log1p(np.clip(c["eps_c"], -0.99, None)) * 100.0
    with open(os.path.join(carpeta, "curva_procesada.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "F_N", "deformacion_pct", "esfuerzo_MPa", "deform_verdadera_pct",
                    "esfuerzo_verdadero_MPa"])
        for i in range(len(c["t"])):
            w.writerow(["%.4f" % c["t"][i], "%.3f" % c["F"][i], "%.5f" % eps_pct[i],
                        "%.4f" % c["sig"][i], "%.5f" % eps_v[i], "%.4f" % sig_v[i]])
    with open(os.path.join(carpeta, "resultados.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
    a = ax[0]
    a.plot(eps_pct, c["sig"], lw=1.6, label="Esfuerzo ingenieril")
    if c["E"]:
        xs = np.linspace(0, 1.1 * res["sigma_max_MPa"] / c["E"] * 100.0, 50)   # hasta ~110 % de la resistencia
        a.plot(xs, c["E"] * xs / 100.0, "--", color="gray", lw=1,
               label="E = %.0f MPa (R²=%.4f)" % (res["E_MPa"], res["E_R2"]))
    ipk = c["ipk"]
    etiqueta = ("Fluencia (pico): %.1f MPa" if res.get("fluencia_pico_MPa") else "Máximo: %.1f MPa")
    a.plot(eps_pct[ipk], c["sig"][ipk], "ro", label=etiqueta % res["sigma_max_MPa"])
    if res.get("fluencia_offset_MPa"):
        a.plot(res["fluencia_offset_def_pct"], res["fluencia_offset_MPa"], "gs",
               label="Fluencia 0.2 %%: %.1f MPa" % res["fluencia_offset_MPa"])
    if c["ibr"] is not None:
        a.plot(eps_pct[c["ibr"]], c["sig"][c["ibr"]], "kx", ms=9, mew=2,
               label="Rotura: %.1f %%" % res["rotura_def_pct"])
    a.set_ylim(0, 1.15 * res["sigma_max_MPa"])
    a.set_xlabel("Deformación ingenieril (%)")
    a.set_ylabel("Esfuerzo ingenieril (MPa)")
    a.set_title("Curva esfuerzo-deformación - %s" % meta.get("nombre", ""))
    a.grid(alpha=0.3)
    a.legend(loc="lower right", fontsize=8)
    b = ax[1]
    b.plot(c["t"], c["F"], color="tab:blue")
    b.set_xlabel("Tiempo (s)")
    b.set_ylabel("Fuerza (N)", color="tab:blue")
    b.grid(alpha=0.3)
    b2 = b.twinx()
    b2.plot(c["t"], c["eps_raw"] * 100.0, color="tab:red", lw=1)
    b2.set_ylabel("Deformación (%)", color="tab:red")
    b.set_title("Fuerza y deformación vs tiempo")
    fig.tight_layout()
    png = os.path.join(carpeta, "curva_esfuerzo_deformacion.png")
    fig.savefig(png, dpi=150)
    print("  Gráfica: %s" % png)
    try:
        plt.show()
    except Exception:
        pass


def main():
    carpeta = elegir_carpeta()
    meta, d = cargar(carpeta)
    res, c = analizar(meta, d)
    print("\n" + "=" * 62)
    print("RESULTADOS  -  %s" % meta.get("nombre", ""))
    print("=" * 62)
    print("Probeta: %.2f x %.2f mm (A0 = %.2f mm2) | L0 = %.1f mm"
          % (meta["probeta"]["ancho_mm"], meta["probeta"]["espesor_mm"],
             meta["probeta"]["area_mm2"], meta["probeta"]["L0_mm"]))
    print("Datos usados       : %s" % res["fuente_datos"])
    print("Resistencia máxima : %.2f MPa  (F = %.1f N, deformación %.2f %%)"
          % (res["sigma_max_MPa"], res["F_max_N"], res["deformacion_en_max_pct"]))
    if res.get("E_MPa"):
        print("Módulo E           : %.0f MPa (%.3f GPa)  R² = %.4f  (%d puntos, toe %.3f %%)"
              % (res["E_MPa"], res["E_GPa"], res["E_R2"], res["E_n_puntos"], res["toe_deformacion_pct"]))
    else:
        print("Módulo E           : no calculado (%s)" % res.get("aviso_E", ""))
    if res.get("E_tangente_MPa"):
        print("Módulo tangente máx: %.0f MPa (%.3f GPa) en %.2f-%.2f %% de deformación"
              % (res["E_tangente_MPa"], res["E_tangente_MPa"] / 1000.0,
                 res["E_tangente_ventana_pct"][0], res["E_tangente_ventana_pct"][1]))
    if res.get("fluencia_pico_MPa"):
        print("Fluencia (pico)    : %.2f MPa a %.2f %% de deformación (máximo seguido de caída)"
              % (res["fluencia_pico_MPa"], res["fluencia_pico_def_pct"]))
    if res.get("fluencia_offset_MPa"):
        print("Fluencia (0.2 %%)   : %.2f MPa" % res["fluencia_offset_MPa"])
    else:
        print("Fluencia (0.2 %)   : no hay intersección (material frágil o ensayo incompleto)")
    if "rotura_def_pct" in res:
        print("Rotura             : deformación %.2f %% | esfuerzo %.2f MPa"
              % (res["rotura_def_pct"], res["rotura_sigma_MPa"]))
    else:
        print("Rotura             : %s" % res.get("aviso_rotura", ""))
    q = res["calidad"]
    print("Calidad            : score mín. %.2f | edad máx. de fuerza %.0f ms | %d muestras"
          % (q["score_min"], q["edad_F_max_ms"], q["muestras"]))
    if q["score_min"] < 0.7:
        print("  [AVISO] La correlación bajó de 0.7 en algún momento: revisa las marcas/iluminación.")
    print("Resultados guardados en: %s" % carpeta)
    guardar_y_graficar(carpeta, meta, res, c)


if __name__ == "__main__":
    main()