# -*- coding: utf-8 -*-
"""
medir_en_vivo.py
================
Lectura en vivo de la fuerza con la calibración YA guardada (calibracion_celda.json).
NO recalibra y NO modifica el JSON. Sirve para comprobar con cualquier otra masa.

Uso:
  1. Deja la celda SIN la masa de prueba (con la placa/herraje puestos).
  2. Ejecuta este archivo en Thonny. Se tara 5 s.
  3. Escribe la masa de prueba (kg) y el sentido (T/C) para ver el error contra
     lo esperado. ENTER en la masa = solo mostrar la fuerza, sin comparar.
  4. Apoya la masa suavemente y mira la línea cuando se estabilice.
  5. Ctrl+C (o Stop en Thonny) abre un menú: tarar de nuevo, otra masa o salir.

Requiere en la misma carpeta: celda_calibrada.py y calibracion_celda.json
"""
import time

from celda_calibrada import CeldaCalibrada, configurar_nau7802

G_ESTANDAR = 9.80665


def pedir_masa():
    while True:
        txt = input("Masa de prueba en kg (ENTER = sin comparar): ").strip().replace(",", ".")
        if txt == "":
            return None, None
        try:
            masa = float(txt)
            if masa <= 0:
                raise ValueError
        except ValueError:
            print("  Valor no válido.")
            continue
        sent = input("¿Se aplica en TRACCIÓN o COMPRESIÓN? [T/C] (ENTER = C): ").strip().upper()
        sent = "T" if sent == "T" else "C"
        return masa, sent


def tarar(cel):
    print("\nTarando 5 s: la celda debe estar SIN la masa de prueba y sin tocarla...")
    est = cel.tarar(5.0)
    print("  Tara: %.1f cuentas (std %.1f = %.2f N)\n" % (cel.tara, est["std"], est["std"] / abs(cel.k)))


def vivo(cel, g, masa, sent):
    f_esp = None
    if masa is not None:
        f_esp = masa * g * (1.0 if sent == "T" else -1.0)
        print("Fuerza esperada: %+.2f N (%.3f kg, g = %.5f m/s2). Apoya la masa suavemente.\n"
              % (f_esp, masa, g))
    else:
        print("Mostrando fuerza en vivo. Apoya la masa suavemente.\n")
    print("Ctrl+C / Stop para el menú.\n")
    while True:
        c = cel.leer_cuentas(5)
        f = cel.fuerza_N(c)
        linea = "cuentas %10.0f | F_uso %+10.2f N (%+8.3f kgf)" % (c, f, f / G_ESTANDAR)
        if f_esp is not None:
            err = f - f_esp
            linea += " | error vs esperado %+7.2f N (%+6.2f %%)" % (err, err / abs(f_esp) * 100.0)
        print(linea)
        time.sleep(0.5)


def main():
    nau = configurar_nau7802()
    cel = CeldaCalibrada(nau)
    g = float(cel.cal.get("patron", {}).get("g_m_s2", G_ESTANDAR))
    print("Calibración cargada: k = %+.3f cuentas/N (%s)" %
          (cel.k, cel.cal.get("fecha", "sin fecha")))
    tarar(cel)
    masa, sent = pedir_masa()
    while True:
        try:
            vivo(cel, g, masa, sent)
        except KeyboardInterrupt:
            print("\n--- Menú ---")
            op = input("[T]arar de nuevo / [M]asa nueva / [S]alir: ").strip().upper()
            if op.startswith("S"):
                print("Fin.")
                return
            if op.startswith("T"):
                input("Quita la masa de prueba y pulsa ENTER... ")
                tarar(cel)
            elif op.startswith("M"):
                input("Quita la masa de prueba y pulsa ENTER para tarar... ")
                tarar(cel)
                masa, sent = pedir_masa()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nFin.")
    except (OSError, RuntimeError, FileNotFoundError) as e:
        print("\n[ERROR] %s" % e)
        print("Revisa el I2C (sudo i2cdetect -y 1 -> 2a) y que exista calibracion_celda.json.")