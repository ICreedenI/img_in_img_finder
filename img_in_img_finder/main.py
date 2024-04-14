# -*- coding: utf-8 -*-

import copy
import ctypes
import os
import pickle
import sys
import textwrap
from pathlib import Path
from time import sleep  # , os
from time import time

import cv2
from keyboard import wait, add_hotkey, remove_hotkey
import matplotlib.pyplot as plt
import numpy as np
from pyautogui import position
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
from icecream import ic
from matplotlib import pyplot as plt
from PIL import ImageGrab
from Zeit_Datum import Zeit

from colorful_terminal import *


def Finde_Bild(
    Pfad, Treffsicherheit=0.8, Kasten_Abstand=0, Suchbereich=(0, 0, 7680, 2160)
):
    try:
        global screenhot

        if type(Pfad) != str:
            try:
                Pfad = Pfad[0]
            except:
                pass

        screenshot_PIL = ImageGrab.grab(
            all_screens=True, bbox=Suchbereich
        )  # Screenshot im gewünschten Bereich erstellen
        screenhot = cv2.cvtColor(
            np.asarray(screenshot_PIL), cv2.COLOR_RGB2BGR
        )  # Farbschema des Screenshot konvertieren
        small_image = cv2.imread(
            Pfad, cv2.COLOR_RGB2BGR
        )  # Zu suchendes Bild mit OpenCV öffnen - Farbschema Blau Grün Rot

        result = cv2.matchTemplate(
            screenhot, small_image, cv2.TM_CCOEFF_NORMED
        )  # Eigentliche Bildsuche

        # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        # print(min_val, "--", max_val, "--", min_loc, "--", max_loc)

        breite = small_image.shape[1]  # Breite und Höhe des gesuchten Bildes bestimmen
        hoehe = small_image.shape[0]

        yloc, xloc = np.where(
            result >= Treffsicherheit
        )  # Koordinaten der linken oberen Ecke aller Ergebnisse über der Schwelle bestimmen

        l = len(xloc)  # Anzahl an Rohfunden
        # print("Anzahl an Funden:", l)

        rectangles = []
        for (x, y) in zip(xloc, yloc):  # Rohfunde verdoppeln um nichtszu verlieren
            rectangles.append([int(x), int(y), int(breite), int(hoehe)])
            rectangles.append([int(x), int(y), int(breite), int(hoehe)])

        rectangles, weights = cv2.groupRectangles(rectangles, 1, Kasten_Abstand)

        Positionen = []
        for (
            x,
            y,
            breite,
            hoehe,
        ) in rectangles:  # Zentrum der Position an die Positions-Liste anfügen

            x_center, y_center = x + breite / 2, y + hoehe / 2
            # Relative Koordinaten auf absolute beziehen
            left, top, right, bottom = Suchbereich
            x_center_abs, y_center_abs = x_center + left, y_center + top
            x_center_abs, y_center_abs = int(x_center_abs), int(y_center_abs)
            Positionen.append((x_center_abs, y_center_abs))

        return Positionen

    except AttributeError as e:
        print('Fehler in der Funktion "Finde_Bild"!\n\tFehlernachricht:')
        wrapper = textwrap.TextWrapper(
            width=200, initial_indent="\t\t", subsequent_indent="\t\t"
        )
        word_list = wrapper.wrap(text=str(e))
        # Print each line.
        for element in word_list:
            print(element)
        print("\tFehlerart:\n\t\t{}".format(type(e)))
        print("\tDer Pfad des zu suchenden Bildes könnte falsch sein.")
        print("\tBist du dir sicher, dass der folgende Pfad korrekt ist?")
        print("\t\t{}".format(Pfad))
        return
    except cv2.error as e:
        # print("Fehler in der Funktion \"alpha_template_match_ds\"!\n\tFehlernachricht:\n\t\t{}".format(e))
        print('Fehler in der Funktion "Finde_Bild"!\n\tFehlernachricht:')
        wrapper = textwrap.TextWrapper(
            width=200, initial_indent="\t\t", subsequent_indent="\t\t"
        )
        word_list = wrapper.wrap(text=str(e))
        # Print each line.
        for element in word_list:
            print(element)
        print("\tFehlerart:\n\t\t{}".format(type(e)))
        print("\tDer gewünschte Suchbereich könnte zu klein sein.")
        print(
            "\tVeruche diesen zu vergrößern oder eine geringere Verkleinerung zu wählen."
        )
        print("\tDer gewünschte Suchbereich wurde festgelegt als:")
        print("\t\t{}".format(Suchbereich))
        return
    except Exception as e:
        print('Fehler in der Funktion "Finde_Bild"!\n\tFehlernachricht:')
        wrapper = textwrap.TextWrapper(
            width=200, initial_indent="\t\t", subsequent_indent="\t\t"
        )
        word_list = wrapper.wrap(text=str(e))
        # Print each line.
        for element in word_list:
            print(element)
        print("\tFehlerart:\n\t\t{}".format(type(e)))
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print(f"\tFehler in Linie:\n\t\t{exc_tb.tb_lineno}")
        return


def Template_Matching(
    Pfad, Treffsicherheit=0.8, Kasten_Abstand=0, Suchbereich=(0, 0, 7680, 2160)
):

    global screenhot

    screenshot_PIL = ImageGrab.grab(
        all_screens=True, bbox=Suchbereich
    )  # Screenshot im gewünschten Bereich erstellen
    screenhot = cv2.cvtColor(
        np.asarray(screenshot_PIL), cv2.COLOR_RGB2BGR
    )  # Farbschema des Screenshot konvertieren
    small_image = cv2.imread(
        Pfad, cv2.COLOR_RGB2BGR
    )  # Zu suchendes Bild mit OpenCV öffnen - Farbschema Blau Grün Rot

    result = cv2.matchTemplate(
        screenhot, small_image, cv2.TM_CCOEFF_NORMED
    )  # Eigentliche Bildsuche

    # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    # print(min_val, "--", max_val, "--", min_loc, "--", max_loc)

    breite = small_image.shape[1]  # Breite und Höhe des gesuchten Bildes bestimmen
    hoehe = small_image.shape[0]

    yloc, xloc = np.where(
        result >= Treffsicherheit
    )  # Koordinaten der linken oberen Ecke aller Ergebnisse über der Schwelle bestimmen

    l = len(xloc)  # Anzahl an Rohfunden
    # print("Anzahl an Funden:", l)

    rectangles = []
    for (x, y) in zip(xloc, yloc):  # Rohfunde verdoppeln um nichtszu verlieren
        rectangles.append([int(x), int(y), int(breite), int(hoehe)])
        rectangles.append([int(x), int(y), int(breite), int(hoehe)])

    rectangles, weights = cv2.groupRectangles(rectangles, 1, Kasten_Abstand)

    Positionen = []
    for (
        x,
        y,
        breite,
        hoehe,
    ) in rectangles:  # Zentrum der Position an die Positions-Liste anfügen

        x_center, y_center = x + breite / 2, y + hoehe / 2
        # Relative Koordinaten auf absolute beziehen
        left, top, right, bottom = Suchbereich
        x_center_abs, y_center_abs = x_center + left, y_center + top
        x_center_abs, y_center_abs = int(x_center_abs), int(y_center_abs)
        Positionen.append((x_center_abs, y_center_abs))

    return Positionen


def Feature_Matching_ORB(
    Pfad, Treffsicherheit=0.8, Kasten_Abstand=0, Suchbereich=(0, 0, 7680, 2160)
):

    screenshot_PIL = ImageGrab.grab(
        all_screens=True, bbox=Suchbereich
    )  # Screenshot im gewünschten Bereich erstellen
    # screenhot = cv2.cvtColor(np.asarray(screenshot_PIL),cv2.COLOR_RGB2BGR)                          # Farbschema des Screenshot konvertieren
    screenhot = cv2.cvtColor(
        np.asarray(screenshot_PIL), cv2.IMREAD_GRAYSCALE
    )  # Farbschema des Screenshot konvertieren
    # small_image = cv2.imread(Pfad, cv2.COLOR_RGB2BGR)                                    # Zu suchendes Bild mit OpenCV öffnen - Farbschema Blau Grün Rot
    small_image = cv2.imread(
        Pfad, cv2.IMREAD_GRAYSCALE
    )  # Zu suchendes Bild mit OpenCV öffnen - Farbschema Blau Grün Rot

    # Initiate ORB detector
    orb = cv2.ORB_create()

    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(screenhot, None)
    kp2, des2 = orb.detectAndCompute(small_image, None)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1, des2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)

    # Draw first 10 matches.
    img3 = cv2.drawMatches(
        screenhot,
        kp1,
        small_image,
        kp2,
        matches[:10],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    plt.imshow(img3), plt.show()


def Feature_Matching_SIFT(Pfad, Treffsicherheit=0.8, Suchbereich=(0, 0, 7680, 2160)):
    screenshot_PIL = ImageGrab.grab(
        all_screens=True, bbox=Suchbereich
    )  # Screenshot im gewünschten Bereich erstellen
    # screenhot = cv2.cvtColor(np.asarray(screenshot_PIL),cv2.COLOR_RGB2BGR)                          # Farbschema des Screenshot konvertieren
    screenhot = cv2.cvtColor(
        np.asarray(screenshot_PIL), cv2.IMREAD_GRAYSCALE
    )  # Farbschema des Screenshot konvertieren
    # small_image = cv2.imread(Pfad, cv2.COLOR_RGB2BGR)                                    # Zu suchendes Bild mit OpenCV öffnen - Farbschema Blau Grün Rot
    small_image = cv2.imread(
        Pfad, cv2.IMREAD_GRAYSCALE
    )  # Zu suchendes Bild mit OpenCV öffnen - Farbschema Blau Grün Rot

    # img1 = cv2.imread('box.png',cv2.IMREAD_GRAYSCALE)          # queryImage
    # img2 = cv2.imread('box_in_scene.png',cv2.IMREAD_GRAYSCALE) # trainImage
    img1 = screenhot
    img2 = small_image
    # Initiate SIFT detector
    sift = cv2.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    # Apply ratio test
    good = []
    for m, n in matches:
        # if m.distance < 0.75*n.distance:
        if m.distance < 0.1 * n.distance:
            good.append([m])
    # cv2.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatchesKnn(
        img1,
        kp1,
        img2,
        kp2,
        good,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    plt.imshow(img3), plt.show()


def Finde_Bild_25_Prozent(
    Pfad, Treffsicherheit=0.8, Kasten_Abstand=0, Suchbereich=(0, 0, 7680, 2160)
):
    """
    Mit dieser Funktion kannst du Bilder ohne Alphakanal auf deinen Bildschirmen suchen. Dazu wird mittels PIL ein Bildschirmfoto erstellt und durch cv2 mit dem zu suchendem Bild abgeglichen.\n
    Es muss der \"Pfad\" für das zu suchende Bild gegeben werden.
    Für die Treffsicherheit empfiehlt es sich das \"decimal\" Modul zu nutzen.
    Um die Suche zu beschleunigen werden alle Bilder auf 25% ihrer Größe reduziert.\n
    Der \"Kasten_Abstand\" kann genutzt werden, um ein Objekt nicht mehrfach an fast der gleichen Stelle zu finden. Ein Wert von \"Kasten_Abstand = 0.1\" reicht in der Regel.\n
    Der Suchbereich ist wie folgt anzugeben: ([x_links_oben], [y_links_oben], [x_rechts_unten], [y_rechts_unten])
    """
    try:
        global screenhot

        if type(Pfad) != str:
            try:
                Pfad = Pfad[0]
            except:
                pass

        screenshot_PIL = ImageGrab.grab(
            all_screens=True, bbox=Suchbereich
        )  # Screenshot im gewünschten Bereich erstellen
        screenhot = cv2.cvtColor(
            np.asarray(screenshot_PIL), cv2.COLOR_RGB2BGR
        )  # Farbschema des Screenshot konvertieren
        small_image = cv2.imread(
            Pfad, cv2.COLOR_RGB2BGR
        )  # Zu suchendes Bild mit OpenCV öffnen - Farbschema Blau Grün Rot

        # resizing
        scale_percent = 25  # percent of original size
        width = int(screenhot.shape[1] * scale_percent / 100)
        height = int(screenhot.shape[0] * scale_percent / 100)
        dim = (width, height)
        # resize image
        resized_screenhot = cv2.resize(screenhot, dim, interpolation=cv2.INTER_AREA)

        width = int(small_image.shape[1] * scale_percent / 100)
        height = int(small_image.shape[0] * scale_percent / 100)
        dim = (width, height)
        # resize image
        resized_small_image = cv2.resize(small_image, dim, interpolation=cv2.INTER_AREA)

        result = cv2.matchTemplate(
            resized_screenhot, resized_small_image, cv2.TM_CCOEFF_NORMED
        )  # Eigentliche Bildsuche

        # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        # print(min_val, "--", max_val, "--", min_loc, "--", max_loc)

        breite = resized_small_image.shape[
            1
        ]  # Breite und Höhe des gesuchten Bildes bestimmen
        hoehe = resized_small_image.shape[0]

        yloc, xloc = np.where(
            result >= Treffsicherheit
        )  # Koordinaten der linken oberen Ecke aller Ergebnisse über der Schwelle bestimmen

        l = len(xloc)  # Anzahl an Rohfunden
        # print("Anzahl an Funden:", l)

        rectangles = []
        for (x, y) in zip(xloc, yloc):  # Rohfunde verdoppeln um nichtszu verlieren
            rectangles.append([int(x), int(y), int(breite), int(hoehe)])
            rectangles.append([int(x), int(y), int(breite), int(hoehe)])

        rectangles, weights = cv2.groupRectangles(rectangles, 1, Kasten_Abstand)

        Positionen = []
        for (
            x,
            y,
            breite,
            hoehe,
        ) in rectangles:  # Zentrum der Position an die Positions-Liste anfügen

            x_center, y_center = x + breite / 2, y + hoehe / 2
            # Relative Koordinaten auf absolute beziehen
            left, top, right, bottom = Suchbereich
            x_center_abs, y_center_abs = x_center * 4 + left, y_center * 4 + top
            x_center_abs, y_center_abs = int(x_center_abs), int(y_center_abs)
            Positionen.append((x_center_abs, y_center_abs))

        return Positionen

    except AttributeError as e:
        print('Fehler in der Funktion "Finde_Bild_25_Prozent"!\n\tFehlernachricht:')
        wrapper = textwrap.TextWrapper(
            width=200, initial_indent="\t\t", subsequent_indent="\t\t"
        )
        word_list = wrapper.wrap(text=str(e))
        # Print each line.
        for element in word_list:
            print(element)
        print("\tFehlerart:\n\t\t{}".format(type(e)))
        print("\tDer Pfad des zu suchenden Bildes könnte falsch sein.")
        print("\tBist du dir sicher, dass der folgende Pfad korrekt ist?")
        print("\t\t{}".format(Pfad))
        return
    except cv2.error as e:
        # print("Fehler in der Funktion \"Finde_Bild_25_Prozent\"!\n\tFehlernachricht:\n\t\t{}".format(e))
        print('Fehler in der Funktion "Finde_Bild_25_Prozent"!\n\tFehlernachricht:')
        wrapper = textwrap.TextWrapper(
            width=200, initial_indent="\t\t", subsequent_indent="\t\t"
        )
        word_list = wrapper.wrap(text=str(e))
        # Print each line.
        for element in word_list:
            print(element)
        print("\tFehlerart:\n\t\t{}".format(type(e)))
        print("\tDer gewünschte Suchbereich könnte zu klein sein.")
        print(
            "\tVeruche diesen zu vergrößern oder eine geringere Verkleinerung zu wählen."
        )
        print("\tDer gewünschte Suchbereich wurde festgelegt als:")
        print("\t\t{}".format(Suchbereich))
        return
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        # fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        # print(exc_type, fname, exc_tb.tb_lineno)
        print('Fehler in der Funktion "Finde_Bild_25_Prozent"!\n\tFehlernachricht:')
        wrapper = textwrap.TextWrapper(
            width=200, initial_indent="\t\t", subsequent_indent="\t\t"
        )
        word_list = wrapper.wrap(text=str(e))
        # Print each line.
        for element in word_list:
            print(element)
        print("\tFehlerart:\n\t\t{}".format(type(e)))
        print(f"\tFehler in Linie:\n\t\t{exc_tb.tb_lineno}")
        return


def f25(Pfad, Treffsicherheit=0.8, Kasten_Abstand=0, Suchbereich=(0, 0, 7680, 2160)):
    positionen = Finde_Bild_25_Prozent(
        Pfad,
        Treffsicherheit=Treffsicherheit,
        Kasten_Abstand=Kasten_Abstand,
        Suchbereich=Suchbereich,
    )
    return positionen


def template_matching_scaled(
    image_path, template_path=None, threshold=0.95, match_distannce=0, search_area=None, scaling=1,
):
    if template_path:
        template = cv2.imread(
            template_path, cv2.COLOR_RGB2BGR
        )  

    else:
        screenshot_PIL = ImageGrab.grab(
            all_screens=True, bbox=search_area
        )  # screenshot
        template = cv2.cvtColor(
            np.asarray(screenshot_PIL), cv2.COLOR_RGB2BGR
        )  # convert colors
    small_image = cv2.imread(
        image_path, cv2.COLOR_RGB2BGR
    ) 

    # resizing
    width = int(template.shape[1] * scaling)
    height = int(template.shape[0] * scaling)
    dim = (width, height)
    resized_template = cv2.resize(template, dim, interpolation=cv2.INTER_AREA)

    width = int(small_image.shape[1] * scaling)
    height = int(small_image.shape[0] * scaling)
    dim = (width, height)
    resized_small_image = cv2.resize(small_image, dim, interpolation=cv2.INTER_AREA)

    result = cv2.matchTemplate(
        resized_template, resized_small_image, cv2.TM_CCOEFF_NORMED
    )  

    breite = resized_small_image.shape[
        1
    ] 
    hoehe = resized_small_image.shape[0]

    yloc, xloc = np.where(
        result >= threshold
    )  

    l = len(xloc)  # amount of raw locations
    # print("Anzahl an Funden:", l)

    rectangles = []
    for (x, y) in zip(xloc, yloc):  # double raw locations to avoid losing any
        rectangles.append([int(x), int(y), int(breite), int(hoehe)])
        rectangles.append([int(x), int(y), int(breite), int(hoehe)])

    rectangles, weights = cv2.groupRectangles(rectangles, 1, match_distannce)

    positions = []
    for (
        x,
        y,
        breite,
        hoehe,
    ) in rectangles:  # add center of match to positions

        x_center, y_center = x + breite / 2, y + hoehe / 2
        # Relative Koordinaten auf absolute beziehen
        if search_area:
            left, top, right, bottom = search_area
        else:
            left, top = 0, 0
        x_center_abs, y_center_abs = int(x_center + left), int(y_center + top)
        positions.append((x_center_abs, y_center_abs))

    return positions


def feature_matching_SIFT(image_path, template_path=None, threshold=0.8, search_area=None):
    if template_path:
        template = cv2.imread(
            template_path, cv2.COLOR_RGB2BGR
        )  

    else:
        screenshot_PIL = ImageGrab.grab(
            all_screens=True, bbox=search_area
        )  # screenshot
        template = cv2.cvtColor(
            np.asarray(screenshot_PIL), cv2.COLOR_RGB2BGR
        )  # convert colors
    small_image = cv2.imread(
        image_path, cv2.COLOR_RGB2BGR
    ) 

    # Initiate SIFT detector
    sift = cv2.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(template, None)
    kp2, des2 = sift.detectAndCompute(small_image, None)
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    # Apply ratio test
    good = []
    for m, n in matches:
        # if m.distance < 0.75*n.distance:
        if m.distance < 0.1 * n.distance:
            good.append([m])
    # cv2.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatchesKnn(
        template,
        kp1,
        small_image,
        kp2,
        good,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    return matches, img3


def feature_matching_ORB(
    image_path, template_path=None, threshold=0.8, match_distannce=0, search_area=None, 
):
    if template_path:
        template = cv2.imread(
            template_path, cv2.COLOR_RGB2BGR
        )  

    else:
        screenshot_PIL = ImageGrab.grab(
            all_screens=True, bbox=search_area
        )  # screenshot
        template = cv2.cvtColor(
            np.asarray(screenshot_PIL), cv2.COLOR_RGB2BGR
        )  # convert colors
    small_image = cv2.imread(
        image_path, cv2.COLOR_RGB2BGR
    ) 

    # Initiate ORB detector
    orb = cv2.ORB_create()

    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(template, None)
    kp2, des2 = orb.detectAndCompute(small_image, None)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1, des2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)

    # Draw first 10 matches.
    img3 = cv2.drawMatches(
        template,
        kp1,
        small_image,
        kp2,
        matches[:10],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    return matches, img3



def template_matching_scaleda(
    path,
    accuracy=0.8,
    scale=25,
    hit_distance=0,
    search_area=(0, 0, 7680, 2160),
):
    """
    Mit dieser Funktion kannst du Bilder ohne Alphakanal auf deinen Bildschirmen suchen. Dazu wird mittels PIL ein Bildschirmfoto erstellt und durch cv2 mit dem zu suchendem Bild abgeglichen.\n
    Es muss der \"Pfad\" für das zu suchende Bild gegeben werden.
    Für die Treffsicherheit empfiehlt es sich das \"decimal\" Modul zu nutzen.
    Mit der \"Prozent\" wird die Skalierung angegeben. \"Prozent = 50\" bedeutet die Skalierung auf 50% der Ausgangsmaße. \"Prozent = 0.5\" ist gleichbedeutend.\n
    Der \"Kasten_Abstand\" kann genutzt werden, um ein Objekt nicht mehrfach an fast der gleichen Stelle zu finden. Ein Wert von \"Kasten_Abstand = 0.1\" reicht in der Regel.\n
    Der Suchbereich ist wie folgt anzugeben: ([x_links_oben], [y_links_oben], [x_rechts_unten], [y_rechts_unten])
    """
    Pfad = path
    Treffsicherheit = accuracy
    Prozent = scale
    Kasten_Abstand = hit_distance
    Suchbereich = search_area
    try:
        global screenhot

        if Prozent <= 1:
            Prozent = Prozent * 100

        screenshot_PIL = ImageGrab.grab(
            all_screens=True, bbox=Suchbereich
        )  # Screenshot im gewünschten Bereich erstellen
        screenhot = cv2.cvtColor(
            np.asarray(screenshot_PIL), cv2.COLOR_RGB2BGR
        )  # Farbschema des Screenshot konvertieren
        small_image = cv2.imread(
            Pfad, cv2.COLOR_RGB2BGR
        )  # Zu suchendes Bild mit OpenCV öffnen - Farbschema Blau Grün Rot

        # resizing
        scale_percent = Prozent  # percent of original size
        width = int(screenhot.shape[1] * scale_percent / 100)
        height = int(screenhot.shape[0] * scale_percent / 100)
        dim = (width, height)
        # resize image
        resized_screenhot = cv2.resize(screenhot, dim, interpolation=cv2.INTER_AREA)

        width = int(small_image.shape[1] * scale_percent / 100)
        height = int(small_image.shape[0] * scale_percent / 100)
        dim = (width, height)
        # resize image
        resized_small_image = cv2.resize(small_image, dim, interpolation=cv2.INTER_AREA)

        result = cv2.matchTemplate(
            resized_screenhot, resized_small_image, cv2.TM_CCOEFF_NORMED
        )  # Eigentliche Bildsuche

        # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        # print(min_val, "--", max_val, "--", min_loc, "--", max_loc)

        breite = resized_small_image.shape[
            1
        ]  # Breite und Höhe des gesuchten Bildes bestimmen
        hoehe = resized_small_image.shape[0]

        yloc, xloc = np.where(
            result >= Treffsicherheit
        )  # Koordinaten der linken oberen Ecke aller Ergebnisse über der Schwelle bestimmen

        l = len(xloc)  # Anzahl an Rohfunden
        # print("Anzahl an Funden:", l)

        rectangles = []
        for (x, y) in zip(xloc, yloc):  # Rohfunde verdoppeln um nichtszu verlieren
            rectangles.append([int(x), int(y), int(breite), int(hoehe)])
            rectangles.append([int(x), int(y), int(breite), int(hoehe)])

        rectangles, weights = cv2.groupRectangles(rectangles, 1, Kasten_Abstand)

        Positionen = []
        for (
            x,
            y,
            breite,
            hoehe,
        ) in rectangles:  # Zentrum der Position an die Positions-Liste anfügen

            x_center, y_center = x + breite / 2, y + hoehe / 2
            # Relative Koordinaten auf absolute beziehen
            left, top, right, bottom = Suchbereich
            x_center_abs, y_center_abs = (
                x_center / (scale_percent / 100) + left,
                y_center / (scale_percent / 100) + top,
            )
            x_center_abs, y_center_abs = int(x_center_abs), int(y_center_abs)
            Positionen.append((x_center_abs, y_center_abs))

        return Positionen

    except AttributeError as e:
        print(
            'Fehler in der Funktion "Finde_Bild_beliebiges_Prozent"!\n\tFehlernachricht:'
        )
        wrapper = textwrap.TextWrapper(
            width=200, initial_indent="\t\t", subsequent_indent="\t\t"
        )
        word_list = wrapper.wrap(text=str(e))
        # Print each line.
        for element in word_list:
            print(element)
        print("\tFehlerart:\n\t\t{}".format(type(e)))
        print("\tDer Pfad des zu suchenden Bildes könnte falsch sein.")
        print("\tBist du dir sicher, dass der folgende Pfad korrekt ist?")
        print("\t\t{}".format(Pfad))
        return
    except cv2.error as e:
        # print("Fehler in der Funktion \"Finde_Bild_beliebiges_Prozent\"!\n\tFehlernachricht:\n\t\t{}".format(e))
        print(
            'Fehler in der Funktion "Finde_Bild_beliebiges_Prozent"!\n\tFehlernachricht:'
        )
        wrapper = textwrap.TextWrapper(
            width=200, initial_indent="\t\t", subsequent_indent="\t\t"
        )
        word_list = wrapper.wrap(text=str(e))
        # Print each line.
        for element in word_list:
            print(element)
        print("\tFehlerart:\n\t\t{}".format(type(e)))
        print("\tDer gewünschte Suchbereich könnte zu klein sein.")
        print(
            "\tVeruche diesen zu vergrößern oder eine geringere Verkleinerung zu wählen."
        )
        print("\tDer gewünschte Suchbereich wurde festgelegt als:")
        print("\t\t{}".format(Suchbereich))
        return
    except Exception as e:
        print(
            'Fehler in der Funktion "Finde_Bild_beliebiges_Prozent"!\n\tFehlernachricht:'
        )
        wrapper = textwrap.TextWrapper(
            width=200, initial_indent="\t\t", subsequent_indent="\t\t"
        )
        word_list = wrapper.wrap(text=str(e))
        # Print each line.
        for element in word_list:
            print(element)
        print("\tFehlerart:\n\t\t{}".format(type(e)))
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print(f"\tFehler in Linie:\n\t\t{exc_tb.tb_lineno}")
        return


def alpha_template_match_ds_alt(Pfad, Prozent=100, Suchbereich=(0, 0, 7680, 2160)):
    """
    Mit dieser Funktion kannst du Bilder mit Alphakanal auf deinen Bildschirmen suchen. Dazu wird mittels PIL ein Bildschirmfoto erstellt und durch cv2 mit dem zu suchendem Bild abgeglichen.\n
    Es wird nur ein Ergebnis ausgegeben.\n
    Es muss der \"Pfad\" für das zu suchende Bild gegeben werden.
    Mit der \"Prozent\" wird die Skalierung angegeben. \"Prozent = 50\" bedeutet die Skalierung auf 50% der Ausgangsmaße. \"Prozent = 0.5\" ist gleichbedeutend.\n
    Der Suchbereich ist wie folgt anzugeben: ([x_links_oben], [y_links_oben], [x_rechts_unten], [y_rechts_unten])
    """
    try:
        if Prozent <= 1:
            Prozent = Prozent * 100

        screenshot_PIL = ImageGrab.grab(
            all_screens=True, bbox=Suchbereich
        )  # Screenshot im gewünschten Bereich erstellen
        screenhot = cv2.cvtColor(
            np.asarray(screenshot_PIL), cv2.COLOR_RGB2BGR
        )  # Farbschema des Screenshot konvertieren
        small_image = cv2.imread(
            Pfad, cv2.COLOR_RGB2BGR
        )  # Zu suchendes Bild mit OpenCV öffnen - Farbschema Blau Grün Rot

        # resizing
        scale_percent = Prozent  # percent of original size
        width = int(screenhot.shape[1] * scale_percent / 100)
        height = int(screenhot.shape[0] * scale_percent / 100)
        dim = (width, height)
        # resize image
        resized_screenhot = cv2.resize(screenhot, dim, interpolation=cv2.INTER_AREA)

        width = int(small_image.shape[1] * scale_percent / 100)
        height = int(small_image.shape[0] * scale_percent / 100)
        dim = (width, height)
        # resize image
        resized_small_image = cv2.resize(small_image, dim, interpolation=cv2.INTER_AREA)

        # w, h = resized_small_image.shape[::-1]
        w, h = width, height

        method1 = "cv2.TM_CCOEFF"

        method = eval(method1)

        # Apply template Matching
        # res = cv2.matchTemplate(img,template,method)
        res = cv2.matchTemplate(
            resized_screenhot, resized_small_image, method, None, resized_small_image
        )

        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        top_left = max_loc

        center = top_left[0] + w / 2, top_left[1] + h / 2

        return center

    except AttributeError as e:
        print(
            'Fehler in der Funktion "Finde_Bild_beliebiges_Prozent"!\n\tFehlernachricht:'
        )
        wrapper = textwrap.TextWrapper(
            width=200, initial_indent="\t\t", subsequent_indent="\t\t"
        )
        word_list = wrapper.wrap(text=str(e))
        # Print each line.
        for element in word_list:
            print(element)
        print("\tFehlerart:\n\t\t{}".format(type(e)))
        print("\tDer Pfad des zu suchenden Bildes könnte falsch sein.")
        print("\tBist du dir sicher, dass der folgende Pfad korrekt ist?")
        print("\t\t{}".format(Pfad))
        return
    except cv2.error as e:
        # print("Fehler in der Funktion \"Finde_Bild_beliebiges_Prozent\"!\n\tFehlernachricht:\n\t\t{}".format(e))
        print(
            'Fehler in der Funktion "Finde_Bild_beliebiges_Prozent"!\n\tFehlernachricht:'
        )
        wrapper = textwrap.TextWrapper(
            width=200, initial_indent="\t\t", subsequent_indent="\t\t"
        )
        word_list = wrapper.wrap(text=str(e))
        # Print each line.
        for element in word_list:
            print(element)
        print("\tFehlerart:\n\t\t{}".format(type(e)))
        print("\tDer gewünschte Suchbereich könnte zu klein sein.")
        print(
            "\tVeruche diesen zu vergrößern oder eine geringere Verkleinerung zu wählen."
        )
        print("\tDer gewünschte Suchbereich wurde festgelegt als:")
        print("\t\t{}".format(Suchbereich))
        return
    except Exception as e:
        print(
            'Fehler in der Funktion "Finde_Bild_beliebiges_Prozent"!\n\tFehlernachricht:'
        )
        wrapper = textwrap.TextWrapper(
            width=200, initial_indent="\t\t", subsequent_indent="\t\t"
        )
        word_list = wrapper.wrap(text=str(e))
        # Print each line.
        for element in word_list:
            print(element)
        print("\tFehlerart:\n\t\t{}".format(type(e)))
        return


def alpha_template_match_ds(
    Pfad,
    Prozent=100,
    Suchbereich=(0, 0, 0, 0),
    max_Anzahl_der_Funde=10,
    Config_Pfad=r"C:\Users\Creed\OneDrive\Schul-Dokumente\Python\Meine_einzelnen_Codes\OpenCV\alpha_template_match_ds\alpha_template_match_ds_pickles",
    force_new_threshold=False,
):
    """
    Mit dieser Funktion kannst du Bilder mit Alphakanal auf deinen Bildschirmen suchen. Dazu wird mittels PIL ein Bildschirmfoto erstellt und durch cv2 mit dem zu suchendem Bild abgeglichen.\n
    Die Ergebnisse werden in einer Liste als (x, y)-Tuple ausgegeben.\n
    Es muss der \"Pfad\" für das zu suchende Bild gegeben werden.
    Mit der \"Prozent\" wird die Skalierung angegeben. \"Prozent = 50\" bedeutet die Skalierung auf 50% der Ausgangsmaße. \"Prozent = 0.5\" ist gleichbedeutend.\n
    Der Suchbereich ist wie folgt anzugeben: ([x_links_oben], [y_links_oben], [x_rechts_unten], [y_rechts_unten]). Bei der Standard-Angabe von \"(0, 0, 0, 0)\" wird die Größe der kombinierten virtuellen Monitore ermittelt und genutzt.
    \"max_Anzahl_der_Funde\" lässt dich die maximale Anzahl an Funden einschränken. Ist für diese Anzahl noch keine Datei mit ereits ermitteltem Thrashold-Wert bekannt, wird einmalig für die gegebene Anzahl an Funden ein passender Thrashold-Wert ermittelt.
    Die Thrashold-Daten werden in einer \".pickle\"-Datei unter dem \"Config_Pfad\" gespeichert. Die Dateinamenendung weist auf die entsprechende Anzahl an Funden hin. Es kann die Neu-Bestimmung des Thrashold-Wertes erzwungen werden, indem
    \"force_new_threshold\" auf \"True\" gesetzt wird.
    """
    global Fehlernachricht
    Fehlernachricht = ""

    try:

        if Suchbereich == (0, 0, 0, 0):
            Fehlernachricht += (
                "\n\t- Suchbereich soll an die Bildschirmgröße angepasst werden."
            )
            user32 = ctypes.windll.user32
            user32.SetProcessDPIAware()
            screensize = user32.GetSystemMetrics(78), user32.GetSystemMetrics(79)
            Suchbereich = (0, 0, screensize[0], screensize[1])
        if Prozent <= 1:
            Fehlernachricht += "\n\t- Prozente wurden als Dezimalzahl angegeben und werden umgerechnet."
            Prozent = Prozent * 100

        Fehlernachricht += "\n\t- Screenshot wird gemacht."
        screenshot_PIL = ImageGrab.grab(
            all_screens=True, bbox=Suchbereich
        )  # Screenshot im gewünschten Bereich erstellen
        Fehlernachricht += "\n\t- Farben des Screenshots werden angepasst (rgb -> bgr)."
        screenhot = cv2.cvtColor(
            np.asarray(screenshot_PIL), cv2.COLOR_RGB2BGR
        )  # Farbschema des Screenshot konvertieren
        Fehlernachricht += "\n\t- Zu suchendes Bild wird gelesen."
        small_image = cv2.imread(
            Pfad, cv2.COLOR_RGB2BGR
        )  # Zu suchendes Bild mit OpenCV öffnen - Farbschema Blau Grün Rot

        # resizing
        Fehlernachricht += "\n\t- Anpassung der Größe des Screenshots wird gestartet."
        scale_percent = Prozent  # percent of original size
        width = int(screenhot.shape[1] * scale_percent / 100)
        height = int(screenhot.shape[0] * scale_percent / 100)
        dim = (width, height)
        # resize image
        resized_screenhot = cv2.resize(screenhot, dim, interpolation=cv2.INTER_AREA)

        Fehlernachricht += (
            "\n\t- Anpassung der Größe des zu suchenden Bildes wird gestartet."
        )
        width = int(small_image.shape[1] * scale_percent / 100)
        height = int(small_image.shape[0] * scale_percent / 100)
        dim = (width, height)
        # resize image
        resized_small_image = cv2.resize(small_image, dim, interpolation=cv2.INTER_AREA)

        w, h = width, height

        meth = "cv2.TM_CCOEFF"

        method = eval(meth)

        # Apply template Matching
        # res = cv2.matchTemplate(img,template,method)
        Fehlernachricht += "\n\t- Bild wird gesucht."
        res = cv2.matchTemplate(
            resized_screenhot, resized_small_image, method, None, resized_small_image
        )

        data = res
        neighborhood_size = (width + height) / 2
        # --- threshold einmalig bestimmen, dann aus .pickle Datei abrufen
        Fehlernachricht += "\n\t- Suche nach bekanntem Threshold-Wert."
        alpha_pickle_verzeichnis = Config_Pfad
        template_name = str(copy.deepcopy(Pfad))

        # template_name anpassen zu new_template_name - als pickle-Dateiname
        for i in range(1):
            template_name_list = []
            for i in template_name:
                template_name_list.append(i)
            str_to_replace = ["\\", ".", " ", "/", ":"]
            counter = 0
            for i in template_name_list:
                if i in str_to_replace:
                    template_name_list[counter] = "_"
                counter += 1
            del counter
            new_template_name = ""
            for i in template_name_list:
                new_template_name += i
            start_len = len(new_template_name)
            new_template_name_stripped = new_template_name.split(
                "Meine_einzelnen_Codes_"
            )
            new_template_name = new_template_name_stripped[-1]
            after_len = len(new_template_name)
            if after_len == start_len:
                template_name_list = []
                for i in new_template_name:
                    template_name_list.append(i)
                while len(template_name_list) > 40:
                    template_name_list.pop(0)
                new_template_name = ""
                for i in template_name_list:
                    new_template_name += i
            del (
                template_name_list,
                start_len,
                after_len,
                template_name,
                str_to_replace,
                new_template_name_stripped,
            )

        pickle_Data = (
            alpha_pickle_verzeichnis
            + "\\"
            + new_template_name
            + "_maxAnzFund_10.pickle"
        )
        my_file = Path(pickle_Data)
        (threshold, max_Anzahl_der_Funde_alt) = (None, None)

        slices = []

        if my_file.is_file() == True:
            Fehlernachricht += "\n\t- Bekannten Threshold-Wert gefunden."
            if force_new_threshold == False:
                Fehlernachricht += "\n\t- Ignoriere bekannten Threshold-Wert."
                max_Anzahl_der_Funde_alt = max_Anzahl_der_Funde - 1
            with open(pickle_Data, "rb") as f:
                (threshold, max_Anzahl_der_Funde_alt) = pickle.load(f)

        if (
            my_file.is_file() == False
            or max_Anzahl_der_Funde_alt != max_Anzahl_der_Funde
        ):
            Fehlernachricht += "\n\t- Keinen bekannten Threshold-Wert gefunden."
            Fehlernachricht += "\n\t- Ein für {} Ergebnisse geeigneter Threshold-Wert wird ermittelt und die Ergebnisse der Bildsuche werden ausgewertet.".format(
                max_Anzahl_der_Funde
            )

            def threshold_ermittlung(pickle_Data):  # threshold Anpassung
                def threshold_anpassung(threshold):
                    global Fehlernachricht
                    data_max = filters.maximum_filter(data, neighborhood_size)
                    maxima = data == data_max
                    data_min = filters.minimum_filter(data, neighborhood_size)
                    diff = (data_max - data_min) > threshold
                    maxima[diff == 0] = 0
                    labeled, num_objects = ndimage.label(maxima)
                    slices = ndimage.find_objects(labeled)
                    return slices, threshold

                global Fehlernachricht
                max_value = np.amax(res)
                threshold = max_value * 0.95

                slices, threshold = threshold_anpassung(threshold)

                while len(slices) > max_Anzahl_der_Funde:
                    threshold += max_value * 0.01
                    slices, threshold = threshold_anpassung(threshold)
                while len(slices) == 0:
                    threshold -= max_value * 0.0001
                    slices, threshold = threshold_anpassung(threshold)

                threshold_prozentual = threshold / max_value * 100

                with open(pickle_Data, "wb") as f:
                    pickle.dump((threshold, max_Anzahl_der_Funde), f)
                # print("threshold =",threshold)
                # print("Prozentual am max_value =", to_precision.std_notation(threshold_prozentual, 5),"%")

                return slices

            Fehlernachricht += "\n\t- Speichere Threshold-Wert."
            pickle_Data_List = pickle_Data.rpartition("_maxAnzFund_10.pickle")
            pickle_Data = pickle_Data_List[0] + "_maxAnzFund_{}.pickle".format(
                max_Anzahl_der_Funde
            )
            my_file = Path(pickle_Data)
            slices = threshold_ermittlung(pickle_Data)

        if len(slices) == 0:
            Fehlernachricht += "\n\t- Die Ergebnisse der Bildsuche werden ausgewertet."
            data_max = filters.maximum_filter(data, neighborhood_size)
            maxima = data == data_max
            data_min = filters.minimum_filter(data, neighborhood_size)
            diff = (data_max - data_min) > threshold
            maxima[diff == 0] = 0
            labeled, num_objects = ndimage.label(maxima)
            slices = ndimage.find_objects(labeled)

        Fehlernachricht += (
            "\n\t- Die ermittelten Positionen werden in die Positionen-Liste eingefügt."
        )
        x, y = [], []
        Positionen = []
        for dy, dx in slices:
            x_top_left = (dx.start + dx.stop - 1) / 2
            y_top_left = (dy.start + dy.stop - 1) / 2

            x_center = int(x_top_left + w / 2)
            y_center = int(y_top_left + h / 2)
            x.append(x_center)  # noch downscale
            y.append(y_center)

            x_center_abs = x_center / (scale_percent / 100)  # auf Bildschirmmaße
            y_center_abs = y_center / (scale_percent / 100)

            x_center_abs, y_center_abs = int(x_center_abs), int(y_center_abs)

            Positionen.append((x_center_abs, y_center_abs))

        # x,y als Liste -> Rechtecke markieren Ziele, Bild wird nicht gespeichert - Zentrum markiert
        for i in range(0):
            # slices = ndimage.find_objects(labeled)
            x, y = [], []
            for dy, dx in slices:
                # x_center = (dx.start + dx.stop - 1)/2
                x_top_left = (dx.start + dx.stop - 1) / 2
                # x.append(x_center)
                # x.append(x_top_left)
                # y_center = (dy.start + dy.stop - 1)/2
                y_top_left = (dy.start + dy.stop - 1) / 2
                # y.append(y_center)
                # y.append(y_top_left)

                x_center = int(x_top_left + w / 2)
                y_center = int(y_top_left + h / 2)
                x.append(x_center)
                y.append(y_center)

            print("# Koordinatenpaare:", len(x))

            # for x_top_left in x:
            #     index = x.index(x_top_left)
            #     y_top_left = y[index]
            #     top_left = (int(x_top_left), int(y_top_left))
            #     bottom_right = (int(x_top_left+w), int(y_top_left+h))

            #     cv2.rectangle(resized_screenhot,top_left, bottom_right, 255, 2)

            for x_center in x:
                index = x.index(x_center)
                y_center = y[index]
                top_left = (int(x_center - w / 2), int(y_center - h / 2))
                bottom_right = (int(x_center + w / 2), int(y_center + h / 2))

                cv2.rectangle(resized_screenhot, top_left, bottom_right, 255, 2)

            plt.subplot(121), plt.imshow(res, cmap="gray")
            plt.title("Matching Result"), plt.xticks([]), plt.yticks([])
            plt.subplot(122), plt.imshow(resized_screenhot, cmap="gray")
            plt.title("Detected Point"), plt.xticks([]), plt.yticks([])
            plt.suptitle(meth)

            # pyautogui.moveTo(x_center/(25/100), y_center/(25/100))

            plt.show()

        return Positionen

    except:
        ic(Exception)
        ic(type(Exception))
        colored_print(Fore.YELLOW + "Protokoll:", Fehlernachricht)

        Fehlernachricht_2 = (
            "\n\t- Suchbereich soll an die Bildschirmgröße angepasst werden."
        )
        Fehlerantwort_2 = "Es gab einen Fehler bei der Anpassung dese Suchbereiches an die Bildschirmgröße."

        Fehlernachricht_3 = (
            "\n\t- Prozente wurden als Dezimalzahl angegeben und werden umgerechnet."
        )
        Fehlerantwort_3 = (
            "Es gab einen Fehler bei der Umrechnung der Dezimalzahl in Prozente."
        )

        Fehlernachricht_4 = "\n\t- Screenshot wird gemacht."
        Fehlerantwort_4 = "Es gab einen Fehler beim aufnehmen des Screenshots."

        Fehlernachricht_5 = (
            "\n\t- Farben des Screenshots werden angepasst (rgb -> bgr)."
        )
        Fehlerantwort_5 = (
            "Es gab einen Fehler bei der Anpassung der Farben des Screenshots."
        )

        Fehlernachricht_6 = "\n\t- Zu suchendes Bild wird gelesen."
        Fehlerantwort_6 = "Es gab einen Fehler beim Lesen des zu suchenden Bidles."

        Fehlernachricht_7 = "\n\t- Anpassung der Größe des Screenshots wird gestartet."
        Fehlerantwort_7 = "Es gab einen Fehler bei der Größenanpassung des Screenshots."

        Fehlernachricht_8 = (
            "\n\t- Anpassung der Größe des zu suchenden Bildes wird gestartet."
        )
        Fehlerantwort_8 = (
            "Es gab einen Fehler bei der Größenanpassung des zu suchenden Bildes."
        )

        Fehlernachricht_9 = "\n\t- Bild wird gesucht."
        Fehlerantwort_9 = "Es gab einen Fehler beim Template-Match-Verfahren."

        Fehlernachricht_10 = "\n\t- Suche nach bekanntem Threshold-Wert."
        Fehlerantwort_10 = (
            "Es gab einen Fehler bei der Suche nach einem bekannten Threshold-Wert."
        )

        Fehlernachricht_11 = "\n\t- Bekannten Threshold-Wert gefunden."
        Fehlerantwort_11 = "Es gab einen unbekannten Fehler."

        Fehlernachricht_12 = "\n\t- Ignoriere bekannten Threshold-Wert."
        Fehlerantwort_12 = (
            "Es gab einen Fehler beim ignorieren des bekannten Threshold-Wertes."
        )

        Fehlernachricht_13_0 = "\n\t- Keinen bekannten Threshold-Wert gefunden."
        Fehlernachricht_13 = "\n\t- Ein für {} Ergebnisse geeigneter Threshold-Wert wird ermittelt und die Ergebnisse der Bildsuche werden ausgewertet.".format(
            max_Anzahl_der_Funde
        )
        Fehlerantwort_13 = (
            "Es gab einen Fehler bei der Auswertung des Template-Match-Ergebnisse."
        )

        Fehlernachricht_14 = "\n\t- Speichere Threshold-Wert."
        Fehlerantwort_14 = "Es gab einen Fehler beim Speichern des Threshold-Wertes."

        Fehlernachricht_15 = "\n\t- Die Ergebnisse der Bildsuche werden ausgewertet."
        Fehlerantwort_15 = (
            "Es gab einen Fehler bei der Auswertung des Template-Match-Ergebnisse."
        )

        Fehlernachricht_16 = (
            "\n\t- Die ermittelten Positionen werden in die Positionen-Liste eingefügt."
        )
        Fehlerantwort_16 = (
            "Es gab einen Fehler beim Einfügen der Positionen in die Positionen-Liste."
        )

        Fehlernachricht_Liste = [
            Fehlernachricht_2,
            Fehlernachricht_3,
            Fehlernachricht_4,
            Fehlernachricht_5,
            Fehlernachricht_6,
            Fehlernachricht_7,
            Fehlernachricht_8,
            Fehlernachricht_9,
            Fehlernachricht_10,
            Fehlernachricht_11,
            Fehlernachricht_12,
            Fehlernachricht_13_0,
            Fehlernachricht_13,
            Fehlernachricht_14,
            Fehlernachricht_15,
            Fehlernachricht_16,
        ]

        for index, Fehler in enumerate(Fehlernachricht_Liste, 2):
            if Fehlernachricht.endswith(Fehler):
                Fehlerantwort_str = "Fehlerantwort_{}".format(index)
                Fehlerantwort = eval(Fehlerantwort_str)
                colored_print(Fore.RED + Fehlerantwort)

        exc_type, exc_obj, exc_tb = sys.exc_info()
        print(f"\tFehler in Linie:\n\t\t{exc_tb.tb_lineno}")


def Bildabgleich(
    Pfad_Basis,
    Pfad_Abgleich,
    return_decimal_percentage: bool = True,
    raise_NotReadableException: bool = True,
):
    class NotReadableException(Exception):
        pass

    img_A = cv2.imread(Pfad_Basis, cv2.COLOR_RGB2BGR)
    img_B = cv2.imread(Pfad_Abgleich, cv2.COLOR_RGB2BGR)

    try:
        img_A.shape != img_B.shape
        lesbar = True
    except:
        import os
        import shutil

        try:
            tmp_folder = r"C:\Users\Creed\OneDrive\Schul-Dokumente\Programmieren\Python\Code Sammlung\Bilder\tmp-Dateien-Ordner"
            Pfad_Basis_2 = "Pfad_Basis" + os.path.splitext(Pfad_Basis)[-1]
            Pfad_Basis_2_pfad = os.path.join(tmp_folder, Pfad_Basis_2)
            shutil.copyfile(Pfad_Basis, Pfad_Basis_2)
            sleep(0.1)
            img_A = cv2.imread(Pfad_Basis_2_pfad, cv2.COLOR_RGB2BGR)
            Pfad_Abgleich_2 = "Pfad_Abgleich" + os.path.splitext(Pfad_Abgleich)[-1]
            Pfad_Abgleich_2_pfad = os.path.join(tmp_folder, Pfad_Abgleich_2)
            shutil.copyfile(Pfad_Abgleich, Pfad_Abgleich_2)
            sleep(0.1)
            img_B = cv2.imread(Pfad_Abgleich_2_pfad, cv2.COLOR_RGB2BGR)
            lesbar = False
            test = img_A.shape != img_B.shape
        except:
            print("Nicht lesbares Bild")
            try:
                os.remove(Pfad_Basis_2_pfad)
                os.remove(Pfad_Abgleich_2_pfad)
            except:
                pass
            if raise_NotReadableException:
                NotReadableException(
                    f"At least one picture is not readable. Check whether they really are pictures or not.\n\tPfad_Basis:   {Pfad_Basis}\n\tPfad_Abgleich: {Pfad_Abgleich}"
                )
            return

    if img_A.shape != img_B.shape:
        width = img_A.shape[1]
        height = img_A.shape[0]
        dim = (width, height)
        img_B = cv2.resize(img_B, dim, interpolation=cv2.INTER_AREA)
        resized = True
    else:
        resized = False

    errorL2 = cv2.norm(img_A, img_B, cv2.NORM_L2)
    similarity = 1 - errorL2 / (img_A.shape[1] * img_A.shape[0])

    if not lesbar:
        os.remove(Pfad_Basis_2_pfad)
        os.remove(Pfad_Abgleich_2_pfad)

    if return_decimal_percentage:
        return similarity
    else:
        return f"Similarity = {similarity}\n\tresized to match = {resized}"


def save_screenshot(
    path: str,
    box=None,
    timestamp_name: bool = False,
    format: str = "jpg",
    reverb: bool = True,
):
    img = ImageGrab.grab(box, all_screens=True)
    if timestamp_name:
        fp = os.path.join(path, Zeit(time()).stempel_4 + "." + format)
        sfp = fp
    else:
        if not os.path.isdir(path):
            sfp = path
        else:
            fp = os.path.join(path, Zeit(time()).stempel_4 + "." + format)
            sfp = fp
    dp = Path(sfp).parent
    if not os.path.isdir(dp):
        os.makedirs(dp)
    img.save(sfp)
    if reverb:
        colored_print(Fore.GREEN + "Image saved at:", Fore.YELLOW + sfp)


def save_screenshot_by_mouse_rect(
    path: str,
    box=None,
    timestamp_name: bool = False,
    format: str = "jpg",
    reverb: bool = True,
):
    """press ctrl for left upper corner, shift for right bottom corner and space to take the screenshot"""

    class Box:
        pass

    area = Box()
    area.luc = area.rdc = (None, None)

    def set_c1():
        area.luc = position()
        print("left upper corner:", area.luc)

    def set_c2():
        area.rdc = position()
        print("right bottom corner:", area.rdc)

    hk1 = add_hotkey("ctrl", set_c1)
    hk2 = add_hotkey("shift", set_c2)
    while True:
        wait("space")
        box = *area.luc, *area.rdc
        if any([n == None for n in box]):
            print("box:", box)
            print("None values!")
        else:
            break
    remove_hotkey(hk1)
    remove_hotkey(hk2)
    box = *area.luc, *area.rdc
    print(box)
    img = ImageGrab.grab(box, all_screens=True)
    if timestamp_name:
        fp = os.path.join(path, Zeit(time()).stempel_4 + "." + format)
        sfp = fp
    else:
        if not os.path.isdir(path):
            sfp = path
        else:
            if reverb:
                colored_print(
                    Fore.RED
                    + "Speichere Bild mit Zeitstempel-Namen, da der angegebene Pfad ein Verzeichnis ist."
                )
            fp = os.path.join(path, Zeit(time()).stempel_4 + "." + format)
            sfp = fp
    dp = Path(sfp).parent
    if not os.path.isdir(dp):
        os.makedirs(dp)
    img.save(sfp)
    if reverb:
        colored_print(Fore.GREEN + "Bild gespeichert unter:", Fore.YELLOW + sfp)
