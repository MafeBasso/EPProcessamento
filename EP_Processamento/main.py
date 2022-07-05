import numpy as np
import cv2
import os

class Main():
    def __init__(self):
        files_path = Main.list_files(self)
        for file_path in files_path:
            mask, result, result_edges = Main.find_red_lines(self, file_path)
            Main.print_images(self, mask, result, result_edges)

    def list_files(self):
        dir_path = os.path.abspath(os.path.dirname(__file__))
        files_path = [os.path.join(dir_path, file) for file in os.listdir(dir_path) if file.startswith("teste")]
        return files_path

    def find_red_lines(self, file_path):
        #leitura da imagem
        image = cv2.imread(file_path)
        
        #imagem em escala de cinza
        image_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        #econtra as bordas com o algoritmo de Canny
        edges_image = cv2.Canny(image_grayscale, 50, 150)
        
        #aplica o filtro de dilatação na imagem das bordas
        dialated = cv2.dilate(edges_image, cv2.getStructuringElement(cv2.MORPH_CROSS,(4,4)), iterations = 2)
        
        #retira da imagem original tudo aquilo que não está na máscara
        result_edges = cv2.bitwise_and(image, image, mask=dialated)
        
        #criação inicial da imagem resultado como cópia da original
        result = image.copy()
        
        #conversão da imagem somente com as bordas de RGB para HSV
        image_hsv = cv2.cvtColor(result_edges, cv2.COLOR_BGR2HSV)
        
        #valores máximo e mínimo para vermelho em HSV
        lower = np.array([140,50,20])
        upper = np.array([180,255,255])
        
        #máscara para selecionar as partes da imagem que estão entre esses valores
        mask = cv2.inRange(image_hsv, lower, upper)
        
        #retira da imagem original tudo aquilo que não está na máscara
        result = cv2.bitwise_and(result, result, mask=mask)

        return mask, result, result_edges

    def print_images(self, mask, result, result_edges):
        #configuração da janela para aparecer na tela
        cv2.namedWindow( "mask", cv2.WINDOW_NORMAL)
        cv2.namedWindow( "result", cv2.WINDOW_NORMAL)
        cv2.namedWindow( "result_edges", cv2.WINDOW_NORMAL)

        #mostrar na tela
        cv2.imshow('mask', mask)
        cv2.imshow('result', result)
        cv2.imshow('result_edges', result_edges)
        cv2.waitKey()

if __name__ == "__main__":
    Main()