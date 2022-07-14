import numpy as np
import cv2
import os

class Main():
    def __init__(self):
        files_path = Main.list_files(self)
        text_file = open(os.path.join(os.path.abspath(os.path.dirname(__file__)), "results.txt"), "w")
        for file_path in files_path:
            mask, result, result_edges = Main.find_red_lines(self, file_path)
            Main.print_images(self, mask, result, result_edges, file_path, text_file)
        text_file.close()

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
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        #valores máximo e mínimo para vermelho em HSV
        mask1 = cv2.inRange(image_hsv, (0,50,50), (10,255,255))
        mask2 = cv2.inRange(image_hsv, (145,30,50), (180,255,255))

        ## Merge the mask and crop the red regions
        mask = cv2.bitwise_or(mask1, mask2)
        # lower = np.array([100,50,20])
        # upper = np.array([180,255,255])
        
        #máscara para selecionar as partes da imagem que estão entre esses valores
        # mask = cv2.inRange(image_hsv, lower, upper)
        # tessst = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5)), iterations = 3)
        
        #retira da imagem original tudo aquilo que não está na máscara
        result = cv2.bitwise_and(result, result, mask=mask)

        return mask, result, image

    def print_images(self, mask, result, result_edges, file_path, text_file):
        #Utilizando algorítimo para reconhecer retangulos na imagem
        print(file_path + ":")
        cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        text_file.write(file_path + "\n")
        for cnt in cnts:
            approx = cv2.contourArea(cnt)
            if (5.0 < approx < 1500):
                x,y,w,h = cv2.boundingRect(cnt)
                cv2.rectangle(result,(x,y),(x+w,y+h),(0,255,0),2)
                print(approx)
                text_file.write(str(approx) + " ")
        text_file.write("\n")
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