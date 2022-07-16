import numpy as np
import cv2
import os
import pandas as pd

class Main():
    def __init__(self):
        files_path = Main.list_files(self)
        text_file = open(os.path.join(os.path.abspath(os.path.dirname(__file__)), "results.txt"), "w")
        df_final = pd.DataFrame()
        for file_path in files_path:
            mask, result, result_edges, image_height, image_width = Main.find_red_lines(self, file_path)
            df = Main.print_images(self, mask, result, result_edges, file_path, text_file, image_height, image_width)
            df_final = df_final.append(df)
        text_file.close()
        df_final.columns = ['File', 'x', 'y', 'h', 'w', 'approx', 'w*h', 'image_height', 'image_width']
        df_final.sort_values(by=['File'], inplace=True)
        df_final.reset_index(drop=True, inplace=True)
        df_final.to_excel(os.path.join(os.path.join(os.path.abspath(os.path.dirname(__file__))), 'dataframe.xlsx'), index=False)
        
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
        # mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3)), iterations = 3)
        
        #retira da imagem original tudo aquilo que não está na máscara
        result = cv2.bitwise_and(result, result, mask=mask)

        return mask, result, image, image.shape[0], image.shape[1]

    def print_images(self, mask, result, result_edges, file_path, text_file, image_height, image_width):
        #Utilizando algorítimo para reconhecer retangulos na imagem
        text_file.write(file_path + "\n")
        text_file.write(str(image_height) + " " + str(image_width)+"\n")

        start_height = (image_height * 2)/10
        end_height = (image_height * 8)/10
        start_width = (image_width * 2)/10
        end_width = (image_width * 8)/10

        cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        file = os.path.basename(file_path).split('.')[0]

        df = pd.DataFrame()
        for cnt in cnts:
            approx = cv2.contourArea(cnt)
            x,y,w,h = cv2.boundingRect(cnt)
            multiplication = w*h
            if (start_height < y < end_height and start_width < x < end_width):
                if (3.0 < approx < 2000 and approx > 2*multiplication/10):
                    df = df.append(pd.Series([file, x, y, h, w, approx, multiplication, image_height, image_width]), ignore_index=True)
                    cv2.rectangle(result,(x,y),(x+w,y+h),(0,255,0),2)
                    text_file.write("w:" + str(w) + " ")
                    text_file.write("h:" + str(h) + " ")
                    text_file.write("x:" + str(x) + " ")
                    text_file.write("y:" + str(y) + " ")
                    text_file.write(str(approx) + "\n")

        if (df.shape[0] > 2):
            df = df[(df[df.columns[3]] > 5.0) & (df[df.columns[4]] > 5.0) & (df[df.columns[5]] > 50.0)]
            print(df)
            df.reset_index(drop=True, inplace=True)
        
        text_file.write("\n")
        #configuração da janela para aparecer na tela
        # cv2.namedWindow( "mask", cv2.WINDOW_NORMAL)
        # cv2.namedWindow('result '+file_path, cv2.WINDOW_NORMAL)
        # cv2.namedWindow( "result_edges", cv2.WINDOW_NORMAL)

        #mostrar na tela
        # cv2.imshow('mask', mask)
        # cv2.imshow('result '+file_path, result)
        # cv2.imshow('result_edges', result_edges)
        # cv2.waitKey()

        return df

if __name__ == "__main__":
    Main()