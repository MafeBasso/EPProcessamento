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

        start_image_height = (image_height * 2)/10
        end_image_height = (image_height * 8)/10
        start_image_width = (image_width * 2)/10
        end_image_width = (image_width * 8)/10

        cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        file = os.path.basename(file_path).split('.')[0]

        df = pd.DataFrame()
        for cnt in cnts:
            approx = cv2.contourArea(cnt)
            x,y,w,h = cv2.boundingRect(cnt)
            area_draw_rect = w*h
            if (start_image_height < y < end_image_height and start_image_width < x < end_image_width):
                if (3.0 < approx < 2000 and approx > 2*area_draw_rect/10):
                    df = pd.concat([df, pd.DataFrame([[file, x, y, h, w, approx, area_draw_rect, image_height, image_width]])], ignore_index=True)
                    cv2.rectangle(result,(x,y),(x+w,y+h),(0,255,0),2)
                    text_file.write("w:" + str(w) + " ")
                    text_file.write("h:" + str(h) + " ")
                    text_file.write("x:" + str(x) + " ")
                    text_file.write("y:" + str(y) + " ")
                    text_file.write(str(approx) + "\n")
        
        #se ainda tem mais de 2 retângulos
        if (df.shape[0] > 2):
            #altura e largura do retangulo tem que ser maior que 5 e a aproximação da sua área tem que ser maior que 50
            df = df[(df[df.columns[3]] > 5.0) & (df[df.columns[4]] > 5.0) & (df[df.columns[5]] > 50.0)]
            df.reset_index(drop=True, inplace=True)
            #se ainda tem mais de 1 retângulo
            if (df.shape[0] > 1):
                print(df)
                center_height = image_height/2
                center_width = image_width/2
                for index, row in df.iterrows():
                    height = df.at[index, 3]
                    width = df.at[index, 4]
                    x_rect = df.at[index, 1]
                    y_rect = df.at[index, 2]
                    #se a altura do retângulo é maior que sua largura
                    if (height > width):
                        end_height_dist = y_rect + height
                        if (y_rect <= center_height <= end_height_dist):
                            df.at[index, 6] = 0.0
                            df.at[index, 9] = 'h'
                        else:
                            start_center_height_dist = abs(y_rect - center_height)
                            end_center_height_dist = abs(end_height_dist - center_height)
                            if (start_center_height_dist <= end_center_height_dist):
                                df.at[index, 6] = start_center_height_dist
                                df.at[index, 9] = 'h'
                            else:
                                df.at[index, 6] = end_center_height_dist
                                df.at[index, 9] = 'h'
                    else:
                        end_width_dist = x_rect + width
                        if (x_rect <= center_width <= end_width_dist):
                            df.at[index, 6] = 0.0
                            df.at[index, 9] = 'w'
                        else:
                            start_center_width_dist = abs(x_rect - center_width)
                            end_center_width_dist = abs(end_width_dist - center_width)
                            if (start_center_width_dist <= end_center_width_dist):
                                df.at[index, 6] = start_center_width_dist
                                df.at[index, 9] = 'w'
                            else:
                                df.at[index, 6] = end_center_width_dist
                                df.at[index, 9] = 'w'
                df.sort_values(by=[6], inplace=True)
                df = df[(df[df.columns[6]] <= 50.0)]
                print(df)
                df.reset_index(drop=True, inplace=True)

        
        text_file.write("\n")
        #configuração da janela para aparecer na tela
        # cv2.namedWindow( "mask", cv2.WINDOW_NORMAL)
        cv2.namedWindow('result '+file_path, cv2.WINDOW_NORMAL)
        # cv2.namedWindow( "result_edges", cv2.WINDOW_NORMAL)

        #mostrar na tela
        # cv2.imshow('mask', mask)
        cv2.imshow('result '+file_path, result)
        # cv2.imshow('result_edges', result_edges)
        cv2.waitKey()

        return df

if __name__ == "__main__":
    Main()