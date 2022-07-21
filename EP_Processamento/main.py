import numpy as np
import cv2
import os
import pandas as pd

class Main():
    def __init__(self):
        #caminhos das imagens dos testes
        files_path, dir_path = Main.list_files(self)
        #inicialização do dataframe final
        df_final = pd.DataFrame()
        #para cada caminho
        for file_path in files_path:
            #encontrar vermelho na imagem
            mask, image, image_height, image_width = Main.find_red(self, file_path)
            #encontrar as linhas de teste
            df, file = Main.find_lines(self, file_path, image_height, image_width, mask)
            #desenha retângulo no que foi encontrado como linha de teste
            Main.draw_found_rectangles(self, df, image)
            #salva a imagem desenhada em um diretório chamado resultados
            cv2.imwrite(os.path.join(os.path.join(dir_path, 'resultados'), os.path.basename(file_path).split('.')[0]+'.jpg'), image)
            #resultado do teste
            df_resultado = Main.write_results(self, df, file)
            #junta com o dataframe final
            df_final = pd.concat([df_final, df_resultado])
        #cria o dataframe de resultado
        Main.make_result_dataframe(self, df_final, dir_path)
        
    def list_files(self):
        #caminho do diretório desse arquivo
        dir_path = os.path.abspath(os.path.dirname(__file__))
        
        #todos os caminhos dos arquivos com 'teste'
        files_path = [os.path.join(dir_path, file) for file in os.listdir(dir_path) if file.startswith("teste")]
        
        #retorna os caminhos e o diretório
        return files_path, dir_path

    def find_red(self, file_path):
        #aqui foi utilizada a referência de encontrar vermelho na imagem: https://stackoverflow.com/questions/51225657/detect-whether-a-pixel-is-red-or-not/51228567#51228567
        
        #leitura da imagem
        image = cv2.imread(file_path)
        
        #conversão da imagem para HSV
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        #máscaras dos valores máximos e mínimos para vermelho em HSV encontrados na imagem em HSV
        mask1 = cv2.inRange(image_hsv, (0,50,50), (10,255,255))
        mask2 = cv2.inRange(image_hsv, (145,30,50), (180,255,255))

        #junção das máscaras
        mask = cv2.bitwise_or(mask1, mask2)

        #retorna a máscara, imagem e altura e largura da imagem
        return mask, image, image.shape[0], image.shape[1]

    def find_lines(self, file_path, image_height, image_width, mask):
        #nome do teste (ignora o caminho e a extensão do arquivo)
        file = os.path.basename(file_path).split('.')[0]

        #inicializa um dataframe para a imagem
        df = pd.DataFrame(columns=['file', 'x', 'y', 'h', 'w', 'approx'])

        #variáveis para ignorar 20% da imagem nas bordas
        start_image_height = (image_height * 2)/10
        end_image_height = (image_height * 8)/10
        start_image_width = (image_width * 2)/10
        end_image_width = (image_width * 8)/10

        #aqui foi utilizada a referência de encontrar contornos e o retângulo referente a cada contorno: https://www.pythonpool.com/cv2-boundingrect/
        #encontra os contornos da máscara
        cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        #para cada contorno
        for cnt in cnts:
            #área aproximada do contorno
            approx = cv2.contourArea(cnt)
            #retângulo equivalente do contorno, pontos x e y, altura e largura h e w
            x,y,w,h = cv2.boundingRect(cnt)
            #area do retângulo acima
            area_draw_rect = w*h
            #ignora o retângulo se seu x e y se encontra nos 20% das bordas da imagem
            if (start_image_height < y < end_image_height and start_image_width < x < end_image_width):
                #ignora o retângulo que não tem área maior que 3 e menor que 2000
                #e que a área aproximada do contorno não ocupa mais de 20% da área do retângulo
                if (3.0 < approx < 2000 and approx > 2*area_draw_rect/10):
                    #insere uma linha no dataframe nome do teste, variáveis do retângulo e área aproximada do contorno
                    df = pd.concat([df, pd.DataFrame([[file, x, y, h, w, approx]], columns=['file', 'x', 'y', 'h', 'w', 'approx'])], ignore_index=True)
                    
        #se ainda tem mais de 2 retângulos (linhas no dataframe)
        if (df.shape[0] > 2):
            #como encontrou muito vermelho (mais de 2 retângulos)
            #então os retângulos que contém as prováveis linhas tem que ter:
            #altura e largura maior que 5 e área aproximada de seu contorno maior que 50
            df = df[(df[df.columns[3]] > 5.0) & (df[df.columns[4]] > 5.0) & (df[df.columns[5]] > 50.0)]
            
            #reseta os índices do dataframe
            df.reset_index(drop=True, inplace=True)
            
            #se ainda tem mais de 1 retângulo
            if (df.shape[0] > 1):
                #variáveis de centro da imagem
                center_height = image_height/2
                center_width = image_width/2
                #para cada linha do dataframe
                for index in range(0, df.shape[0]):
                    #variáveis de altura, largura e pontos x e y do retângulo inserido na linha
                    height = df.at[index, 'h']
                    width = df.at[index, 'w']
                    x_rect = df.at[index, 'x']
                    y_rect = df.at[index, 'y']
                    #se a altura do retângulo é maior que sua largura
                    if (height > width):
                        #ponto onde termina a altura do retângulo
                        end_height_dist = y_rect + height
                        #se o centro da altura da imagem está contido no retângulo em relação a sua altura
                        if (y_rect <= center_height <= end_height_dist):
                            #guarda nessa linha do dataframe que a distância entre o retângulo e o centro da imagem é zero (em relação a altura)
                            df.at[index, 6] = 0.0
                            #guarda o caractere 'h' nessa linha do dataframe para identificar que a distância inserida anteriormente é relativa a altura
                            df.at[index, 7] = 'h'
                        #caso contrário
                        else:
                            #calcula a distância em módulo do ponto inicial (y) e final do centro da imagem em relação a altura
                            start_center_height_dist = abs(y_rect - center_height)
                            end_center_height_dist = abs(end_height_dist - center_height)
                            #se a variável em relação ao início do retângulo é menor que a em relação ao final dele
                            if (start_center_height_dist < end_center_height_dist):
                                #guarda nessa linha do dataframe a distância entre a variável em relação ao início do retângulo e o centro da imagem (em relação a altura)
                                df.at[index, 6] = start_center_height_dist
                                #guarda o caractere 'h' nessa linha do dataframe para identificar que a distância inserida anteriormente é relativa a altura
                                df.at[index, 7] = 'h'
                            #caso contrário
                            else:
                                #guarda nessa linha do dataframe a distância entre a variável em relação ao final do retângulo e o centro da imagem (em relação a altura)
                                df.at[index, 6] = end_center_height_dist
                                #guarda o caractere 'h' nessa linha do dataframe para identificar que a distância inserida anteriormente é relativa a altura
                                df.at[index, 7] = 'h'
                    #caso contrário
                    else:
                        #ponto onde termina a largura do retângulo
                        end_width_dist = x_rect + width
                        #se o centro da largura da imagem está contido no retângulo em relação a sua largura
                        if (x_rect <= center_width <= end_width_dist):
                            #guarda nessa linha do dataframe que a distância entre o retângulo e o centro da imagem é zero (em relação a largura)
                            df.at[index, 6] = 0.0
                            #guarda o caractere 'w' nessa linha do dataframe para identificar que a distância inserida anteriormente é relativa a largura
                            df.at[index, 7] = 'w'
                        #caso contrário
                        else:
                            #calcula a distância em módulo do ponto inicial (x) e final do centro da imagem em relação a largura
                            start_center_width_dist = abs(x_rect - center_width)
                            end_center_width_dist = abs(end_width_dist - center_width)
                            #se a variável em relação ao início do retângulo é menor que a em relação ao final dele
                            if (start_center_width_dist <= end_center_width_dist):
                                #guarda nessa linha do dataframe a distância entre a variável em relação ao início do retângulo e o centro da imagem (em relação a largura)
                                df.at[index, 6] = start_center_width_dist
                                #guarda o caractere 'w' nessa linha do dataframe para identificar que a distância inserida anteriormente é relativa a largura
                                df.at[index, 7] = 'w'
                            #caso contrário
                            else:
                                #guarda nessa linha do dataframe a distância entre a variável em relação ao final do retângulo e o centro da imagem (em relação a largura)
                                df.at[index, 6] = end_center_width_dist
                                #guarda o caractere 'w' nessa linha do dataframe para identificar que a distância inserida anteriormente é relativa a largura
                                df.at[index, 7] = 'w'
                
                #ordena as linhas do dataframe pela distância calculada anteriormente
                df.sort_values(by=[6], inplace=True)
                
                #remove aquelas linhas em que a distância calculada anteriormente é maior que 50
                #isso por causa do corte nas imagens, espera-se que as linhas estejam próximas ao centro (em relação a orientação vertical ou horizontal das imagens)
                df = df[(df[df.columns[6]] <= 50.0)]
                
                #reseta os índices do dataframe
                df.reset_index(drop=True, inplace=True)

                #se ainda tem mais de 1 retângulo
                if (df.shape[0] > 1):
                    #variável para parar o for mais externo, inicializada com false
                    stop = False
                    #para cada linha do dataframe menos a última
                    for index in range(0, df.shape[0]-1):
                        #se a variável citada anteriormente for verdadeira, o for é encerrado
                        if(stop == True):
                            break
                        #para cada linha seguinte do dataframe (incluindo a última)
                        for index_next_rect in range(index+1, df.shape[0]):
                            #se a entratégia da distância do centro usada foi a mesma para as duas linhas 'h' ou 'w'
                            if(df.at[index, 7] == df.at[index_next_rect, 7]):
                                #se a estratégia foi a de altura, 'h'
                                if(df.at[index, 7] == 'h'):
                                    #diferença em módulo dos pontos y das linhas
                                    y_proximit = abs(df.at[index, 'y'] - df.at[index_next_rect, 'y'])
                                    #se a diferença é menor ou igual a 3
                                    if (y_proximit <= 3.0):
                                        #seleciona essas duas linhas (retângulos) como as linhas positivas do teste
                                        #por causa da próximidade desses retângulos do centro e proximidade do ponto y entre os dois
                                        #é muito provável que eles sejam as linhas positivas do teste
                                        df = pd.concat([df.loc[[index]], df.loc[[index_next_rect]]])
                                        #reseta os índices do dataframe
                                        df.reset_index(drop=True, inplace=True)
                                        #iguala a variável stop a true para encerrar o for externo
                                        stop = True
                                        #encerra o for interno
                                        break
                                #se a estratégia foi a de largura, 'w'
                                elif(df.at[index, 7] == 'w'):
                                    #diferença em módulo da altura dos retângulos das linhas
                                    h_proximit = abs(df.at[index, 'h'] - df.at[index_next_rect, 'h'])
                                    #se a diferença é menor que 2
                                    if (h_proximit < 2.0):
                                        #seleciona essas duas linhas (retângulos) como as linhas positivas do teste
                                        #por causa da próximidade desses retângulos do centro e proximidade entre suas alturas
                                        #é muito provável que eles sejam as linhas positivas do teste
                                        df = pd.concat([df.loc[[index]], df.loc[[index_next_rect]]])
                                        #reseta os índices do dataframe
                                        df.reset_index(drop=True, inplace=True)
                                        #iguala a variável stop a true para encerrar o for externo
                                        stop = True
                                        #encerra o for interno
                                        break
            #se existir as colunas a mais que foram criadas para a estratégia acima
            if (df.shape[1] == 7):
                #deleta essas colunas do dataframe
                df = df.drop(7, axis=1)
                df = df.drop(6, axis=1)
        
        #retorna o dataframe e o nome do teste
        return df, file

    def draw_found_rectangles(self, df, image):
        #para cada linha do dataframe
        for index in range(0, df.shape[0]):
            #variáveis de altura, largura e pontos x e y do retângulo inserido na linha
            height = df.at[index, 'h']
            width = df.at[index, 'w']
            x = df.at[index, 'x']
            y = df.at[index, 'y']
            #aqui é utilizada a mesma fonte para desenhar o retângulo: https://www.pythonpool.com/cv2-boundingrect/
            #desenha o retângulo na imagem
            cv2.rectangle(image,(x,y),(x+width,y+height),(0,255,0),2)

    def write_results(self, df, file):
        #se o dataframe estiver vazio
        if(df.shape[0] == 0):
            #dataframe com o nome do teste e resultado como erro de leitura
            df = pd.DataFrame([[file, 'erro de leitura']])
        #se o dataframe conter só 1 linha
        elif(df.shape[0] == 1):
            #dataframe com o nome do teste e resultado como negativo
            df = pd.DataFrame([[file, 'negativo']])
        #se o dataframe conter só 2 linhas
        elif(df.shape[0] == 2):
            #dataframe com o nome do teste e resultado como positivo
            df = pd.DataFrame([[file, 'positivo']])
        #se o dataframe conter mais linhas
        else:
            #dataframe com o nome do teste e resultado como inconclusivo
            df = pd.DataFrame([[file, 'inconclusivo']])
        
        #retorna o dataframe
        return df

    def make_result_dataframe(self, df_final, dir_path):
        #nomeia as colunas do dataframe
        df_final.columns = ['Imagem', 'Resultado']
        #ordena os valores por nome da imagem
        df_final.sort_values(by=['Imagem'], inplace=True)
        #reseta os índices do dataframe
        df_final.reset_index(drop=True, inplace=True)
        #lê o dataframe dos resultados esperados
        df_expected = pd.read_excel(os.path.join(dir_path, 'classificacao_das_imagens.xlsx'))
        #merge dos dataframes
        df_final = df_final.merge(df_expected, left_on='Imagem', right_on='ID', suffixes=(False, False))
        #se o resultado esperado é igual ao resultado encontrado insere 1 senão insere 0 nessa linha em nova coluna 'Iguais'
        df_final['Iguais'] = np.where(df_final['Resultado']==df_final['Classe'], 1, 0)
        #renomeia a coluna 'Classe' para 'Esperado'
        df_final.rename(columns = {'Classe':'Esperado'}, inplace=True)
        #remove a coluna 'ID'
        df_final.drop(columns=['ID'], inplace=True)
        #porcentagem de acerto do algoritmo com 2 casas decimais
        hit_percentage = round(100*(df_final['Iguais'].sum())/(df_final.shape[0]), 2)
        #insere a porcentagem de acerto no fim do dataframe na coluna 'Iguais'
        df_final = pd.concat([df_final, pd.DataFrame([[np.NaN, np.NaN, np.NaN, hit_percentage]], columns=['Imagem', 'Resultado', 'Esperado', 'Iguais'])], ignore_index=True)
        #cria um excel chamado resultado com o dataframe final
        df_final.to_excel(os.path.join(dir_path, 'resultado.xlsx'), index=False)
        
if __name__ == "__main__":
    Main()