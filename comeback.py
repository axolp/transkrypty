import cv2
import pytesseract
from pytesseract import Output
from pypinyin import lazy_pinyin, Style
import numpy as np

def find_black_column(frame, side, margin):
    
    image= frame[1]
    #wysokosc
    #print("wysokosc", image.shape[0])

    #szerokosc
    #print("szerokosc", image.shape[1])

    #print(image[0][0])
    n_rows, n_columns, = image.shape[0], image.shape[1]

    if side == 'left':
       left= 0
       for i in range(n_columns):
            for j in range(n_rows):
                if image[j][i] == 0:
                    left= i
                    break

            if left != 0:
                return left - margin
       return 0
    
    else:
        right= 0
        for i in range(n_columns):
            for j in range(n_rows):
                if image[j][n_columns-i-1] == 0:
                    right= n_columns-i-1
                    break

            if right != 0:
                return right + margin
        return n_columns

def dilate_image(image, kernel_size, iterations):
    inverted_image = cv2.bitwise_not(image)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_image = cv2.dilate(inverted_image, kernel, iterations=iterations)
    final_image = cv2.bitwise_not(dilated_image)
    return final_image


def add_border(image, thickness):
    top, bottom, left, right = [thickness]*4
    # Create the border around the image
    bordered_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return bordered_image

# Konfiguracja ścieżki do Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\PC\Desktop\Uczelnia\projekty\chinski\tessaract\tesseract.exe'  # Zmień ścieżkę według swojej instalacji
custom_config = r'--psm 7'
starting_second= 1
# Odczyt wideo
video = cv2.VideoCapture('polyglot.mp4')
# Ustalanie framerate
fps = video.get(cv2.CAP_PROP_FPS)
#print("FPS: ", fps)
interval = int(fps*0.5)  # Co 0,5 sekundy

video.set(cv2.CAP_PROP_POS_MSEC, starting_second*1000)

# Przetwarzanie klatek
prev_l, prev_r= -69, -69
frame_number = fps*starting_second -1# bo dodaje na poczatku
sekunda= starting_second
lyrics= {

}
while True:
    success, frame = video.read()
    frame_number += 1
    if not success:
        break

    x= 200
    y= 601
    h= 45
    w= 825

    sekunda= (frame_number / fps)
    #teraz dziala
    # Przetwarzanie co 0,5 sekundy
    if frame_number % interval == 0 :
        
        # Wyciągnij określony obszar ekranu (x, y, szerokość, wysokość)
        roi = frame[y:y+h, x:x+w]
        gray_roi= cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        final_roi= cv2.threshold(gray_roi, 175, 255, cv2.THRESH_BINARY_INV)
        
        #sum_cols = cv2.reduce(final_roi, 0, cv2.REDUCE_SUM, dtype=cv2.CV_32S)
        #left = cv2.findNonZero(sum_cols.T)[0][0]
        #right = cv2.findNonZero(sum_cols.T)[-1][0]

        # Przycięcie ROI
        #cropped_roi = final_roi[:, left:right]
        
        #cv2.imwrite(f"roi{frame_number}.jpg", final_roi)
        l= find_black_column(final_roi, 'left', 7)
        r= find_black_column(final_roi, 'right', 7)
        #print(l, r)

        if l ==0:
            continue
        if l == prev_l and r == prev_r:
            #print("skipuje bo to samo")
            continue

        #print(final_roi.shape)
        #try:    
        final_roi= final_roi[1]
        #print("final roi: ", final_roi)
        cropped_roi = final_roi[0:40, l:r]
        
        final_image= dilate_image(cropped_roi, 2, 1)
        final_image= add_border(final_image, 2)
      
        

        
        #cv2.imwrite(f"aftercut{frame_number/fps}.jpg", final_image)
            # Stosowanie OCR
        text = pytesseract.image_to_string(final_image, lang='chi_sim', config=custom_config)  # Użyj 'chi_sim' dla uproszczonego chińskiego
        text = text.rstrip('\n')

        pinyin= lazy_pinyin(text, style=Style.TONE3)
        #except:
            #text= "error"
        # Zapis do pliku
        with open('polyglot_pinyin.txt', 'a', encoding='utf-8') as file:
            file.write(f'{frame_number//fps}:  "{text}",\n')
        lyrics[frame_number//fps]= [text]#tutaj moge dodac pinyin
        print(lyrics)

    try:
       
        prev_l, prev_r= l, r
    except:
        print("nie aktualizuje numeru klatek")
    
with open('polyglot_dict.txt', 'a', encoding='utf-8') as file:
            file.write(str(lyrics))
video.release()

