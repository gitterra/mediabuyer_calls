# Работа с массивами данных
import numpy as np 

# Загрузка датасетов из облака google
import gdown

# Функции операционной системы
import os

# Загрузка моделей
from keras.models import load_model

# Запись в файлы и чтение из файлов структур данных Python
import pickle

# Модуль pandas для обработки табличных данных
import pandas as pd

# Модуль для проигрывания аудио в colab
import IPython.display as ipd 

# Пакет для анализа аудио
import librosa

# Визуализация аудио
import librosa.display 

# Загрузка и сохренение аудио файлов
import soundfile as sf

from IPython.display import Audio # для воспроизведение np. массива => Audio(x[350000:500000], rate=22050)

import sklearn #Для нормирования

from scipy.signal import argrelextrema

from time import time

# Печать текста с подсветкой
from IPython.display import display, Markdown

# Конвертация в WAV-файл
from pydub import AudioSegment

# Библиотека с популярными сервисами распознавания речи
import speech_recognition as sR

# Библиотека Yandex Speech Kit (платная)

from speechkit import Session, SpeechSynthesis, ShortAudioRecognition

# OAuth-токен для авторизации
oauth_token = "y0_AgAAAABdthwBAATuwQAAAADVdFJ07eBrIqfzQWqI9LU62LX7zYqCBHg"
# Идентификатор каталога
catalog_id = "b1g3lhmmjghjk5jkbd15"

from collections import Counter

from natasha import (
    Segmenter,
    MorphVocab,
    
    NewsEmbedding,
    NewsMorphTagger,
    NewsSyntaxParser,
    NewsNERTagger,
    
    PER,
    # LOC,
    NamesExtractor,
    # DatesExtractor,
    # MoneyExtractor,
    # AddrExtractor,

    Doc
)

segmenter = Segmenter()
morph_vocab = MorphVocab()

emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
syntax_parser = NewsSyntaxParser(emb)
ner_tagger = NewsNERTagger(emb)

names_extractor = NamesExtractor(morph_vocab)

from pullenti_wrapper.langs import (
    set_langs,
    RU
)
set_langs([RU])

from pullenti_wrapper.referent import Referent

from pullenti_wrapper.processor import (
    Processor,
    MONEY,
    PHONE,
    DATE,
    DEFINITION,
    GEO,
    ADDRESS,
    ORGANIZATION,
    PERSON,
    DECREE,
    TITLEPAGE,
    BOOKLINK
)

processor = Processor([
    PERSON,
    ORGANIZATION
])

# Фильтрация предупреждений, чтобы не librosa.load не ругалась на mp3
import warnings
warnings.filterwarnings('ignore')

"""## 0. Функции для ограничения длины строки выводимых текстов"""

MAX_SYMBOLS = 100     # Максимальная длина строки при выводе текстов
MAX_LINE = 50         # Состоит из удвоенных символов

# Вспомогательная функция ограничения ширины печатаемого текста (возможно есть штатные средства?)
def print_lim(text, sym_lim = MAX_SYMBOLS):
  if len(text) < sym_lim: print(text)
  else:
    st_sym = 0
    en_sym = sym_lim
    while en_sym < len(text):
      while text[en_sym] != ' ' and en_sym != st_sym:
        en_sym -= 1
      if en_sym == st_sym:    # Выход из цикла когда пробел та и не был найден
        print(text[st_sym:st_sym + sym_lim])
        st_sym = st_sym + sym_lim
      else:                   # Выход из цикла когда найден пробел
        print(text[st_sym:en_sym])
        st_sym = en_sym + 1   # Найденный пробел в начале следующей строки уже не нужен    
      en_sym = st_sym + sym_lim
    print(text[st_sym:])

# Вспомогательная функция ограничения ширины печатаемого текста (cut_sym для print - '\n', а для Markdown - '<br>')
def text_cutter(text, cut_sym = '\n', sym_lim = MAX_SYMBOLS):
  text = text.replace('\n',' ')
  text = text.replace('  ',' ')
  out_text = ''
  if len(text) < sym_lim: out_text += text
  else:
    st_sym = 0
    en_sym = sym_lim
    while en_sym < len(text):
      while text[en_sym] != ' ' and en_sym != st_sym:
        en_sym -= 1
      if en_sym == st_sym:    # Выход из цикла когда пробел та и не был найден
        out_text += text[st_sym:st_sym + sym_lim] + cut_sym
        st_sym = st_sym + sym_lim
      else:                   # Выход из цикла когда найден пробел
        out_text += text[st_sym:en_sym] + cut_sym
        st_sym = en_sym + 1   # Найденный пробел в начале следующей строки уже не нужен      
      en_sym = st_sym + sym_lim
    out_text += text[st_sym:]
  return out_text

"""## 1. Удаление автоответчика и музыки"""

def get_hstgrm_1D(arr_max_frmnt, num_fr, prnt ):
    """ Готовим гистограмму из 1D массива (сколько раз встречается данное значение)"""
    # print('данных =', len(arr_max_frmnt))

    hstgrm_arr = np.array( [0] * (num_fr ))

    for elem in arr_max_frmnt:    
        if elem >= num_fr:
            continue
        hstgrm_arr[elem] += 1

    if prnt:
        for f in range(len(hstgrm_arr)):
            print(f, ')  =---> ', hstgrm_arr[f])

    return  hstgrm_arr

def Fourier(sound_Signal, max_frequency):

    X = librosa.stft(sound_Signal[:])  #Вычисляем спектр сигнала 
    Xdb = librosa.amplitude_to_db(abs(X)) # В вещ. числа переводим амплитуду + меняем шкалу на децибелы
   
    N_spectr = Xdb.shape[1] # к-во спектров всего (1 спектр на основе 512 остчетов)
    # print('Преобр. Фурье:  Всего спектров =', N_spectr, '   исх. сигн.:',  sound_Signal.shape)     
    
    min_DB = np.min(Xdb)
    MAX_FREQ = 250
    spectr_Signal = Xdb[:][0:max_frequency] - min_DB  #   + 45.43089676 
        
    return spectr_Signal

def cut_all_music(Signal, min_important_time = 11):  
    """ Подаем на вход исходный сигнал (np.arr). Получаем сигнал без музыки. Музыка будет вырезана и вначале, и в конце, и в середине  
        Время минимального значимого фрагмента в секундах: default = 11 с."""
    # Логика такая::
    # 1) 1-ый значимый фрагмент (> 11 сек. без муз.) будет записан только после того как закончится 1-я музыка.
    # 2) Если музыки не будет вообще, то запишется всё как было в исходном сигнале.
    # 3) Все фрагменты с музыкой будут вырезаны и вначале, и в конце, и в середине без ограничения кол-ва повторов.
    # 4) Если речь автоответчика будет длиться без музыки более 11 сек. (кроме первого фрагмента) она будет записана как значимый фрагмент.
    # 5) Добавлен фильтр однотонного гула, гул теперь не мешает.  
    # 6) Все фрагменты без музыки будут склеены в один. 
 
    if Signal.shape[0] < 300000:
        return [0]
    
    MAX_FREQ = 250
    t1 = time() # ********* Сначала ФУРЬЕ (спектр) ****************
    Xnorm = Fourier(Signal, MAX_FREQ)           
    N_spectr =  Xnorm.shape[1]

    MIN_TARGET_TIME = 43 * min_important_time # 43 спектра/сек умножить на число сек. => минимально значимый разговор.    = 43 * 11 

    FIX_TIME = 21 #   === 25 === длительность кусочка, в котором анализируем муз. ноты (43/сек)
    MUSIC_TAKT = 0.61 * FIX_TIME # === 0.55 ===    = 0.65 * FIX_TIME    = 0.61 * FIX_TIME
    MIN_FREQ_FOR_MUS = 20 # 20
    MIN_MEAN_A = 0.15 * np.mean(Xnorm)   
    MIN_MAX = 0.3511111 * np.max(Xnorm)    
    N_SMAL = 4
    NOISE_EXEPTION_LEVEL = 0.28 * N_spectr # —> Подавление гула. При увеличении коэф-та снижается влияние  

    flag_music = False
    next_point_after_music = 0
    
    fix_t = 0
     
    ok_start = 0
    ok_end = -1
    count_time_no_music = 0
    all_pieces_no_mus = []

       
    arr_to_all_hstgrm = []    
    for ix in range(N_spectr): #FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF  
        arr_i_max = argrelextrema(Xnorm[:MAX_FREQ, ix], np.greater_equal, order = N_SMAL )[0]       
        arr_to_all_hstgrm.extend(arr_i_max)
        # print(' arr_i_max =', arr_i_max)
    #FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
  
    hstgrm = get_hstgrm_1D(arr_to_all_hstgrm, MAX_FREQ, prnt=False ) 

    exception_list = []    
    i = -1
    for one_fr in hstgrm: #ffffffffffffffffffffffffffffffff
        i += 1
        if one_fr > NOISE_EXEPTION_LEVEL  and  i > 5:
            exception_list.append(i)
            # print(i, ') исключение --> ', one_fr)
    #ffffffffffffffffffffffffffffffffFFFFFFFFFFFFFFFFFFFFFF  


    arr_to_fix_hstgrm = [] 
    start_music = 0
    for ix in range(N_spectr): #FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
              
        arr_i_max = argrelextrema(Xnorm[:MAX_FREQ, ix], np.greater_equal, order = N_SMAL )[0] 
        
        if fix_t >= FIX_TIME: 
            fix_t = 0
            fix_hstgrm = get_hstgrm_1D(arr_to_fix_hstgrm, MAX_FREQ, prnt=False )           
            max_of_fix_hstg = argrelextrema(fix_hstgrm, np.greater_equal, order = N_SMAL )[0]

            cnt_big_max = 0
            for one_max in max_of_fix_hstg:                
                if fix_hstgrm[one_max] > MUSIC_TAKT and one_max > MIN_FREQ_FOR_MUS:
                    cnt_big_max += 1    
                    # print ('   ', cnt_big_max, ')   частота =' ,  one_max, ' Ampl =', fix_hstgrm[one_max])  ##############--------------------  

            if cnt_big_max >= 3:                    
                flag_music = True 
                count_time_no_music = 0     #

                start_music = max(ix - FIX_TIME, 1) 
                # print(ix, ') ==> 1. ++++++++++ end of music =', next_point_after_music  )   ##############--------------------
                if ok_start : 
                    ok_end = (start_music - 8)   # снова началась музыка, но уже было время для разговора  !!!!! ЗДЕСЬ +++
                    all_pieces_no_mus.append([ok_start, ok_end]) # записали кусок без музыки
                    ok_start = 0                         # и всё начинаем сначала
                    ok_end = -1
                    continue                  
                               
            elif flag_music  :    # Закончилась музыка
                flag_music = False                                          
                next_point_after_music = (ix - FIX_TIME + 8) +  int( np.max(fix_hstgrm) *1.5)
                count_time_no_music = 1
                # print('>>>',ix, ') ==> 2. ++++++++++ end of music =', next_point_after_music  ) 

            else: 
                count_time_no_music += FIX_TIME                                                 ##############++++++++++++++++++++
                if next_point_after_music and count_time_no_music > MIN_TARGET_TIME and ok_start == 0:  #   был звук дост. время    
                    ok_start = next_point_after_music                                             #  !!!!! ЗДЕСЬ +++ 
                    # print(ix, ')   Есть начало!  ok_start =', next_point_after_music  )  ##############++++++++++++++++++++


            arr_to_fix_hstgrm = []  
        
        else: ### /if fix_t >= FIX_TIME: ### &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
            fix_t += 1 
            mean_A_at_1_spect = np.mean(Xnorm[:, ix])           
            if mean_A_at_1_spect  > MIN_MEAN_A :
                for one_max in arr_i_max:
                    if one_max <= 5 or one_max in exception_list:
                        continue
                    if one_max > MAX_FREQ -10:
                        break                    
                    if Xnorm[one_max, ix] > MIN_MAX :
                        arr_to_fix_hstgrm.append(one_max)                
      
    #FFFFFFFFFFFFFFFFFFFFFFFFFFFFFF  /for ix in range(N_spectr)  FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
    
    clear_Signal = np.array([])
    clear_Spectr = np.array([])

    if ok_start and  ok_end == -1:
        # clear_Signal = np.append(clear_Signal, Signal[ok_start : ] )
        all_pieces_no_mus.append([ok_start, -1])

    if len(all_pieces_no_mus) == 0 and start_music == 0: # Не было музыки, отдаем всё без изменений
        return Signal

    
    if len( all_pieces_no_mus ) > 0:
        for one_piece in all_pieces_no_mus: ### ffffffffffffffffffffffffffffffffffffffff
            clear_Signal = np.append(clear_Signal, Signal[ one_piece[0]*512 : one_piece[1]*512 ] )
            # clear_Spectr = np.append(clear_Spectr, Xnorm[ one_piece[0] : one_piece[1] ] )
        ### ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff    
        # print('all_pieces_no_mus =>', all_pieces_no_mus)    
        return  clear_Signal
    else:
        # print('----- Не было РАЗГОВОРА -----')
        return []

"""## 2. Нарезка разговора на фразы V3"""

def phrase_by_phrase(Signal_without_musik):
    """ Деление всего диалога на сегменты. На вход — сигнал без музыки. 
        На выходе список отсчетов входного (сюда) сигнала: [ [st0, end0], [st1, end1], ...]  """  

    # ===================  Вот эти самые ВАЖНЫЕ константы здесь: ================================== **1    
    MAX_N_PAUSE = 15 # спектров от 8 до 55
    MIN_N_PAUSE = 8  # спектров
    TARGET_TIME = 25 # сек. целевая длительность фразы
    K_PAUSE = (MAX_N_PAUSE - MIN_N_PAUSE)/(TARGET_TIME*43)
    N_PAUSE = MAX_N_PAUSE # +++++: Фраза будет закончена, если подряд N_PAUSE спектров слабы —> пауза 
    # :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    N_VOICE = 8  #  +++++: Начнем новую фразу, если подряд N_VOICE спектров: уровнь звука > LEVEL_Z  
    NOT_SOUND_KOEF = 0.31 # +++++: определяет звуковой порог, ниже которого считаем, что нет звука 0.31
    LEN_PHRASE_TO_IGNORE = 35 ###  кол-во спектров в слишком короткой фразе: 43 — это секунда 
    # ===================  /Вот эти самые ВАЖНЫЕ константы  ======================================= 

    list_of_segment = []
    list_of_spectr = []
    list_of_timing = []
    len_phrase = 0

    MAX_FREQ = 250
    cut_Spectr = Fourier(Signal_without_musik, MAX_FREQ) 
    N_spectr = cut_Spectr.shape[1]

    TRIM_STRIP = 85
    N_SMAL = 4
    NOISE_EXEPTION_LEVEL = 0.21 * N_spectr # —> Подавление гула. При увеличении коэф-та снижается влияние 0.31
       
    arr_to_all_hstgrm = []    
    for ix in range(N_spectr): #==================================================================  
        arr_i_max = argrelextrema(cut_Spectr[:MAX_FREQ, ix], np.greater_equal, order = N_SMAL )[0]       
        arr_to_all_hstgrm.extend(arr_i_max)        
    #=============================================================================================
  
    hstgrm = get_hstgrm_1D(arr_to_all_hstgrm, MAX_FREQ, prnt=False ) 

    EXEPT_LIST = []    
    i = -1
    for one_fr in hstgrm: #ffffffffffffffffffffffffffffffff
        i += 1
        if one_fr > NOISE_EXEPTION_LEVEL  and  i > 5:
            EXEPT_LIST.append(i)
            # print(i, ') исключение без муз. --> ', one_fr)
    #ffffffffffffffffffffffffffffffffFFFFFFFFFFFFFFFFFFFFFF 
    
    ############################
    cut_Spectr[0:5][:] = 0
    for one_exept in EXEPT_LIST:
        cut_Spectr[one_exept - 2 :one_exept + 3][:] = 0
    ############################
    mean_A = np.mean(cut_Spectr, axis=0)  # Нашли среднюю амплитуду (энергию) всех частот в кажд. момент времени

    sort_mean_A = sorted(mean_A)
    all_quant = len(sort_mean_A)
    min_lev = sort_mean_A[int(all_quant*0.20)]
    max_lev = sort_mean_A[int(all_quant*0.90)]

    ############################    
    LEVEL_Z = (min_lev + NOT_SOUND_KOEF*(max_lev - min_lev) )
    # LEVEL_Z = min(mean_A) + NOT_SOUND_KOEF * (max(mean_A) - min(mean_A))
    ############################     
    
    # print('min_lev =', min_lev, 'max_lev =', max_lev)
    # print( 'Отсчеты исх. сигнала:', Signal_without_musik.shape,  '   Всего спектров =', mean_A.shape, all_quant )   
    
    # print('cut_Spectr.shape =', cut_Spectr.shape, 'LEVEL_Z =', LEVEL_Z)
    # histogr_Level = np.histogram(mean_A)
    # print('Всего значений =', sum(histogr_Level[0]) , ' Гистогр.:', histogr_Level[0], '   ', histogr_Level[1])    


    count_pause = 0
    count_sound = 0
    start_rec = -3
    stop_rec = -3
    n_seg = 0
    count_mus = 0
    max_N_pred = -1
    flag_musik = False
    x_wind = 0

    OY_freq = MAX_FREQ
   
    for ix in range( N_spectr ): #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        if mean_A[ix] < LEVEL_Z : # or max_N <=5 : #or max_N in EXEPT_LIST: # ПАУЗА
            count_pause += 1  # ПАУЗА
            count_sound = 0   # ПАУЗА
        else:  # ЗВУК
            count_pause = 0   # ЗВУК
            count_sound += 1  # ЗВУК  
            

        if count_pause > N_PAUSE and stop_rec < -1 and start_rec >= 0:        
            stop_rec = min( ix - N_PAUSE +8,  N_spectr -1 )        # 

        if count_sound > N_VOICE and start_rec < -1:        
            start_rec = max( ix -N_VOICE -8,  0)    


        # ########################################################################################
        # print ('', ix,')  sound =', count_sound, '  pause =', count_pause, '  start_rec = ', start_rec, 
        #     '  stop_rec = ', stop_rec, '   mean_A[ix]=', mean_A[ix].round(1), '   LEVEL_Z=', LEVEL_Z.round(2)) # , 
        #     # '   Min lev:', min(mean_wind).round(1), '   Max lev:', max(mean_wind).round(1)  )   ###---------------------------
        # ########################################################################################  

        if ix >= N_spectr - 1:           
            if start_rec >=0 :
                stop_rec = ix                
            else:
                break    
            
        if start_rec >= 0 and stop_rec > 0 and start_rec < stop_rec  : # записываем сегмент и его тип в массивы                                          
            if stop_rec - start_rec > LEN_PHRASE_TO_IGNORE:
                list_of_timing.append([ start_rec*512, stop_rec*512 ]) # отсчеты исходного сигнала  
                len_phrase += (start_rec - start_rec)
                N_PAUSE = MAX_N_PAUSE - K_PAUSE * len_phrase 
                N_PAUSE = N_PAUSE if N_PAUSE > MIN_N_PAUSE else MIN_N_PAUSE
            ############### 
            # print( '============ Сегмент', n_seg ,':  start_rec = ', start_rec, '   stop_rec =', stop_rec, ' =======  '  )# (Xnorm[:,start_rec:stop_rec]).shape )
            ###############
            n_seg += 1
            stop_rec = -3
            start_rec = -3
            count_pause = 0
            count_sound = 0
    #%%%%%%%%%%%%%%%%%%%%%%%% /for ix in range(N_spectr): #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
   
    return list_of_timing  # отсчеты исходного сигнала

"""## 3. Распознавание по списку интервалов

Необходимо для поиска триггером сущностей
"""

# Функция распознавания файла по частям с помощью Google Speech Recognition на основании переданного списка интервалов
def separec_Google_timed(path_wav,
                         list_of_timing,
                         rate = 22050):
  r = sR.Recognizer()
  textes = []
  starts = []
  durations = []
  #print('Всего фрагментов:',len(list_of_timing))
  for i in range(len(list_of_timing)):
    start = round(list_of_timing[i][0]/rate, 1)
    duration = round(list_of_timing[i][1]/rate-start, 1)
    with sR.AudioFile(path_wav) as source:
        audio_part = r.record(source, offset = start, duration = duration)      # Cчитываю часть аудиофайла
        try: text_part = r.recognize_google(audio_part, language='ru')
        except Exception as e:
          #print('Ошибка нарезки на ',start,'-й секунде с интервалом ',duration,' сек.', sep='') # Пока оставил, чтобы понимать как часто мы отдаем пустой кусок SpeechRecognition
          text_part = ''  # Если ошибка распознавания то обнуляю текст
        
        textes.append(text_part)
        starts.append(start)
        durations.append(duration)

  return textes, starts, durations

# Функция распознавания файла по частям с помощью Google Speech Recognition на основании переданного списка интервалов с визуализацией
def separec_Google_timed_visualized(path_wav,
                           list_of_timing,
                           rate = 22050):
  
  fullAudio = AudioSegment.from_wav(path_wav)
  wav_len_sec = librosa.get_duration(filename = path_wav)
  wav_len_aud = len(fullAudio)
  sr = wav_len_aud/wav_len_sec
  r = sR.Recognizer()

  print('\nВсего фрагментов:',len(list_of_timing))
  for i in range(len(list_of_timing)):
    start = round(list_of_timing[i][0]/rate, 1)
    stop = round(list_of_timing[i][1]/rate, 1)
    tmpAudio = fullAudio[start*sr:stop*sr]
    tmpAudio.export('tmp.wav', format="wav", )
    duration = round(stop-start, 1)
    print('Обработка фрагмента',i,'с началом на',start,'сек и длительностью',duration,'сек.')
    with sR.AudioFile('tmp.wav') as source:
        audio_part = r.record(source)      # Cчитываю часть аудиофайла
        try: text_part = r.recognize_google(audio_part, language='ru')  # Проверка на успешную работу распознавания
        except Exception as e:
          #print('Ошибка нарезки на ',start,'-й секунде с интервалом ',duration,' сек.', sep='') # Пока оставил, чтобы понимать как часто мы отдаем пустой кусок SpeechRecognition
          text_part = ''  # Если ошибка распознавания то обнуляю текс
        
        ipd.display(ipd.Audio('tmp.wav')) 
        print_lim(text_part)

# Функция распознавания файла по частям с помощью Google Speech Recognition на основании переданного списка интервалов
def separec_Yandex_timed(path_wav,
                         list_of_timing,
                         rate = 22050):
                         #visualization = False):

  session = Session.from_yandex_passport_oauth_token(oauth_token, catalog_id)
  framerate = '8000'  # Используется при сохранении файла и передаче на распознавание в Яндекс
  
  fullAudio = AudioSegment.from_wav(path_wav)
  wav_len_sec = librosa.get_duration(filename = path_wav)
  wav_len_aud = len(fullAudio)
  sr = wav_len_aud/wav_len_sec 
  textes = []
  starts = []
  durations = []
  #if visualization == True: print('Всего фрагментов:',len(list_of_timing))
  for i in range(len(list_of_timing)):
    start = round(list_of_timing[i][0]/rate, 1)
    stop = round(list_of_timing[i][1]/rate, 1)
    tmpAudio = fullAudio[start*sr:stop*sr]
    tmpAudio.export('tmp.wav', format="wav", parameters=['-ar', framerate]) # Сохраняю фрагмент во временный tmp.wav
    duration = round(stop-start, 1)
    if duration > 30: duration = 29.99   # Проверка, что фрагмент не получился больше. Если таких фрагментов будет много, то необходимо будет их делить!!!
    #if visualization == True: print('Обработка фрагмента',i,'с началом на',start,'сек и длительностью',duration,'сек.')
    with open('tmp.wav', 'rb') as f:
      data = f.read()
    
      # Создаем экземпляр класса с помощью `session` полученного ранее
      recognizeShortAudio = ShortAudioRecognition(session)

      # Передаем файл и его формат в метод `.recognize()`, 
      # который возвращает строку с текстом
      try: text_part = recognizeShortAudio.recognize(data, format='lpcm', sampleRateHertz=framerate)
      except Exception as e: text_part = ''  # Если ошибка распознавания то обнуляю текст
        
      textes.append(text_part)
      starts.append(start)
      durations.append(duration)
      #if visualization == True:
      #  ipd.display(ipd.Audio('tmp.wav')) 
      #  print_lim(text_part)

  return textes, starts, durations

# Функция распознавания файла по частям с помощью Google Speech Recognition на основании переданного списка интервалов
def separec_All_timed_visualized(path_wav,
                         list_of_timing,
                         rate = 22050):
  
  session = Session.from_yandex_passport_oauth_token(oauth_token, catalog_id)
  framerate = '8000'  # Используется при сохранении файла и передаче на распознавание в Яндекс

  r = sR.Recognizer()
  fullAudio = AudioSegment.from_wav(path_wav)
  wav_len_sec = librosa.get_duration(filename = path_wav)
  wav_len_aud = len(fullAudio)
  sr = wav_len_aud/wav_len_sec 
  print('Всего фрагментов:',len(list_of_timing))
  
  G_errors = 0
  Y_errors = 0
  G_zeros = 0
  Y_zeros = 0
  
  for i in range(len(list_of_timing)):
    start = round(list_of_timing[i][0]/rate, 1)
    stop = round(list_of_timing[i][1]/rate, 1)
    duration = round(stop-start, 1)
    print('\nОбработка фрагмента',i,'с началом на',start,'сек и длительностью',duration,'сек.')
    if duration > 29.99:
      duration = 29.99    # Проверка, что фрагмент не получился больше. Если таких фрагментов будет много, то необходимо будет их делить!!!
      stop = start + duration
      print('Из-за ограничений Yandex пришлось обрезать фрагмент до 29,9 сек.')
    tmpAudio = fullAudio[start*sr:stop*sr]
    tmpAudio.export('tmp.wav', format="wav", parameters=['-ar', framerate]) # Сохраняю фрагмент во временный tmp.wav

    with sR.AudioFile('tmp.wav') as source:
      audio_part = r.record(source)      # Cчитываю часть аудиофайла
      try: text_google = r.recognize_google(audio_part, language='ru')  # Проверка на успешную работу распознавания
      except Exception as e:
        text_google = ''  # Если ошибка распознавания то обнуляю текст
        G_errors += 1
      

    with open('tmp.wav', 'rb') as f:
      data = f.read()

      # Создаем экземпляр класса с помощью `session` полученного ранее
      recognizeShortAudio = ShortAudioRecognition(session)

      # Передаем файл и его формат в метод `.recognize()`, 
      # который возвращает строку с текстом
      try: text_yandex = recognizeShortAudio.recognize(data, format='lpcm', sampleRateHertz=framerate)
      except Exception as e:
        text_yandex = ''  # Если ошибка распознавания то обнуляю текст
        Y_errors += 1

    ipd.display(ipd.Audio('tmp.wav'))
    if text_google != '':
      print('Текст Google:') 
      print_lim(text_google)
    else: G_zeros += 1
    if text_yandex != '':
      print('Текст Yandex:') 
      print_lim(text_yandex)
    else: Y_zeros += 1

  if (G_zeros + Y_zeros) == 0: print('\nПустые результаты распознавания отсутствуют.')
  else: 
   if G_zeros != 0: print('\nGoogle вернул',G_zeros,'пустых текстов',end='') # По идее для Google это будет равно количеству ошибок
   if Y_zeros != 0: print('\nYandex вернул',Y_zeros,'пустых текстов',end='')

  
  if (G_errors + Y_errors) == 0: print('\nОшибки распознавания отсутствуют.')
  else: 
   if G_errors != 0: print('\nВсего ошибок при распознавании Google',G_errors,end='')
   if Y_errors != 0: print('\nВсего ошибок при распознавании Yandex',Y_errors,end='')

"""## 4. Cклейка списка фрагментов V2.1"""

# Функция для склейки списка непересекающихся строк (для подачи на классификацию)
def glue_uncrossed_text(textes):
  s = 0
  # Проверка не яывляется ли результатом распознавания один единственный пустой текст (бывает у Yandex)
  if len(textes) != 1:
    # Пропуск пустых текстов в начале
    while len(textes[s]) == 0 or s == len(textes):
      s += 1
  result = textes[s]
  # Объединение в одит текст всех не пустых
  if s < len(textes):
    for i in range(s,len(textes)):
      if len(textes[i]) != 0:
        result += ' ' + textes[i]
  return result.replace('  ',' ') # Замена двойных пробелов (иногда результат распознавания, почему-то начинается с пробела)

"""## 5. Классификация"""

# Загрузка модели Tinkoff c Google диска
gdown.download('https://drive.google.com/uc?id=1VF_ByGviod-klt8Q0pNYjsSV-hk8YQw2', None, quiet=True)

# Распаковка архива в папку Models
!unzip -qo model.zip -d Model

# Просмотр содержимого папки
!ls Model

# Распаковка модели
model = load_model('Model/model.h5')

# Распаковка токенайзера
with open('Model/tokenizer.pickle', 'rb') as f:
    tokenizer = pickle.load(f)

# Список классов моделей (порядок который у меня был при обучении)
CLASS_LIST = ['1_Order','2_Recall','3_Phone','4_Break','5_Prank','6_Auto']

# Подготовка и запуск предсказания объединенные в функцию
def give_predict(text):
  # Преобразую строку в последовательность индексов согласно частотному словарю с помощью токенайзера, формирую в виде разреженной матрицы (bag of words) и отправляю в НС
  y_pred = model.predict(np.array(tokenizer.sequences_to_matrix(tokenizer.texts_to_sequences([text]))), verbose='0')

  return CLASS_LIST[np.argmax(y_pred)], round(np.max(y_pred)*100,2)

"""## 6. Обработка текстов для выделения триггеров сущностей"""

# Словарь сущностей с соответствующими триггерами
ENTITIES_DICT = {'ФИО': [' имя ', ' фамилия ', ' отчество ', ' зовут ', ' обращаться '],
                 'ТЕЛЕФОН': [' телефон', ' мобильн', ' номер '],
                 'АДРЕС': [' обращаетесь ', ' обращайтесь ', ' индекс ', ' адрес ', ' край ', ' район ', ' област', ' город', ' село ', ' деревня ', ' посёлок ', ' улица ', ' переулок ', ' проспект ', 'аллея ', ' проезд ', ' дом ', ' строение ', ' квартира '],
                 'ТОВАР': [' товар ', ' продукт ', ' заказ ', ' заказать ', ' купить ', ' отправ', ' вышлите ' , ' присла'],
                 'ЦЕНА': [' цена ', ' стоимость ', ' стоит ', ' скидка ', ' руб ', ' руб. ', ' р ' ' рублей ', ' рубля ', ' всего ', ' сумма ']
                }

# Оработка словаря

# Список триггеров сущностей
ENTITIES_TRIGGERS = []
# Список напменований сущностей для поиска
ENTITIES_NAMES = []

ENTITIES_UNIC = []

for entity in ENTITIES_DICT.keys():
  ENTITIES_UNIC.append(entity)
  for trigger in ENTITIES_DICT[entity]:
    ENTITIES_TRIGGERS.append(trigger)
    ENTITIES_NAMES.append(entity)

# Функция поиска триггера в тексте, которая возвращает сам триггер и соответствующую ему сущность V2
def entrigger_search(text):
  founded_triggers = []
  founded_entities = []
  for ent_tr in ENTITIES_TRIGGERS:
    if ent_tr in (' ' + text.lower() + ' '):  # Добавлены пробелы, чтобы корректро обрабатывать триггеры в конце и начале фраз
      founded_triggers.append(ent_tr)
      ent_nm = ENTITIES_NAMES[ENTITIES_TRIGGERS.index(ent_tr)]
      if ent_nm not in founded_entities:
        founded_entities.append(ent_nm)
  return founded_triggers, founded_entities

# Функция поиска триггеров и соответствующих им сущностей во всех текстах
def entrigger_scanner(textes, starts, durations):
  scanned_triggers = []
  scanned_entities = []
  tr_starts = []
  tr_durations = []
  for i in range(len(textes)):
    founded_triggers, founded_entities = entrigger_search(textes[i])
    if len(founded_entities) != 0:
      scanned_triggers.append(founded_triggers)
      scanned_entities.append(founded_entities)
      tr_starts.append(starts[i])
      tr_durations.append(durations[i])
  return scanned_triggers, scanned_entities, tr_starts, tr_durations

# Функция выделения всех уникальных сущностей и соответствующих им временных промежутков
def entities_locator(scanned_entities, tr_starts, tr_durations):
  sortr_starts = []
  sortr_durations = []
  for ent in ENTITIES_UNIC:
    starts = []
    durations = []
    for i in range(len(scanned_entities)):
      if ent in scanned_entities[i]:
        starts.append(tr_starts[i])
        durations.append(tr_durations[i])
    sortr_starts.append(starts)
    sortr_durations.append(durations)
  return sortr_starts, sortr_durations

# Визуализация полученных данных
def visualize_fragments(path_wav, sortr_starts, sortr_durations, textes, starts):
  audio, rate = sf.read(path_wav)
  # Прохожу по списку уникальных сущностей
  for i in range(len(ENTITIES_UNIC)):
    print('\n'+'-.'*MAX_LINE+'-\n')
    if len(sortr_starts[i]) == 0:
      message = '<b>Сущность "' + ENTITIES_UNIC[i] + '" не была обнаружена с помощью триггеров.</b><br>'
      display(Markdown(message))
    else:
      message = '<b>Сущность "' + ENTITIES_UNIC[i] + '" была обнаружена с помощью триггеров в следующих временных интервалах:</b>'
      display(Markdown(message))
      # Прохожу по интервалам для каждой сущности
      for n in range(len(sortr_starts[i])):
        # Подготовлю фрагмент текста для NER
        text_fragment = textes[starts.index(sortr_starts[i][n])]
        # Определю интервалы фрагмента для прослушивания
        start = sortr_starts[i][n]
        duration = sortr_durations[i][n]
        # Нарезаю фрагмент
        fragment = audio[int(start*rate):int((start+duration)*rate)]
        # Сохраняю фрагмент в файл
        sf.write(f'temp_{start}.wav', fragment, rate)
        # Выведу время фрагмента и найденые сущности
        print(f'\nФрагмент с {start}-й сек, длительностью {duration} сек.')
        # Создам кнопку для прослушивания
        ipd.display(ipd.Audio(f'temp_{start}.wav'))
        # Выведу время фрагмента и найденые сущности
        print('Текст в котором был обнаружен триггер сущности:')
        print_lim(text_fragment)

"""## 7. Подготовка текстов для NER"""

# Переменные которые позволяют регулировать получающиеся для NER тексты

# Список границ интервалов для поиска сущностей
EN_FRAG_LIM = [[5,15], [0,15], [5,15], [5,20], [5,15]]
# Длина фрагментов аудиофайлафайла, подаваемых на распознавание (в секундах)
WIN_SIZE = 5
# Перекрытие между фрагментами айдиофайла подаваемыми на распознавание (в секундах)
WIN_HOP = 2
# Размер текстового окна для склейки
TEXT_WIN = 3

# Оптимизация найденных координат триггеров для получения интервалов локализации сущностей
def entities_fragments(sortr_starts, sortr_durations):
  en_starts = []
  en_ends = []
  for i in range(len(sortr_starts)):
    if len(sortr_starts[i]) > 1: 
      start = [sortr_starts[i][0]-EN_FRAG_LIM[i][0]]
      if start[0] < 0: start[0] = 0
      end = [start[-1] + sortr_durations[i][0] + EN_FRAG_LIM[i][1]]
      for n in range(1,len(sortr_starts[i])):
        if sortr_starts[i][n] < end[-1]:
          end[-1] = sortr_starts[i][n] + sortr_durations[i][n] + EN_FRAG_LIM[i][1]
        else:
          start.append(sortr_starts[i][n]-EN_FRAG_LIM[i][0])
          end.append(start[-1]+sortr_durations[i][n] + EN_FRAG_LIM[i][1])

      en_starts.append([round(start[j],1) for j in range(len(start))])
      en_ends.append([round(end[j],1) for j in range(len(end))])
    elif len(sortr_starts[i]) == 1:
      en_starts.append([sortr_starts[i][0]-EN_FRAG_LIM[i][0]])
      en_ends.append([sortr_starts[i][0] + sortr_durations[i][0] + EN_FRAG_LIM[i][1]])  
    else:
      en_starts.append([])
      en_ends.append([])  
  return en_starts, en_ends

# Функция распознавания файла по частям с помощью Google Speech Recognition
def separec_Google(path_wav,
                   starts,
                   ends,
                   win_size = WIN_SIZE,
                   win_hop = WIN_HOP):
  # Проверки на выход за границу файла
  if starts[0] < 0: starts[0] = 0
  wav_length = librosa.get_duration(filename = path_wav)
  if ends[-1] > wav_length: ends[-1] = wav_length

  r = sR.Recognizer()
  all_textes = []
  
  for i in range(len(starts)):
    start = starts[i]
    textes = []
    while start + win_size < ends[i]:
      with sR.AudioFile(path_wav) as source:
        if start + 2*win_size - win_hop > ends[i]:    # Если на следующем шаге не хватит файла на еще одно окно, то сразу распознаем остаток файла
          audio_part = r.record(source, offset = start, duration = ends[i]-start)      # Cчитываю остаток аудиофайла
        else:
          audio_part = r.record(source, offset = start, duration = win_size)      # Cчитываю часть аудиофайла
        try: text_part = r.recognize_google(audio_part, language='ru')
        except: text_part = ''
      textes.append(text_part)
      start += win_size - win_hop

    all_textes.append(textes)
    
  return all_textes

# Функция распознавания файла по частям с помощью Google Speech Recognition
def separec_Yandex(path_wav,
                   starts,
                   ends,
                   win_size = WIN_SIZE,
                   win_hop = WIN_HOP):

  if win_size > 30: win_size = 29.99    # Подстраховка для распознавания Яндекс
  
  session = Session.from_yandex_passport_oauth_token(oauth_token, catalog_id)
  framerate = '8000'      # Задал 8000 инача распознавание яндек выдает ошибки

  fullAudio = AudioSegment.from_wav(path_wav)
  wav_len_sec = librosa.get_duration(filename = path_wav)
  wav_len_aud = len(fullAudio)
  sr = wav_len_aud/wav_len_sec

  # Проверки на выход за границу файла    
  if starts[0] < 0: starts[0] = 0
  if ends[-1] > wav_len_sec: ends[-1] = wav_len_sec

  all_textes = []

  for i in range(len(starts)):
    start = starts[i]
    textes = []
    while start < ends[i]:
      if start + win_size < ends[i]: duration = win_size
      else: duration = ends[i] - start
      t1 = start*sr
      t2 = (start+duration)*sr

      tmpAudio = fullAudio[t1:t2]
      tmpAudio.export('tmp.wav', format="wav", parameters=['-ar', framerate]) # Сохраняю фрагмент во временный tmp.wav

      with open('tmp.wav', 'rb') as f:
        data = f.read()

        # Создаем экземпляр класса с помощью `session` полученного ранее
        recognizeShortAudio = ShortAudioRecognition(session)

        # Передаем файл и его формат в метод `.recognize()`, 
        # который возвращает строку с текстом
        try: text = recognizeShortAudio.recognize(data, format='lpcm', sampleRateHertz=framerate)
        except Exception as e:
          text = ''  # Если ошибка распознавания то обнуляю текст
          #print('Ошибка нарезки на ',start,'-й секунде с интервалом ',duration,' сек.', sep='')
          #print(e)
    
        textes.append(text)
        start += win_size - win_hop

    all_textes.append(textes)
    
  return all_textes

# Функция склейки 2-х строк по совпадающему фрагменту
def glue_pair (text1,
               text2,
               text_win):
  len1 = len(text1)
  len2 = len(text2)
  if len2 <= 1: return text1
  max_len = min(len1,len2)
  if text_win > 0:
    for i in range(text_win,max_len+1):  # Цикл задающий смещение текстов друг относительно друга
      for j in range(0,i-text_win+1): # Цикл смещения окна внутри области перекрытия текстов
        if text1[len1-i+j:len1-i+text_win+j] == text2[j:j+text_win]: return text1[:len1-i+j] + text2[j:]
  
  return text1 + ' ' + text2[0:]  #+ text2[0].lower() + text2[1:]  # Надо проверить будут ли лишние большие буквы

# Функция для склейки списка строк с заданным размером текстового окна
def glue_text(textes, text_win = TEXT_WIN):
  result = textes[0]
  for i in range(1,len(textes)):
    result = glue_pair(result, textes[i], text_win)
  return result

# Распознание текстов из аудио по полученным интервалам локализации сущностей
def textes_for_NER(path_wav, en_starts, en_ends, rec_tool = ''):
  NER_textes = []
  for i in range(len(en_starts)):
    glued_textes = []
    if len(en_starts[i]) != 0:
      if rec_tool == 'yandex':
        all_textes = separec_Google(path_wav, en_starts[i], en_ends[i])
      else:
        all_textes = separec_Yandex(path_wav, en_starts[i], en_ends[i])
      for n in range(len(all_textes)):
        glued_textes.append(glue_text(all_textes[n]))
    NER_textes.append(glued_textes)
  return NER_textes

"""## 8. Поиск имён операторов, ФИО клиентов с помощью NER V3

Функция для запуска - get_person_from_ner(lst_text,text)

Тест-пример вызова функции - в конце файла

При реализации кода используются:

словари имён, фамилий, отчеств , иностранных имён в формате файлов .xlsx
NER Natasha в настройке для поиска сущностей PER
NER PullEnti в настройке для поиска сущностей PER
"""

# NER Natasha - функция распознавания

def get_natasha_ent_per(text):
  # возвращает результата разбора в формате структуры:
  result_structure = {
              'ENT': [],       #все найденные сущности PER, 
              'POSITION': []      #координаты сущности в тексте в формате списка [start,stop]
  }

  doc = Doc(text)

  doc.segment(segmenter)
  doc.parse_syntax(syntax_parser)
  doc.tag_morph(morph_tagger)
  doc.tag_ner(ner_tagger)
  # doc.ner.print()

  # заполнение структуры
  for dd in doc.ner.spans:
     if dd.type == 'PER':
      ss = doc.ner.text[dd.start:dd.stop]
      # print(ss, dd.start, dd.stop)
      result_structure['ENT'].append(ss)
      result_structure['POSITION'].append([dd.start, dd.stop])    

  return(result_structure)

# NER PullEnti - функция распознавания

def get_pullenti_ent_per(text):
  # возвращает результата разбора в формате структуры:
  result_structure = {
              'ENT': [],            #все найденные сущности PER, 
              'POSITION': [],       #координаты сущности в тексте в формате списка [start,stop]
              'FIO': [],            #все найденные сущности PER, распознанные как FIO, 
              'LASTNAME': [],       #все найденные сущности PER, распознанные как LASTNAME 
              'MIDDLENAME': [],     #все найденные сущности PER, распознанные как MIDDLENAME 
              'FIRSTNAME': []       #все найденные сущности PER, распознанные как FIRSTNAME              
  }

  result = processor(text)

  lst_per = []
  lst_span = []
  lst_fio = []
  slastname = ''
  smiddlename = ''
  sfirstname = ''

  for ent in result.raw.entities:
    if ent.type_name == 'PERSON':
      s = ''
      sex = ''
      sfio = ''

      for sl in ent.slots:
          # print('sl___',sl)
          # print('________________', sl.type_name, sl.value)
        s_ent = sl.value
        lst_per.append(s_ent)
        sex += ", " + str(sl.type_name) + " ='" + str(sl.value) + "'"
        if str(sl.type_name) in ['LASTNAME', 'FIRSTNAME', 'MIDDLENAME']:
          sfio += str(sl.value) + ' '
        if str(sl.type_name) in ['LASTNAME']:
          slastname = str(sl.value)
        if str(sl.type_name) in ['MIDDLENAME']:
          smiddlename = str(sl.value)
        if str(sl.type_name) in ['FIRSTNAME']:
          sfirstname = str(sl.value)

      for tt in ent.occurrence:
        # print('tt-',tt.begin_char, tt.end_char, tt.end_char-tt.begin_char)
        s = "Span(start=" + str(tt.begin_char) + ", stop=" + str(tt.end_char) +\
         ", type='" + str(ent.type_name) + "'"
        s = s + sex
        s = s + ")"
        # print(s)
      
      lst_span.append(s)
      lst_fio.append(' '.join(sfio.split()))
  # print('___________________________________')
  # print(lst_per)   
  # print(lst_span)
  # print(lst_fio)
 
  result_structure['ENT'] = lst_per
  result_structure['POSITION'] = lst_span
  result_structure['FIO'] = lst_fio
  result_structure['LASTNAME'] = slastname
  result_structure['MIDDLENAME'] = smiddlename
  result_structure['FIRSTNAME'] = sfirstname

  return(result_structure)

# функция загрузка справочников фио
def load_fio_from_catalogs(sheet_name_id,sheet_surname_id,\
                           sheet_midname_id,sheet_foreigname_id):
  # на входе - id расшаренных на GoogleDrive файлов
  # на выходе - списки имён,фамилий,отчеств,всех фио
  
  # Имена
  url = 'https://docs.google.com/spreadsheets/d/{0}/gviz/tq?tqx=out:csv'.format(sheet_name_id)
  dfs = pd.read_csv(url)
  lst_names = dfs.Name
  # print(len(lst_names),lst_names[0:5])

  # # Фамилии
  url = 'https://docs.google.com/spreadsheets/d/{0}/gviz/tq?tqx=out:csv'.format(sheet_surname_id)
  dfs = pd.read_csv(url)
  lst_surnames = dfs.Surname
  # print(len(lst_surnames),lst_surnames[0:5])

  # # Отчества
  url = 'https://docs.google.com/spreadsheets/d/{0}/gviz/tq?tqx=out:csv'.format(sheet_midname_id)
  dfs = pd.read_csv(url)
  lst = dfs.MiddleName
  lst_midnames = []
  ss = ''
  for i in lst:
    ss = ss + i.replace(' ','') + ','
  lst_midnames = ss.split(',')
  # print(len(lst_midnames),lst_midnames[0:5])

  # иностранные имена
  url = 'https://docs.google.com/spreadsheets/d/{0}/gviz/tq?tqx=out:csv'.format(sheet_foreigname_id)
  dfs = pd.read_csv(url)
  lst_foreignnames = dfs.Name
  # print(len(lst_foreignnames),lst_foreignnames[0:5])

  #сформируем общий справочник для русских ФИО
  lst_fullnames = [*lst_surnames, *lst_names, *lst_midnames]
  lst_fullnames.sort()
  # print(len(lst_fullnames))

  # переведём все слова справочника в нижний регистр
  lst_fullnames_lower = [text.lower() for text in lst_fullnames]

  # print(lst_fullnames[0:100])
  # print(lst_fullnames_lower[0:100])
  # print('Справочники фио загружены')

  return(lst_names,lst_surnames,lst_midnames,lst_foreignnames,lst_fullnames,lst_fullnames_lower)

def get_trigers_for_per():
  # Тригеры - граничные слова для ИМЕНИ оператора
  lst_operator_words_before = [
      'вас слушаю здравствуйте',
      'ваш звонок',
      'ваш звонок очень важен для нас',
      'дождитесь ответа оператора',
      'здравствуйте вы позвонили',
      'здравствуйте меня зовут',
      'зовут меня',
      'кампания',
      'компания',
      'магазин',
      'меня зовут',
      'оператор',
      'оператора',
      'ответа оператора',
      'очень важен',
      'слушаю вас',
      'существующий заказ',
      'телемагазин',
      'я вас слушаю']
  # приведение списка к нижнему регистру+сортировка
  # lst_operator_words_before = [x.lower() for x in lst_operator_words_before]
  # lst_operator_words_before.sort()

  lst_operator_words_after = [
      'вам помочь',
      'вас как зовут',
      'вас слушаю',
      'город',
      'добрый день',
      'доставка',
      'заказать',
      'заявка',
      'заявку',
      'здравствуйте девушка',
      'индекс',
      'конечно',
      'могу',
      'могу вам помочь',
      'могу помочь',
      'можно',
      'область',
      'подскажите',
      'помочь',
      'реклама',
      'рекламу',
      'скажите пожалуйста',
      'слушаю',
      'слушаю вас',
      'слышу вас',
      'хотели',
      'чем могу',
      'чем могу вам помочь',
      'чем могу помочь',
      'что вас заинтересовало',
      'я вас слушаю']

  # Тригеры - граничные слова для ФАМИЛИИ клиента
  lst_fio_words_before = [
      'адрес',
      'ваши данные',
      'данные',
      'здравствуйте',
      'имя',
      'как ваша',
      'могу к вам обращаться',
      'назовите',
      'назовите фамилию',
      'обращаться',
      'отчество',
      'почтовый',
      'представьтесь',
      'фамилию',
      'фамилию имя отчество',
      'фамилия',
      'фамилия имя отчество']
  lst_fio_words_after = [
      'адрес', 
      'верно', 
      'отчество', 
      'почтовый', 
      'телефон']

  return(lst_operator_words_before,lst_operator_words_after,lst_fio_words_before,\
         lst_fio_words_after)

# функция простого поиска в словарях 
def find_in_dictionaries(text,lst_names, lst_midnames, lst_surnames):
  # возвращает результата разбора в формате структуры:
  result_structure = {
              'ENT': [],          #все найденные фио(имена, фамилии, отчества), 
              'ENTTYPE': [],      #тип: 1-имя/2-отчество/3-фамилия, 
              'POSITION': []      #координаты сущности в тексте в формате списка [start,stop]
  }
  lst_words = text.split()
  lst_words_lower = text.lower().split()

  # список словарей
  lst_dict = [lst_names, lst_midnames, lst_surnames]
  lst_types = ['1-имя', '2-отчество', '3-фамилия']
  i = 0
  for lst in lst_dict:
    lst_lower = [text.lower() for text in lst]
    ent_list = list(Counter(lst) & Counter(lst_words)) + list(Counter(lst_lower) & Counter(lst_words))

    for ent in ent_list:
      ent_name = ent
      ent_position = text.find(' ' + ent_name + ' ')
      result_structure['ENT'].append(ent_name)
      result_structure['ENTTYPE'].append(lst_types[i])
      result_structure['POSITION'].append([ent_position,ent_position+len(ent_name)])

    i +=1

  # оптимизация найденных сущностей, если надо...
  
  return(result_structure)

# функция простого поиска в словарях MIDNAME
def find_in_dictionaries_midname(text,lst_midnames):
  # возвращает результата разбора в формате структуры:
  result_structure = {
              'MIDNAME': [],      #все найденные слова
              'POSITION': []      #координаты сущности в тексте в формате списка [start,stop]

  }
  lst_words = text.split()
  lst_words_lower = text.lower().split()

  # список словарей
  lst_dict = [lst_midnames]
  i = 0
  for lst in lst_dict:
    lst_lower = [text.lower() for text in lst]
    ent_list = list(Counter(lst) & Counter(lst_words)) + list(Counter(lst_lower) & Counter(lst_words))

    for ent in ent_list:
      ent_name = ent
      ent_position = text.find(' ' + ent_name + ' ')
      result_structure['MIDNAME'].append(ent_name)
      result_structure['POSITION'].append([ent_position,ent_position+len(ent_name)])
    i +=1

  return(result_structure)

# функция простого поиска в словарях SURNAME
def find_in_dictionaries_surname(text,lst_surnames):
  # возвращает результата разбора в формате структуры:
  result_structure = {
              'SURNAME': [],      #все найденные слова
              'POSITION': []      #координаты сущности в тексте в формате списка [start,stop]

  }
  lst_words = text.split()
  lst_words_lower = text.lower().split()

  # список словарей
  lst_dict = [lst_surnames]
  i = 0
  for lst in lst_dict:
    lst_lower = [text.lower() for text in lst]
    ent_list = list(Counter(lst) & Counter(lst_words)) + list(Counter(lst_lower) & Counter(lst_words))

    for ent in ent_list:
      ent_name = ent
      ent_position = text.find(' ' + ent_name + ' ')
      result_structure['SURNAME'].append(ent_name)
      result_structure['POSITION'].append([ent_position,ent_position+len(ent_name)])
    i +=1

  return(result_structure)

# функция простого поиска в словарях NAME
def find_in_dictionaries_name(text,lst_names):
  # возвращает результата разбора в формате структуры:
  result_structure = {
              'NAME': [],      #все найденные слова
              'POSITION': []      #координаты сущности в тексте в формате списка [start,stop]

  }
  lst_words = text.split()
  lst_words_lower = text.lower().split()

  # список словарей
  lst_dict = [lst_names]
  i = 0
  for lst in lst_dict:
    lst_lower = [text.lower() for text in lst]
    ent_list = list(Counter(lst) & Counter(lst_words)) + list(Counter(lst_lower) & Counter(lst_words))

    for ent in ent_list:
      ent_name = ent
      ent_position = text.find(' ' + ent_name + ' ')
      result_structure['NAME'].append(ent_name)
      result_structure['POSITION'].append([ent_position,ent_position+len(ent_name)])
    i +=1

  return(result_structure)

def clear_per_ent(ts_ner,lst_fullnames):
# функция ищет в списке строк слова, входящие в словарь ФИО
# на входе: список найденных сущностей типа PERSONAL/PER
# на выходе: 2 списка - очищенные строки слов(ts_res), "отбракованные строки слов"(ts_delta)
# количество элементов списков одинаковое
  ts_ent = ts_ner['ENT']
  # print('Сущности:', ts_ent)

  ts_res = {'ENT': [],'POSITION': []}
  ts_delta = {'ENT': [],'POSITION': []}
  i = 0
  for ts in ts_ent:
    #ищем слова в словаре фио, оставляем только найденные, 
    # остальные помещаем в другой список для последующего анализа
    ts_list = ts.split()
    ss_new = list(Counter(lst_fullnames) & Counter(ts_list))
    ss_delta = list(set(ts_list).difference(ss_new))
    # print(ss_new)
    ss = ' '.join(ss_new)
    ss_d = ' '.join(ss_delta)
    pos = ts_ner['POSITION'][i]

    if ss>'':
      ts_res['ENT'].append(ss)
      ts_res['POSITION'].append(pos)    
    ts_delta['ENT'].append(ss_d)
    ts_delta['POSITION'].append(pos)  

    i += 1
  
  # print('Очищенные сущности:', ts_res)
  # print('Отбракованные части сущностей:', ts_delta)
  return(ts_res,ts_delta)

def get_operator_name(text,ts_res,lst_operator_words_before,lst_operator_words_after,
                      result_dict_name,lst_names):
  # проверка, что найденное первое имя есть имя оператора
  ts_operator = []
  ts_operator_position = []
  operator_name = ''
  operator_position = 0
  textpart = []

  if len(ts_res['ENT'])>0:
    # print(ts_res['ENT'])
    for i in range(len(ts_res['ENT'])):
      s = ts_res['ENT'][i].capitalize()
      isfind = find_in_dictionaries_name(s,lst_names)
      if len(isfind['NAME'])>0:
        operator_name = ts_res['ENT'][i]
        operator_position = ts_res['POSITION'][i][0]
        check_res = 1
        break
    # print('Предположительное имя оператора и его позиция в тексте: ', operator_name, operator_position)

    check_res, operator_name_new, operator_position_new, textpart = check_operator_name(text, 
                operator_name, operator_position, lst_operator_words_before,lst_operator_words_after, lst_names)
    # print('check_res', check_res, operator_name_new, operator_position_new, textpart)

    if check_res<1:
      operator_name = ''
      operator_position = -1
      #если поиск по тригерам дал новое имя оператора
    if check_res==2:
      if len(operator_name_new)>0:
        operator_name = operator_name_new
        operator_position = operator_position_new

  operator_position_list = [operator_position,operator_position+len(operator_name)]

  return([operator_name, operator_position_list, textpart])

def get_client_fio(text, ts_res, result_dict_name,result_dict_midname, result_dict_surname, 
                   result_pullenti,lst_fio_words_before,lst_fio_words_after,
                   lst_names, lst_midnames, lst_surnames):
# поиск Фамилии, отчества, имени
  name = ''
  fullfio = ''
  family = '' #
  family_position = []
  textpart = []
  surname = ''
  surname_first = ''
  surname_first_position = []
  midname = ''
  firstname = ''

  # через тригеры определяем диапазон нахождения фио
  # анализируем первые NS символов текста на наличие тригеров
  ns = len(text) 

  lst_words = text[0:ns].split()
  # print(f'Слова в отрезке {ns} символов:', lst_words) 

  tstart = 0 #граница слева
  tstop = ns
  lst_before = []
  text_part = text[tstart:tstop]
  prizn_last = lst_fio_words_before[0]
  for prizn in lst_fio_words_before:
    prizn_pozition = text_part.lower().find(prizn)
    if prizn_pozition>=0 and tstart>=prizn_pozition:
      tstart = prizn_pozition
      prizn_first = prizn
      lst_before.append(prizn)

  tstop = ns  #граница справа
  text_part = text[tstart:tstop]
  lst_after = []
  text_part = text.lower()# [tstart:tstop]
  prizn_last = lst_fio_words_after[0]
  for prizn in lst_fio_words_after:
    prizn_pozition = text_part.lower().find(prizn)
    if prizn_pozition>=0 and tstop>=prizn_pozition:
      tstop = prizn_pozition
      prizn_last = prizn
      lst_after.append(prizn)
    
    tstop = tstop + tstart + 1  #граница справа

  # print('результат поиска тригеров:', lst_before, lst_after, tstart, tstop)

  # #возьмём за основу фамилию от Natasha
  if len(ts_res['ENT'])>0:
    for i in range(len(ts_res['ENT'])):
      s = ts_res['ENT'][i].capitalize()
      isfind = find_in_dictionaries_surname(s,lst_surnames)
      if len(isfind['SURNAME'])>0:
        family = ts_res['ENT'][i]
        family_position = [ts_res['POSITION'][i][0],ts_res['POSITION'][i][0] + len(family)]
        tstart = family_position[0]
        tstop = tstart + len(family)
        textpart = [tstart, tstop, text[tstart:tstop+10]+'...']
        break
  # чистка family
  if len(family.split())>1:
    family_res = list(Counter(result_dict_surname['SURNAME']) & Counter(family.split()))
    if len(family_res)>0:
      family = family_res[0]

  # 1.начинаем с поиска отчества и фамилии рядом
  # ищем отчество, найденное в словаре, попадающее в зону тригеров
  if len(result_dict_midname['MIDNAME'])>0:
    midname = result_dict_midname['MIDNAME'][0]
    midname_pos_start = result_dict_midname['POSITION'][0][0]
    midname_pos_stop = result_dict_midname['POSITION'][0][1]
    # print('MIDNAME',midname, midname_pos_start, midname_pos_stop )
    # ищем фамилию, смотрим есть ли фамилия левее-правее отчества 
    for ii in range(len(result_dict_surname['SURNAME'])):
      surname = result_dict_surname['SURNAME'][ii]
      surname_pos_start = result_dict_surname['POSITION'][ii][0]
      surname_pos_stop = result_dict_surname['POSITION'][ii][1]
      surname_first = surname
      surname_first_position = [surname_pos_start, surname_pos_stop]
      # print('SURNAME',surname, surname_pos_start, surname_pos_stop )
      #если по порядку следования идёт имя-отчество-фамилия
      if abs(midname_pos_stop-surname_pos_start)<2:
        if not surname in ''.join(lst_names):
          family = surname.title()
          family_position = [surname_pos_start, surname_pos_stop]
          tstart = midname_pos_start - 20
          tstop = surname_pos_stop
      #если по порядку следования идёт фамилия-имя-отчество
      if abs(surname_pos_stop-midname_pos_start)<20: # проверка на близкое расположение
        #если найденная фамилия встречается в списке имён, то не признаём её фамилией
        if not surname in ''.join(lst_names):
          family = surname.title()  
          family_position = [surname_pos_start, surname_pos_stop]
          tstart = surname_pos_start + 1
          tstop = midname_pos_stop + 1

    if len(family_position)>0:
      if abs(midname_pos_start-family_position[0])>40:
        midname = ''

    # 2.ищем имя -  где-то рядом с отчеством и фамилией есть имя
    if family>'' and firstname=='':
      lst_textpart = text[surname_pos_start-30:surname_pos_start+30].split()
      lst_name = list(Counter(result_dict_name['NAME']) & Counter(lst_textpart))
      if len(lst_name)>0:
        if lst_name[0] in ''.join(lst_names):
          firstname = lst_name[0]
    
    # 3.ищем имя - если есть отчество , но имя определить не удалось, то пробуем поискать имя
    # по сущностям
    if firstname=='' and midname>'':
      for i in range(len(ts_res['ENT'])):
        enttext = str(ts_res['ENT'][i])
        iprizn = enttext.find(midname)
        t1 = int(ts_res['POSITION'][i][0])
        t2 = int(ts_res['POSITION'][i][1])
        if iprizn>=0:
          ent_words = enttext.split()
          ss_new = list(Counter(lst_names) & Counter(ent_words))
          names = list(set(ss_new).difference([midname]))
          if len(names)>0:
            firstname = names[0]
            tstart = t1
            tstop = t2
            break

    textpart = [tstart, tstop, text[tstart:tstop+10]+'...']

  # контрольный выстрел - чистка результата
  if len(family.split())>1:
    family_res = list(Counter(result_dict_surname['SURNAME']) & Counter(family.split()))
    if len(family_res)>0:
      family = family_res[0]

  #сборка ФИО
  family = family.title()
  fullfio = family
  if family>'':
    fullfio = (family.title() + ' ' + firstname.title() + ' ' + midname.title())
  else:
    # если фамилию так и не нашли, берём первую из NER['SURNAME']
    family = surname_first
    family_position = surname_first_position
    fullfio = (family.title() + ' ' + firstname.title() + ' ' + midname.title())
  
  # print('YYY', family, family_position, textpart, fullfio)

  return([family, family_position, textpart, fullfio])

def check_operator_name(text, operator_name, operator_position,lst_operator_words_before,
                        lst_operator_words_after,lst_names):
  # определение имени оператора
  # на входе - текст, начальная позиция сущности
  # на выходе - признак подтверждения, что найдено имя оператора
  # (если >0, то подтверждаем), имя и позиция нового оператора, если такого нашли

  res = 0

  # анализируем первые NS символов текста на наличие слов - тригеров
  ns = len(text) 
  lst_words = text[0:ns].split()
  lst_words = [x.lower() for x in lst_words]
  lst_words.sort()

  tstart = 0 #граница слева
  tstop = ns
  lst_before = []
  text_part = text[tstart:tstop]
  prizn_first = lst_operator_words_before[0]
  for prizn in lst_operator_words_before:
    prizn_pozition = text_part.lower().find(prizn)
    if prizn_pozition>=0 and tstart>=prizn_pozition:
      tstart = prizn_pozition
      prizn_first = prizn
      lst_before.append(prizn)

  # print(1,lst_before,tstart,prizn_first)
  if tstart==-1: tstart = 0

  tstop = ns  #граница справа
  text_part = text[tstart:tstop]
  lst_after = []
  text_part = text.lower()# [tstart:tstop]
  prizn_last = lst_operator_words_after[0]
  for prizn in lst_operator_words_after:
    prizn_pozition = text_part.lower().find(prizn)
    if prizn_pozition>=0 and tstop>=prizn_pozition:
      tstop = prizn_pozition
      prizn_last = prizn
      lst_after.append(prizn)
    
  tstop = tstop + tstart + 1  #граница справа
  # print(2,lst_after, tstart,tstop, operator_position,prizn_last)
  # print(text[tstart:tstop])
  
  if operator_position>=tstart and operator_position<=tstop:
    res = 1 

  #ищем имя оператора в словаре имён 
  # на отрезке между наводящими словами (триггерами)
  operator_name_new = ''
  operator_position_new= 0
  text_part = text[tstart:tstop]
  ts_list = text[tstart:tstop].split()
  ss_new = list(Counter(lst_names) & Counter(ts_list))
  if len(ss_new)>0:
    operator_name_new = ss_new[0]
    operator_position_new = text[tstart:tstop].find(operator_name_new) + tstart
    res = 2
  else:
    ss_delta = set(ss_new) - set([operator_name]) #отсекаем уже найденное имя
    ss_new
    if len(ss_delta)>0:
      ss = ' '.join(ss_delta)
      operator_name_new = ss
      operator_position_new = text[0:ns].find(ss[0])
      res = 2

  textpart = text[tstart:tstop+10]+'...'
  # print('XXX',res, operator_name_new, operator_position_new, tstart, tstop, textpart)

  return(res, operator_name_new, operator_position_new, [tstart, tstop, textpart])

# Функция печати текста с подсветкой сущности ФИО
def printmd(text,result):
  stext = text
  # если имя нашли, выделяем
  if result['names'][0]>'':
    s = result['names'][0]
    stext = stext.replace(s,' <b><font color=''blue''>' + s + '</font></b> ')
    stext = stext.replace(s.lower(),' <b><font color=''blue''>' + s.lower()+ '</font></b> ')
 #else:
    # выделим области
 #   if len(result['names'][3])>0 and result['names'][3][2]>'':
 #     s = result['names'][3][2].replace('...','')
 #     stext = stext.replace(s,'<ins>' + s + '</ins>' ) #<ins>-подчёркивание,<b>-жирный

  # если фио нашли, выделяем
  if result['names'][2]>'':
    for s in result['names'][2].split():
      stext = stext.replace(s,' <b><font color=''red''>' + s + '</font></b> ')
      stext = stext.replace(s.lower(),' <b><font color=''red''>' + s.lower()+ '</font></b> ')
 # else:
 #   # выделим области
 #   if len(result['names'][4])>0 and result['names'][4][2]>'':
 #     s = result['names'][4][2].replace('...','')
 #     stext = stext.replace(s,'<ins>' + s + '</ins>' ) #<ins>-подчёркивание,<b>-жирный
  
  display(Markdown(stext))

# Функция получения ФИО заказчика и имени оператора

def get_person_from_ner(lst_text,dftext):
  # на входе -список фрагментов текста, полная строка текста
  # на выходе - словарь с именами в формате: 
  # [имя оператора, фамилия клиента, фио клиента,
  # позиция + фрагмент текста для поиска имени оператора, 
  # позиция + фрагмент текста для поиска фамилии клиента]
  name_structure = {'names':[]}

  # загрузка справочников фио 
  # ссылка на каталог и файлы https://drive.google.com/drive/folders/1XaeA_qe7RAQce3Zbz_pXI1p6q88AVQ1L?usp=share_link

  # 'russian_names.xlsx' https://docs.google.com/spreadsheets/d/1sZeA1KS3qZmeQ6yEX06FkW3AufbWw5v7/edit?usp=share_link&ouid=117930613337469602374&rtpof=true&sd=true
  sheet_name_id = '1sZeA1KS3qZmeQ6yEX06FkW3AufbWw5v7'

  # 'russian_midnames.xlsx' https://docs.google.com/spreadsheets/d/17M_W7ZYeQWYU9FhhIONB7hZvSJbwNlKO/edit?usp=share_link&ouid=117930613337469602374&rtpof=true&sd=true
  sheet_surname_id = '17M_W7ZYeQWYU9FhhIONB7hZvSJbwNlKO'

  # 'russian_surnames.xlsx' https://docs.google.com/spreadsheets/d/1L1c3D7U6rkM92MlPq3XOhwtcaVCk6UbP/edit?usp=share_link&ouid=117930613337469602374&rtpof=true&sd=true
  sheet_midname_id = '1L1c3D7U6rkM92MlPq3XOhwtcaVCk6UbP'

  # 'foreign_names.xlsx' https://docs.google.com/spreadsheets/d/1OhKtd1STRPwX_74cTbZOOElxwCq8LH1v/edit?usp=share_link&ouid=117930613337469602374&rtpof=true&sd=true
  sheet_foreigname_id = '1OhKtd1STRPwX_74cTbZOOElxwCq8LH1v'

  lst_names,lst_surnames,lst_midnames,lst_foreignnames,lst_fullnames,\
  lst_fullnames_lower = load_fio_from_catalogs(sheet_name_id,sheet_surname_id,sheet_midname_id,sheet_foreigname_id)

  # Определение тригеров - граничные слова для ИМЕНИ оператора и ФИО клиента
  lst_operator_words_before = []
  lst_operator_words_after = []
  lst_fio_words_before = []
  lst_fio_words_after = []
  lst_operator_words_before,lst_operator_words_after,lst_fio_words_before,\
      lst_fio_words_after = get_trigers_for_per()

  # 0.поиск фио в словарях
  result_dict = find_in_dictionaries(dftext,lst_names, lst_midnames, lst_surnames)
  result_dict_midname = find_in_dictionaries_midname(dftext, lst_midnames)
  result_dict_surname = find_in_dictionaries_surname(dftext, lst_surnames)
  result_dict_name = find_in_dictionaries_name(dftext,lst_names)
  # print('result_dict', result_dict)
  # print('result_dict_name', result_dict_name)
  # print('result_dict_midname', result_dict_midname)
  # print('result_dict_surname', result_dict_surname)

# 1. распознавание Natasha
  result_natasha = get_natasha_ent_per(dftext)
  # print('result_natasha', result_natasha)

# 2. распознавание PullEnt
  result_pullenti = get_pullenti_ent_per(dftext)
  # print('result_pullenti', result_pullenti)

  # ОБРАБОТКА НАЙДЕННЫХ NER СУЩНОСТЕЙ:')
  # 1.1. чистка распознанного через Natasha
  ts_ent = result_natasha['ENT'] 
  ts_res,ts_delta = clear_per_ent (result_natasha, lst_fullnames)
  # print('result_natasha, очищенные со словарём сущности:', ts_res)
  # print('result_natasha, отбракованные со словарём сущности:', ts_delta)

# 1.2. определение имени оператора,его позиции
  operator_res = get_operator_name(dftext, ts_res,lst_operator_words_before,\
                                   lst_operator_words_after,result_dict_name,lst_names)
  # print('operator_res',operator_res)
  operator_position_end = 0
  if len(operator_res[2])>0 and int(operator_res[2][1])>0: 
    operator_position_end = int(operator_res[2][1])

# 1.3. определение фио клиента
  fio_res = get_client_fio(dftext[operator_position_end:], ts_res, result_dict_name, result_dict_midname,\
                           result_dict_surname, result_pullenti,\
                           lst_fio_words_before,lst_fio_words_after,\
                           lst_names, lst_midnames, lst_surnames)
  # print('fio_res',fio_res)

# 3. результат распознавания 
  soperator = operator_res[0]
  soperator_info = operator_res[2]
  sfio = str(fio_res[0]).strip()
  sfio_info = fio_res[2]
  sfio_full = str(fio_res[3]).strip()

  name_structure['names'] = [soperator, sfio, sfio_full,soperator_info,sfio_info]

  return(name_structure)

"""## 9. Предобработка текстов для поиска телефонов V2.0.2

### Преобразование чисел распознанных текстом в цифры
Пока ограничусь 3-х значными числами, думаю пока этого будет достаточно.
Последовательность:
1. Нахожу сотни и заменяю на соответствующую цифру, но вместо пробела пишу букву 'h'
2. Нахожу десятки от 20 и заменяю на соответствующую цифру, но вместо пробела пишу букву 'd'
3. Нахожу цифры от 10 до 19 и меняю на соответсвующие наборы вида '1d2'
4. Нахожу цифры от 0 до 9 и меняю на соответсвующие.

На данном этапе есть конструкции вида '1h2d3 ' - не круглое число, '1h2d ', '1д ' - круглые десятки, '1h2 ' - сотня с единицей, '1h ' - круглая сотня. В случае круглой десятки нам нужен ' 0', в случае круглой сотни '  0 0'. Самая опасная конструкция это сотня с единицей. В случае некруглого числа 'h' и 'd' можно было бы просто убрать, но это пересекается с круглыми десятками. Поэтому сначала обрабатываю на круглые сотни убирая 'h'.
5. Заменяю тип 'h2d' на ' 2d'
Оказалось, что при обработке последовательностей типа 20-30 могут образовываться констукции типа 2d3d 
6. Заменяю 'd3d' на 'd 3d'
7. Заменяю тип 'd3 ' на ' 3 0 '
8. Заменяю тип 'h3' на ' 0 3'
9. Заменяю '2d' на '2 0 '
10. Заменяю '1h' на '1 0 0 '
"""

# Минимальное количество последовательно расположенных цифр, которые необходимы для признания послдовательности телефонным номером
MIN_PHONE_DIGITS = 6

# Предварительные шаблоны
n100_900s = [' сто ', ' двести ', ' триста ', ' четыреста ', ' пятьсот ', ' шестьсот ', ' семьсот ', ' восемьсот ', ' девятьсот ']
n100_900_rep = [' 1h', ' 2h', ' 3h', ' 4h', ' 5h', ' 6h', ' 7h', ' 8h', ' 9h']

n20_90s = ['двадцать ', 'тридцать ', 'сорок ', 'пятьдесят ', 'шестьдесят ', 'семьдесят ', 'восемьдесят ', 'девяносто ']
n20_90_rep = ['2d', '3d', '4d', '5d', '6d', '7d', '8d', '9d']

n11_19s = ['десять ', 'одиннадцать ', 'двенадцать ', 'тринадцать ', 'четырнадцать ', 'пятнадцать ', 'шестнадцать ', 'семьнадцать ', 'восемьнадцать ', 'девятнадцать ']
n11_19_rep = ['1d0 ', '1d1 ', '1d2 ', '1d3 ', '1d4 ', '1d5 ', '1d6 ', '1d7 ', '1d8 ', '1d9 ']

n09s = ['ноль ','один ', 'два ', 'три ', 'четыре ', 'пять ', 'шесть ', 'семь ', 'восемь ', 'девять ']
n09_rep = ['0 ','1 ', '2 ', '3 ', '4 ', '5 ', '6 ', '7 ', '8 ', '9 ']

# Списки необходимые для работы функции замены
# Ох уж этот русский язык ОДИНнадцать, воСЕМЬ, воСЕМЬдесят, воСЕМЬсот, девяноСТО. Именно по этому списки нужно взять в обратном порядке
num_text = [['восемь восемьсот '],list(reversed(n20_90s)), list(reversed(n100_900s)), list(reversed(n11_19s)), list(reversed(n09s)), ['h'+ el[0] + 'd' for el in n09_rep], ['d' + el[0] + 'd' for el in n09_rep], [el[0] + 'd ' for el in n09_rep], ['d' + el[0] for el in n09_rep], ['h' + el[0] for el in n09_rep], [el[0] + 'd' for el in n09_rep], [el[0] + 'h' for el in n09_rep]]
num_rep = [['8 8 0 0 '],list(reversed(n20_90_rep)), list(reversed(n100_900_rep)), list(reversed(n11_19_rep)), list(reversed(n09_rep)), [' '+ el[0] + 'd' for el in n09_rep], ['d ' + el[0] + 'd' for el in n09_rep], [el[0] + ' 0 ' for el in n09_rep],  [' ' + el[0] for el in n09_rep], [' 0 ' + el[0] for el in n09_rep], [el[0] + ' 0 ' for el in n09_rep], [el[0] + ' 0 0 ' for el in n09_rep]]

# Списки для преобразования в текст повторяющейся последовательности цифр (сознательно не использовал Ё)
pre_f = ['одна', 'две', 'три', 'четыре', 'пять', 'шесть', 'семь', 'восемь', 'девять', 'десять']
pre_m = ['один', 'два', 'три', 'четыре', 'пять', 'шесть', 'семь', 'восемь', 'девять', 'десять']
numbs = [['ноль', 'ноля', 'нолей'],
         ['единица', 'единицы', 'единиц'],
         ['двойка', 'двойки', 'двоек'],
         ['тройка', 'тройки', 'троек'],
         ['четверка', 'четверки', 'четверок'],
         ['пятерка', 'пятерки', 'пятерок'],
         ['шестерка', 'шестерки', 'шестерок'],
         ['семерка', 'семерки', 'семерок'],
         ['восьмерка', 'восьмерки', 'восьмерок'],
         ['девятка', 'девятки', 'девяток']]

# Функция удаления лишних символов
def text_filter(text):
  filters = ['!','"','#','$','%','&','(',')','*','+',',','-','–','—','.','/','…',':',';','<','=','>','?','@','[',']','^','_','`','{','|','}','~','«','»','\t','\n','\xa0','\ufeff']
  for s in filters:
    text = text.replace(s, '')
  return text

# Функция замены цифр распознаных текстом в цифры
def comnum_to_num_replacer(text):
  text = text + ' '       # дополнительный пробел если в конце цифра
  
  for n in range(len(num_text)):
    for i in range(len(num_text[n])):
      done = True
      while done:
        if num_text[n][i] in text:
          text = text.replace(num_text[n][i], num_rep[n][i])
        else: done = False
  return text.replace('  ',' ')

# Функция замены повторяющихся цифр распознаных как текст в цифры
def repnum_to_num_replacer(text):
  text = text + ' '       # дополнительный пробел если в конце цифра
  text = text.replace('ё','е') # Решил не рисковать с Ё
  
  for q in range(2, 6):   # перебор до максимально возможного количества симметричных цифр (сейчас 5)
    for n in range(10):
      if q < 5:
        if n != 0: repnum = pre_f[q-1] + ' ' + numbs[n][1]
        else: repnum = pre_m[q-1] + ' ' + numbs[0][1]
      else: repnum = pre_f[q-1] + ' ' + numbs[n][2]
      text = text.replace(repnum,str(n)*q)
  return text

# Функция удаляющая пробелы между цифрами
def num_space_remover(text):
  # Поскольку пробелы удаляются для того, чтобы найти последовательность цифр для телефонов, то добавил также удаление лишних символов
  for char in ['(',')','-']:
    text = text.replace(char,'')
  for k in range(10):
    for l in range(10):
      done = True
      while done:
        if (str(k) + ' ' + str(l)) in text:
          text = text.replace(str(k) + ' ' + str(l), str(k) + str(l))
        else: done = False
  return text

# Функция добавляющая пробелы между цифрами
def num_spacer(text):
  for k in range(10):
    for l in range(10):
      done = True
      while done:
        if (str(k) + str(l)) in text:
          text = text.replace(str(k) + str(l), str(k) + ' ' + str(l))
        else: done = False
  return text

# Функция замены всех видов текстовых цифр
def nums_replacer(text):
  text = text.lower()
  text_filter(text)
  text = repnum_to_num_replacer(text)
  text = comnum_to_num_replacer(text)
  return text

# Функция поиска цифрового блока в строке 
def find_num (text, min_len):
  digits = [str(q) for q in range(10)]
  result = []
  i = 0
  while i < len(text):
    if text[i] in digits: 
      d = 1
      while i+d < len(text) and text[i+d] in digits:
        d += 1
      if d >= min_len:
        result.append(text[i:i+d])
      i += d
    i += 1
  return result

# Функция маркировки текста для подсветки заданного номера
def marknum(text, num_str, color):
  nums_indexes = []
  nums_in_text = ''
  indexes = []
  # Выборка всех цифр и сооттветствующих им индексов в тексте
  for i in range(len(text)):
    if text[i] in ['0','1','2','3','4','5','6','7','8','9','0']:
      nums_in_text += text[i]
      nums_indexes.append(i)
  nums_in_text +='$'    # для проверки окончания последовательности собранных цифр
  j = 0
  while (j+len(num_str)) < len(nums_in_text): # Прохожу по всей собранной последовательности цифр
    while (nums_in_text[j:j+len(num_str)] != num_str) and (nums_in_text[j+len(num_str)] != '$'): # Ищу совпадение с заданым числом
      j += 1
    if nums_in_text[j+len(num_str)-1] != '$':  # Если выход из while не из-за окончания последовательности
      indexes += nums_indexes[j:j+len(num_str)]
    j += len(num_str)

  stext = ''
  for l in range(len(text)):
    if l in indexes:
      stext += '<b><font color=' + color + '>' + text[l] + '</font></b>'
    else: stext += text[l]

  return stext.replace('</font></b><b><font color=' + color + '>','') # Убираю лишнее форматирование у стоящих рядом цифр

# Функция печати текста с подсветкой заданных номеров
def printnums(text, num_lst, color = 'Aquamarine'):
  for i in range(len(num_lst)):
    text = marknum(text, num_lst[i], color)
  display(Markdown(text))

"""## 10. Поиск товаров по словарю V2.0.2"""

# Словарь товаров
PRODUCTS_DICT = {'клареол': ['клареол', 'аквариу', 'ареола', 'вариола', 'кариу', 'кварил', 'клариус', 'клорел', 'кориол', 'крильона', 'лариол', 'ларион', 'триалога', 'флареон'],
            'велотренажёр': ['тренажёр', 'тренажер', 'велот'],
            'усилитель звука': ['усилит', 'чудо слу', 'слуховой аппарат', 'глух', 'звука ', 'ушной', 'поселите'],
            'браслет здоровье':['браслет', 'умный браслет'],
            'шиатцу подушка': ['подушк', 'шиат', 'подушечк', 'массажная', 'шиацу', 'шацу', 'массажжер'],
            'обогреватель': ['обогреват', 'быстро тепл', 'быстрое тепл' 'ру робот', 'теплорова', 'робус тепл', 'рубус тепл'],
            'термобельё': ['термобельe','термобел', 'бельё', 'твоё тепло', 'твое тепло'],
            'измельчитель': ['измельчител', 'измелчит', 'джой', 'дарима', 'дреман', 'алиман'],
            'версаль шторы': ['шторы', 'версал', 'штора', 'штору', 'перса', 'стора'],
            'пест репеллер': ['репеллер', 'отпугиват', 'насеком', 'обывателя грузинов', 'таракан', 'ловушка для', 'ловушки для'],
            'швабра': ['швабр', 'флекс'],
            'мультипечь': ['мультипеч', 'печь'],
            'шторм пылесос': ['пылесос', 'шторм', 'полисто'],
            'юна': ['система юн', 'системы юн', 'систему юн', 'система июна', 'системы июна', 'систему июна', 'тёну'],
            'Богатый урожай': ['парник ', ' теплиц'],
            'Торнадика': ['культиватор', 'торнадик', 'торнадо', 'торна дик', 'окучиват', 'картошк', 'картошечк'],
            'Флебозол': ['флебо', 'фледо', 'лебозо', 'ивазо', 'сеазал', 'хлеба зо', 'хлёбово'],
            'Секрет императора': ['секрет имп', 'сикрет имп', 'император'],
            'Мозг терапи': ['мозг тера', 'мозг тела', 'мог терапи', 'мо ерпи', 'мозг тереби', 'мы терапи', 'из моллюска', 'помощь мозг', 'мозга памят', 'для мозга', 'японский нам', 'насчет мозга'],
            'Сустаздрав': ['суставздр', 'сустазд'],
            'союз апполон': ['союз аполлон', 'апполон', 'аполон', 'апалон', 'аппалон', 'опалон', 'аполлос', 'аполлон'],
            'Мувмент гель': ['мамингель', 'мумент', 'мувмент', 'мулимен', 'мендел', 'вентигель', 'мовенгенгелен', 'момент гель', 'момент ген', 'лемент гель', 'мои гель', 'ман гель', 'гель момент', 'гели моменту'],
            'Цистинет': ['цистинет', 'цастинет', 'цистенет'],
            'Простатрикум': ['простатрикум'],
            'скидочная карта': ['скидочная','карта','скидочной','карты','скидочную','карту'],
            'набор ножей': ['набор ножей']
           }

import re

def main_product_search(text, products):


    ret = {
            'names': {},
            'phones': {},
            'products': {}
          }

    if text is None or len(text) == 0:
        return ret

    products_dict = product_finder(text, products)
    ret['products'] = products_dict

    return ret

"""
    Функция по поиску товаров.

       Аргументы:
       text -- текст для поиска
       products -- словарь товаров
       show_detail - показывать дополнительную информацию (default = False)
    """
def product_finder(text, products, show_detail = False):
    
    MAX_DETAIL_SYMBOLS = 50

    prep_text = prepare_text(text)

    prep_products = prepare_products(products)

    dct = dict()
    for product in prep_products.items():
        vals = flatten_key_values(product)

        for item in vals:

            start, end, matched_product = find_product_pos_by_regex(prep_text, item)

            inf = {
                'match': matched_product,
            }

            if start > -1:
                if not show_detail:
                    dct[vals[0]] = inf
                else:
                    str_left = get_parts(prep_text, start, MAX_DETAIL_SYMBOLS, 'left')
                    str_right = get_parts(prep_text, end, MAX_DETAIL_SYMBOLS, 'right')

                    text_product_part = ''.join([str_left, matched_product, str_right])

                    inf['text_product_part'] = text_product_part
                    inf['position'] = {
                            'start': start,
                            'end': end
                         }

                    dct[vals[0]] = inf
                break

    return dct

def prepare_products(products):
    ret = dict()

    for item in products.items():
        tpl = prepare_product_items(item)

        ret[tpl[0]] = tpl[1]

    return ret

def prepare_product_items(product_items):
    tpl = replace_to_e_prod_items(product_items)

    return tpl

def prepare_text(text):
    ret = replace_yo_to_e(text)

    return ret

def replace_to_e_prod_items(product_items):
    #k = replace_yo_to_e(product_items[0])
    k = product_items[0]  # Не хочу убирать ё в названии товара
    vs = [replace_yo_to_e(item) for item in product_items[1]]

    return (k, vs)

"""
    Заменяем в тексте все ё на е.

          Аргументы:
          text -- текст для замены
    """
def replace_yo_to_e(text):
    mapping = str.maketrans("ё", "e")
    ret = text.translate(mapping)

    return ret

def find_product_pos_by_regex(text, product):
    match = re.search(product, text, re.IGNORECASE)

    if match is not None:
        return match.start(), match.end(), match.group()

    return -1, -1, None

def flatten_key_values(key_values):
    tpl = tuple([key_values[0]] + key_values[1])

    return tpl

def div_by_regex(text, regex_condition):
    lst = re.split(regex_condition, text)

    return lst

def get_parts(text, pos, num_symb, direction):

    if direction == 'left':
        ret = text[pos - num_symb : pos]

        return ret
    elif direction == 'right':
        ret = text[pos : pos + num_symb]

        return ret
    else:
        raise Exception('Invalid argument direction')

# Функция извлечения слов для маркировки из словоря товаром соответствующих найденным в продуктам
def extract_word_for_marking(finded_prods_dict):
  prods_list = []
  for product in finded_prods_dict['products'].keys():
    prods_list.append(product)
    prods_list.append(finded_prods_dict['products'][product]['match'])
  return prods_list
  
# Функция подсветки в тексте найденного списка слов
def printmarkedwords(text, words_list, color):
  for word in words_list:
    text = text.replace(word,'<b><font color='+ color + '>' + word + '</font></b>')
  display(Markdown(text))

"""## 11. PullEnti для поиска остальных сущностей"""

# Инициализация расширенной версии процессора
processor_other = Processor([
    MONEY,
    PHONE,
    GEO,
    ADDRESS,
    ORGANIZATION,
    PERSON
])

# Процессор для поиска адреса
processor_address = Processor([
    GEO,
    ADDRESS
])

# Процессор для поиска цены
processor_price = Processor([
    MONEY
])

# Извлечение заданной сущности из результата PullEnty
def extract_entity(result, entity):
  results_values = []
  for m in result.matches:
    if m.referent.label == entity:
      for slot in m.referent.slots:
        print(slot)

# Функция извлечения фрагментов адреса
def extract_shortcut(referent, level=0):
    tmp = {}
    a = ""
    b = ""
    for key in referent.__shortcuts__:
        value = getattr(referent, key)
        if value in (None, 0, -1):
          continue
        if isinstance(value, Referent):
          tmp.update(extract_shortcut(value, level + 1))
        else:
          if key == 'type':
            a = value
          if key == 'name':
            b = value
          if key == 'house':
            a = "дом"
            b = value
            tmp[a] = b
          if key == 'flat':
            a = "квартира"
            b = value
            tmp[a] = b
          if key == 'corpus':
            a = "корпус"
            b = value
            tmp[a] = b
    tmp[a] = b
    return tmp

# Функция склейки словарей
def add_dict(main_dict, addition):
  for key in addition.keys():
      if key not in main_dict.keys():
        main_dict[key] = addition[key]
      else:
        if main_dict[key] != addition[key]:
          for word in addition[key].split():
            if word not in main_dict[key]:
              main_dict[key] += ' ' + word
  return main_dict

# Функция извлечения словаря адреса из результата PullEnty
def extract_address(result, adr_dict = {}):
  for rr in result.matches:
    tmp = extract_shortcut(rr.referent)
    add_dict(adr_dict, tmp)
  return adr_dict

# Функция визуализации адреса
def print_address(adr_dict, color = 'Indigo'):
  for key in ['край','область','город','посёлок','район','улица','дом','квартира']:
    if key in adr_dict.keys():
      text = key[0].upper() + key[1:] + ': '
      for word in adr_dict[key].split():
        text += '<b><font color='+ color + '>' + word[0] + word[1:].lower() + '</font></b>, '
      display(Markdown(text[:-2]))

# Извлечение заданной сущности из результата PullEnty
def extract_money(result, money_set):
  for m in result.matches:
    if m.referent.label == 'MONEY':
      for slot in m.referent.slots:
      #  if slot.key == 'CURRENCY':
      #    currency = slot.value
        if slot.key == 'VALUE':
      #    num = slot.value
          money_set.add(slot.value)
      #money_set.add(num + ' ' + currency.replace('RUB','руб')) ## Пока заменяю только рубли, вряд ли другая валюта найдется (но если найдется будет в оригинале)
  return money_set

# Функция визуализации цены
def show_me_your_money(money_set, color = 'Goldenrod'):
  for price in money_set:
    display(Markdown('<b><font color='+ color + '>' + price + '</font></b>'))

"""# Итог: Функция комплексной обработки
Распознает аудио в текст, классификацирует, находит триггеры и сущности, а также выводит обраруженные сущности и соответвующие им аудиофрагменты
"""

# Функция для наглядного отображения нарезки одного файла
def File_Visualization(audio_path):
  print('\nПрослушать оригинальный файл:')
  ipd.display(ipd.Audio(audio_path))
  Signal, sr = librosa.load(audio_path)
  Cleared_Signal = cut_all_music(Signal)
  List_of_Timing = phrase_by_phrase(Cleared_Signal)
  path_wav = 'temporary.wav'
  sf.write('temporary.wav', Cleared_Signal, sr)
  print('\nПрослушать файл после удаления музыки:')
  ipd.display(ipd.Audio(path_wav))
  separec_All_timed_visualized(path_wav, List_of_Timing)

# Комплексная функция для наглядного отображения результатов поиска триггеров
def Total_Visualization(audio_path = 'sample.mp3', # Путь к файлу
                        rec_tool_main = '',        # 'yandex' - выбор Яндекс в качестве основной системы распознавания речи (любое другое значение google)
                        rec_tool_NER = ''          # 'yandex' - выбор Яндекс в качестве системы распознавания теестов для NER (любое другое значение google)
                        ):
  # Вывод на прослушивание оригинального файла
  display(Markdown('<br><b>Прослушать оригинальный файл:</b>'))
  ipd.display(ipd.Audio(audio_path))

  # Удаление музыкальных фрагментов и автоответчика
  Signal, sr = librosa.load(audio_path)
  Cleared_Signal = cut_all_music(Signal)

  # Проверка, что осталось что-либо для обработки
  if len(Cleared_Signal) == 0:
    print('Файл не содержит звуковых данных для анализа. Наиболее вероятный класс:', CLASS_LIST[5])
  else:
    # Создание wav файла из обработанного сигнала (возможно это костыль, но не wav вроде SpechRecognition не отдать)
    path_wav = 'temporary.wav'
    sf.write('temporary.wav', Cleared_Signal, sr)

    display(Markdown('<br><b>Прослушать файл после удаления музыки:</b>'))
    ipd.display(ipd.Audio(path_wav))

    # Определение интервалов для фраз и передача фраз на распознавание 
    List_of_Timing = phrase_by_phrase(Cleared_Signal)
    if rec_tool_main == 'yandex':
      textes, starts, durations = separec_Yandex_timed(path_wav, List_of_Timing)
    else:
      textes, starts, durations = separec_Google_timed(path_wav, List_of_Timing)
    
    # Проверка на наличие распознанных текстов
    if len(textes) == 0:
      print('В файле не обнаружены речевые данные для анализа. Наиболее вероятный класс:', CLASS_LIST[4])
    else:
      # Склейка текста для классификации
      text = glue_uncrossed_text(textes)

      # Запуск предсказания сети
      # Преобразую строку в последовательность индексов согласно частотному словарю с помощью токенайзера, формирую в виде разреженной матрицы (bag of words) и отправляю в НС
      y_pred = model.predict(np.array(tokenizer.sequences_to_matrix(tokenizer.texts_to_sequences([text]))), verbose='0')
      ver = give_predict(text)
      cls = CLASS_LIST[np.argmax(y_pred)]
      ver = round(np.max(y_pred)*100,2)

      # Отображение результата предсказания НС
      print('\n'+'-.'*MAX_LINE+'-')
      message = '<b><br>Сеть предсказала класс "'+ cls + '" с вероятностью '+ str(ver) + ' %</b>'
      display(Markdown(message))
      print(' НC предсказала вероятную принадлежность звонка к другим классам в порядке убывания:')
      y_pred_list = y_pred.tolist()[0]
      for i in range(1,len(CLASS_LIST)):
        ind = y_pred_list.index(sorted(y_pred_list, reverse=True)[i])
        print('   {cls:<8.8} c вероятностью {ver:>8.4%}'.format(cls=CLASS_LIST[ind], ver=y_pred_list[ind]))

      print('\n'+'-.'*MAX_LINE+'-\n')
      display(Markdown('<b>NER поиск ФИО в тексте без предварительной обработки дает следующий результат:</b>'))
      # Передача полного текста для распознавания имени в NER 
      result_names = get_person_from_ner(textes, text)
      # Визуализация результата - печать текста с выделенными именами оператора и фио
      printmd(text_cutter(text,'<br>'),result_names)
      print('\n'+'-.'*MAX_LINE+'-\n')
      display(Markdown('<b>Поиск цифрового блока без предварительной обработки дает следующий результат:</b>'))
      result_nums = find_num(num_space_remover(nums_replacer(text)), MIN_PHONE_DIGITS) # Перед поиском блоков цифр, заменяю все виды текстовых цифр и склеиваю в блоки
      if len(result_nums) == 0:
        print('Последовательность цифр заданной длины не обнаружена.')
      else:
        printnums(nums_replacer(text_cutter(text,'<br>')), result_nums) # Чтобы верно подсветить нужно заменить все виды цифр, а вот склеивать для наглядности уже не надо

      # Запуск PullEnty для обнаружение прочих сущностей
      print('\n'+'-.'*MAX_LINE+'-\n')
      display(Markdown('<b>PullEnty может найти следующие сущности.</b>'))  
      result_OTH = processor_other(text)
      display(result_OTH.graph)

      # Этот цикл позволит запускать поиск сущностей только в случае подходящих нам классов, но пока мы не уверены в точности классификатора, он только для информации. 
      if cls in CLASS_LIST[:3]:
        print('\nВ данном классе можно искать сущности.')
      else:
        print('\nЕсли файл классифицирован верно, то обнаружение сущностей является маловероятным.')

      # Поиск триггеров и соответствующих им наименований сущностей
      scanned_triggers, scanned_entities, tr_starts, tr_durations = entrigger_scanner(textes, starts, durations)

      # Выделение отдельных сущностей и соответствующих им временных интервалов
      if len(tr_starts) == 0:
        print('\nТриггеры сущностей не обнаружены!')
      else:
        # Сортировка фрагментов по каждой сущности
        sortr_starts, sortr_durations = entities_locator(scanned_entities, tr_starts, tr_durations)
        # Вывод фрагментов аудиофайла с найденными сущностями
        visualize_fragments(path_wav,  sortr_starts, sortr_durations, textes, starts)

        # Выделение интервалов фрагментов сущностей на основании заданных границ
        en_starts, en_ends = entities_fragments(sortr_starts, sortr_durations)
        
        # Список текстов для каждой сущности
        NER_textes = textes_for_NER(path_wav, en_starts, en_ends, rec_tool = rec_tool_NER)

        # Отображение всех текстов для каждой сущности
        print('-'+'.-'*MAX_LINE)
        for entity in ENTITIES_UNIC:
          message = '<b><br>Тексты для поиска сущности '+ entity + ':</b>'
          index = ENTITIES_UNIC.index(entity)
          display(Markdown(message))
          if len(NER_textes[index]) != 0:
            for n in range(len(NER_textes[index])):
              print('Текст №',n,sep='')
              print_lim(NER_textes[index][n])
          else: print('Отсутствуют.')

        # Обработка текстов найденных для ФИО
        index = ENTITIES_UNIC.index('ФИО')
        if len(NER_textes[index]) != 0:
          print('\n' + '-.'*MAX_LINE+'-')
          display(Markdown('<b><br>Обработка текстов приготовленных для поиска сущности ФИО:</b>'))
          for n in range(len(NER_textes[index])):
            print('Текст №',n,sep='')
            printmd(text_cutter(NER_textes[index][n],'<br>'), get_person_from_ner([], NER_textes[index][n]))
        
        # Обработка текстов найденных для ТЕЛЕФОН
        index = ENTITIES_UNIC.index('ТЕЛЕФОН')
        if len(NER_textes[index]) != 0:
          print('-.'*MAX_LINE+'-')
          display(Markdown('<b><br>Обработка текстов приготовленных для поиска сущности ТЕЛЕФОН:</b>'))
          for n in range(len(NER_textes[index])):
            print('Текст №',n,sep='')
            result_nums = find_num(num_space_remover(nums_replacer(NER_textes[index][n])), MIN_PHONE_DIGITS) # В тексте от Google нужно по идее только удалить пробелы, но на всякий случай обработаю также текстовые цифры
            if len(result_nums) == 0:
              print('Последовательность цифр заданной длины не обнаружена.')
            else:
              printnums(nums_replacer(text_cutter(NER_textes[index][n],'<br>')), result_nums, color = 'Fuchsia')

        # Обработка текстов найденных для АДРЕС
        index = ENTITIES_UNIC.index('АДРЕС')
        if len(NER_textes[index]) != 0:
          print('-.'*MAX_LINE+'-')
          display(Markdown('<b><br>Обработка текстов приготовленных для поиска сущности АДРЕС:</b>'))
          for n in range(len(NER_textes[index])):
            print('Текст №',n,sep='')
            result_ADR = processor_address(NER_textes[index][n])
            adr_dict = extract_address(result_ADR, {})
            print_address(adr_dict, 'Brown')

        # Обработка текстов найденных для ТОВАР
        index = ENTITIES_UNIC.index('ТОВАР')
        if len(NER_textes[index]) != 0:
          print('\n' + '-.'*MAX_LINE+'-')
          display(Markdown('<b><br>Обработка текстов приготовленных для поиска сущности ТОВАР:</b>'))
          for n in range(len(NER_textes[index])):
            print('Текст №',n,sep='')
            finded_prods_dict = main_product_search(NER_textes[index][n], PRODUCTS_DICT)
            prods_list = extract_word_for_marking(finded_prods_dict)
            if len(prods_list) != 0:
              printmarkedwords(NER_textes[index][n], prods_list, 'LimeGreen')
            else: print('Товары в этом фрагменте не обнаружены.')

        # Обработка текстов найденных для ЦЕНА
        index = ENTITIES_UNIC.index('ЦЕНА')
        if len(NER_textes[index]) != 0:
          print('-.'*MAX_LINE+'-')
          display(Markdown('<b><br>Обработка текстов приготовленных для поиска сущности ЦЕНА:</b>'))
          for n in range(len(NER_textes[index])):
            print('Текст №',n,sep='')
            # Дополнительный блок для поиска числовых последовательностей в NER текстах ЦЕНА
            price_nums = set()
            result_pnums = find_num(num_space_remover(nums_replacer(NER_textes[index][n])), 1)  # В тексте от Google нужно по идее только удалить пробелы, но на всякий случай обработаю также текстовые цифры
            if len(result_pnums) != 0:
              for num in result_pnums:
                price_nums.add(num)

            result_PE = processor_price(NER_textes[index][n])
            show_me_your_money(extract_money(result_PE, price_nums), 'Goldenrod')

# Комплексная функция для извлечения сущностей
def Total_Analization(audio_path = 'sample.mp3', # Путь к файлу
                      rec_tool_main = '',        # 'yandex' - выбор Яндекс в качестве основной системы распознавания речи (любое другое значение google)
                      rec_tool_NER = ''          # 'yandex' - выбор Яндекс в качестве системы распознавания теестов для NER (любое другое значение google)
                      ):
  # Удаление музыкальных фрагментов и автоответчика
  Signal, sr = librosa.load(audio_path)
  Cleared_Signal = cut_all_music(Signal)

  # Задаю пустые переменные для возврата на случай если не будет каких-то данных для анализа
  text = ''
  NER_textes = []
  names = {}
  phone_nums = set()
  adr_dict = {}
  products = set()
  money_set = set()
  ver = 0
  y_pred_list = [0]*len(CLASS_LIST)

  # Проверка, что осталось что-либо для обработки
  if len(Cleared_Signal) == 0:
    cls = CLASS_LIST[5]
    ver = 0
  else:
    # Создание wav файла из обработанного сигнала (возможно это костыль, но не wav вроде SpechRecognition не отдать)
    path_wav = 'temporary.wav'
    sf.write('temporary.wav', Cleared_Signal, sr)

    # Определение интервалов для фраз и передача фраз на распознавание 
    List_of_Timing = phrase_by_phrase(Cleared_Signal)
    if rec_tool_main == 'yandex':
      textes, starts, durations = separec_Yandex_timed(path_wav, List_of_Timing)
    else:
      textes, starts, durations = separec_Google_timed(path_wav, List_of_Timing)


    # Проверка на наличие распознанных текстов
    if len(textes) == 0:
      cls = CLASS_LIST[4]
      ver = 0
    else:
      # Склейка текста для классификации
      text = glue_uncrossed_text(textes)

      # Запуск предсказания сети
      # Преобразую строку в последовательность индексов согласно частотному словарю с помощью токенайзера, формирую в виде разреженной матрицы (bag of words) и отправляю в НС
      y_pred = model.predict(np.array(tokenizer.sequences_to_matrix(tokenizer.texts_to_sequences([text]))), verbose='0')
      ver = give_predict(text)
      cls = CLASS_LIST[np.argmax(y_pred)]
      ver = round(np.max(y_pred)*100,2)
      y_pred_list = y_pred.tolist()[0]

      # Передача полного текста для распознавания имени в NER
      result = get_person_from_ner(textes, text)

      if result['names'][0]>'':
        names = add_dict(names,{'Оператор':result['names'][0]})
      if result['names'][2]>'':
        names = add_dict(names,{'Клиент':result['names'][2]})

      # Поиск телефона в полном тексте
      result_nums = find_num(num_space_remover(nums_replacer(text)), MIN_PHONE_DIGITS)  # В тексте от Google нужно по идее только удалить пробелы, но на всякий случай обработаю также текстовые цифры
      if len(result_nums) != 0:
        for num in result_nums:
          phone_nums.add(num)

      # Передача полного текста для поиска адреса в PullEnty (может взять лишнего?!!) 
      result_ADR = processor_address(text)
      adr_dict = extract_address(result_ADR, adr_dict)

      # Поиск товаров в полном тексте по словарю
      finded_prods_dict = main_product_search(text, PRODUCTS_DICT)
      prods_list = []
      for product in finded_prods_dict['products'].keys():
        products.add(product)

      # Передача полного текста для поиска цены в PullEnty
      result_PRS = processor_price(text)
      money_set = extract_money(result_PRS, set())

      # Этот цикл позволит запускать поиск сущностей только в случае подходящих нам классов, но пока мы не уверены в точности классификатора, он только для информации. 
      if cls in CLASS_LIST[:3]:

        # Поиск триггеров и соответствующих им наименований сущностей
        scanned_triggers, scanned_entities, tr_starts, tr_durations = entrigger_scanner(textes, starts, durations)
        # Выделение отдельных сущностей и соответствующих им временных интервалов если триггеры обнаружены
        if len(tr_starts) != 0:
          # Сортировка фрагментов по каждой сущности
          sortr_starts, sortr_durations = entities_locator(scanned_entities, tr_starts, tr_durations)

          # Выделение интервалов фрагментов сущностей на основании заданных границ
          en_starts, en_ends = entities_fragments(sortr_starts, sortr_durations)
          
          # Список текстов распознанных для каждой сущности 
          NER_textes = textes_for_NER(path_wav, en_starts, en_ends, rec_tool = rec_tool_NER)

          # Обработка текстов найденных для ФИО
          index = ENTITIES_UNIC.index('ФИО')
          if len(NER_textes[index]) != 0:
            for n in range(len(NER_textes[index])):
              result = get_person_from_ner([], NER_textes[index][n])
              if result['names'][0]>'':
                names = add_dict(names,{'Оператор':result['names'][0]})
              if result['names'][2]>'':
                names = add_dict(names,{'Клиент':result['names'][2]})
   
          # Обработка текстов найденных для ТЕЛЕФОН
          index = ENTITIES_UNIC.index('ТЕЛЕФОН')
          if len(NER_textes[index]) != 0:
            for n in range(len(NER_textes[index])):
              result_nums = find_num(num_space_remover(nums_replacer(NER_textes[index][n])), MIN_PHONE_DIGITS)  # В тексте от Google нужно по идее только удалить пробелы, но на всякий случай обработаю также текстовые цифры
              if len(result_nums) != 0:
                for num in result_nums:
                  phone_nums.add(num)

          # Обработка текстов найденных для АДРЕС
          index = ENTITIES_UNIC.index('АДРЕС')
          if len(NER_textes[index]) != 0:
            for n in range(len(NER_textes[index])):
              result_ADR = processor_address(NER_textes[index][n])
              adr_dict = extract_address(result_ADR, adr_dict)   # Здесь пока собираю адреса вместе без повторов для большей компактности, но в некоторых случаях может получиться путаница (надо проверить)

          # Обработка текстов найденных для ТОВАР
          index = ENTITIES_UNIC.index('ТОВАР')
          if len(NER_textes[index]) != 0:
            for n in range(len(NER_textes[index])):
              finded_prods_dict = main_product_search(NER_textes[index][n], PRODUCTS_DICT)
              prods_list = []
              for product in finded_prods_dict['products'].keys():
                products.add(product)

          # Обработка текстов найденных для ЦЕНА
          index = ENTITIES_UNIC.index('ЦЕНА')
          if len(NER_textes[index]) != 0:           
            for n in range(len(NER_textes[index])):
              # Дополнительный блок для поиска числовых послежовательностей в NER текстах ЦЕНА
              money_set = set()
              result_pnums = find_num(num_space_remover(nums_replacer(NER_textes[index][n])), 1)  # В тексте от Google нужно по идее только удалить пробелы, но на всякий случай обработаю также текстовые цифры
              if len(result_pnums) != 0:
                for num in result_pnums:
                  money_set.add(num)

              result_PRS = processor_price(NER_textes[index][n])
              money_set = extract_money(result_PRS, money_set)

  return {'ТЕКСТ': text, 'КЛАСС': cls, 'ВЕРОЯТНОСТЬ': ver, 'РАСПРЕДЕЛЕНИЕ': y_pred_list, 'ФИО': names, 'ТЕЛЕФОН': phone_nums, 'АДРЕС': adr_dict, 'ТОВАР': products, 'ЦЕНА': money_set}
  
  # При необходимости глубокого анализа обработки в возвращаемые значения можно добавить NER_textes

# Функция визуализации результатат анализа
def Аnalization_Visualization(Analization_Result,
                              colors = ['blue','red','Fuchsia','Brown','LimeGreen','Goldenrod'],    # Список цветов для подсветки сущностей
                              line_length = 50,      # Количество символов в линиях разделяющих информацию (будет удвоено)
                              text_length = 100):    # Количество символов в строках текста
  # Отображение результата предсказания НС
  print('-.'*line_length +'-')
  message = '<b>Сеть предсказала класс "'+ Analization_Result['КЛАСС'] + '" с вероятностью '+ str(Analization_Result['ВЕРОЯТНОСТЬ']) + ' %</b>'
  display(Markdown(message))
  print('НC предсказала вероятную принадлежность звонка к другим классам в порядке убывания:')
  y_pred_list = Analization_Result['РАСПРЕДЕЛЕНИЕ']
  endsy = '; '
  for i in range(1,len(CLASS_LIST)):
    ind = y_pred_list.index(sorted(y_pred_list, reverse=True)[i])
    if i+1 == len(CLASS_LIST): endsy = '.\n\n'
    print(CLASS_LIST[ind],' - ',round(y_pred_list[ind]*100,2),' %',sep='',end=endsy)

  # Проверка наличия извлеченных данных
  if Analization_Result['ВЕРОЯТНОСТЬ'] == 0:
    if Analization_Result['КЛАСС'] == '6_Auto':
      print('Файл не содержал звуковых данных для анализа.')
    elif Analization_Result['КЛАСС'] == '5_Prank':
      print('В файле не были обнаружены речевые данные для анализа.')
  else:
      text = text_cutter(Analization_Result['ТЕКСТ'],'<br>',text_length)
      print('-.'*line_length+'-')
      display(Markdown('<b>В тексте различными способами обработки были найдены следующие сущности:</b>'))
      
      # Вывод и разметка сущности ИМЯ
      if len(Analization_Result['ФИО']) != 0:
        if 'Оператор' in Analization_Result['ФИО'].keys():
          print(' Оператор: '+ Analization_Result['ФИО']['Оператор'].replace(' ',', '))
          for name in Analization_Result['ФИО']['Оператор'].split():
            text = text.replace(name,'<b><font color=' + colors[0] + '>' + name + '</font></b>')
            text = text.replace(name.lower(),'<b><font color=' + colors[0] + '>' + name.lower() + '</font></b>')

        if 'Клиент' in Analization_Result['ФИО'].keys():
          print(' Клиент: '+ Analization_Result['ФИО']['Клиент'].replace(' ',', '))
          for name in Analization_Result['ФИО']['Клиент'].split():
            text = text.replace(name,'<b><font color=' + colors[1] + '>' + name + '</font></b>')
      
      # Вывод и разметка сущности ТЕЛЕФОН
      if len(Analization_Result['ТЕЛЕФОН']) != 0:
        phone_text = ' Телефон: '
        for phone in Analization_Result['ТЕЛЕФОН']:
          phone_text += phone + ', '
          text = marknum(text, phone, colors[2])
        print(phone_text[:-2])
      
      # Вывод и разметка сущности АДРЕС
      if len(Analization_Result['АДРЕС']) != 0:
        print(' Адрес:')
        for key in Analization_Result['АДРЕС'].keys():
          text = text.replace(key,'<ins>' + key + '</ins>')
          adr_text = '   ' + key[0].upper() + key[1:]+ ': '
          for word in Analization_Result['АДРЕС'][key].split():
            adr_text += word[0] + word[1:].lower() + ', '
            text = text.replace(word[0] + word[1:].lower(),'<b><font color=' + colors[3] + '>' + word[0] + word[1:].lower() + '</font></b>')
          print(adr_text[:-2])          

      # Вывод и разметка сущности ТОВАР
      if len(Analization_Result['ТОВАР']) != 0:
        product_text = ' Товар: '
        for product in Analization_Result['ТОВАР']:
          product_text += product + ', '
          text = text.replace(product,'<b><font color=' + colors[4] + '>' + product + '</font></b>')
          text = text.replace(product[0].upper() + product[1:],'<b><font color=' + colors[4] + '>' + product[0].upper() + product[1:].lower() + '</font></b>')
        print(product_text[:-2])
      
      # Вывод и разметка сущности ЦЕНА
      if len(Analization_Result['ЦЕНА']) != 0:
        print(' Цена:')
        for price in Analization_Result['ЦЕНА']:
          print('   ' + price)
          text = marknum(text, price.replace(' руб',''), colors[5])

      # Вывод размеченного текста
      display(Markdown('<b>Разметка текста обнаруженными сущностями дает следующий результат:</b>'))
      display(Markdown(text))

      print('-.'*line_length+'-\n')

# Функция анализа предраспознанных текстов
def Text_Analization(txt_path = 'sample.txt'):
  
  # Открываю текстовый файл
  with open(txt_path, 'r', encoding="cp1251") as f:
      text = f.read()

  # Задаю пустые переменные для возврата на случай если не будет каких-то данных для анализа
  NER_textes = []
  names = {}
  phone_nums = set()
  adr_dict = {}
  products = set()
  money_set = set()

  # Запуск предсказания сети
  # Преобразую строку в последовательность индексов согласно частотному словарю с помощью токенайзера, формирую в виде разреженной матрицы (bag of words) и отправляю в НС
  y_pred = model.predict(np.array(tokenizer.sequences_to_matrix(tokenizer.texts_to_sequences([text]))), verbose='0')
  ver = give_predict(text)
  cls = CLASS_LIST[np.argmax(y_pred)]
  ver = round(np.max(y_pred)*100,2)
  y_pred_list = y_pred.tolist()[0]

  # Передача полного текста для распознавания имени в NER
  result = get_person_from_ner('', text)

  if result['names'][0]>'':
    names = add_dict(names,{'Оператор':result['names'][0]})
  if result['names'][2]>'':
    names = add_dict(names,{'Клиент':result['names'][2]})

  # Поиск телефона в полном тексте
  result_nums = find_num(num_space_remover(nums_replacer(text)), MIN_PHONE_DIGITS)  # В тексте от Google нужно по идее только удалить пробелы, но на всякий случай обработаю также текстовые цифры
  if len(result_nums) != 0:
    for num in result_nums:
      phone_nums.add(num)

  # Передача полного текста для поиска адреса в PullEnty (может взять лишнего?!!) 
  result_ADR = processor_address(text)
  adr_dict = extract_address(result_ADR, adr_dict)

  # Поиск товаров в полном тексте по словарю
  finded_prods_dict = main_product_search(text, PRODUCTS_DICT)
  prods_list = []
  for product in finded_prods_dict['products'].keys():
    products.add(product)

  # Передача полного текста для поиска цены в PullEnty
    result_PRS = processor_price(text)
    money_set = extract_money(result_PRS, set())

  return {'ТЕКСТ': text, 'КЛАСС': cls, 'ВЕРОЯТНОСТЬ': ver, 'РАСПРЕДЕЛЕНИЕ': y_pred_list, 'ФИО': names, 'ТЕЛЕФОН': phone_nums, 'АДРЕС': adr_dict, 'ТОВАР': products, 'ЦЕНА': money_set}
