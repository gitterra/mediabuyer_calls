from time import time
from scipy.signal import argrelextrema

import numpy as np
import librosa

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