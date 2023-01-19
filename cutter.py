import numpy as np
import cleaner 

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
    cut_Spectr = cleaner.Fourier(Signal_without_musik, MAX_FREQ) 
    N_spectr = cut_Spectr.shape[1]

    TRIM_STRIP = 85
    N_SMAL = 4
    NOISE_EXEPTION_LEVEL = 0.21 * N_spectr # —> Подавление гула. При увеличении коэф-та снижается влияние 0.31
       
    arr_to_all_hstgrm = []    
    for ix in range(N_spectr): #==================================================================  
        arr_i_max = cleaner.argrelextrema(cut_Spectr[:MAX_FREQ, ix], np.greater_equal, order = N_SMAL )[0]       
        arr_to_all_hstgrm.extend(arr_i_max)        
    #=============================================================================================
  
    hstgrm = cleaner.get_hstgrm_1D(arr_to_all_hstgrm, MAX_FREQ, prnt=False ) 

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