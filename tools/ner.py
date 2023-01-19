import pandas as pd
# Печать текста с подсветкой
from IPython.display import display, Markdown 
# NER Natasha - функция распознавания
from collections import Counter
import librosa

import speech_recognition as sR
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

# Функция печати текста с ограничением по длине
def print_split_by_count(xtext,substrlen=100):
  for i in range(0, len(xtext), substrlen):
    print(xtext[i:i+substrlen])

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
def textes_for_NER(path_wav, en_starts, en_ends):
  NER_textes = []
  for i in range(len(en_starts)):
    glued_textes = []
    if len(en_starts[i]) != 0:
      all_textes = separec_Google(path_wav, en_starts[i], en_ends[i])
      for n in range(len(all_textes)):
        glued_textes.append(glue_text(all_textes[n]))
    NER_textes.append(glued_textes)
  return NER_textes
