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
  for i in range(len(text)):
    if text[i] in ['0','1','2','3','4','5','6','7','8','9','0']:
      nums_in_text += text[i]
      nums_indexes.append(i)
  
  j = 0
  while (nums_in_text[j:j+len(num_str)] != num_str) and (j < len(text)):
    j += 1
  indexes = nums_indexes[j:j+len(num_str)]

  stext = ''
  for l in range(len(text)):
    if l in indexes:
      stext += '<b><font color=' + color + '>' + text[l] + '</font></b>'
    else: stext += text[l]
  
  return stext

# Функция печати текста с подсветкой заданных номеров
def printnums(text, num_lst, color = 'Aquamarine'):
  for i in range(len(num_lst)):
    text = marknum(text, num_lst[i], color)
  display(Markdown(text))