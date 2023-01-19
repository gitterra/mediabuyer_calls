# Печать текста с подсветкой
from IPython.display import display, Markdown 

# Словарь товаров
PRODUCTS_DICT = {'клареол': ['клареол', 'аквариу', 'ареола', 'вариола', 'кариу', 'кварил', 'клариус', 'клорел', 'кориол', 'крильона', 'лариол', 'ларион', 'триалога', 'флареон'],
            'велотренажёр': ['тренажёр', 'тренажер', 'велот'],
            'усилитель звука': ['усилит', 'чудо слу', 'слуховой аппарат', 'глух', 'звука ', 'ушной', 'поселите'],
            'браслет здоровье':['браслет', 'умный браслет'],
            'шиатцу подушка': ['подушк', 'шиат', 'подушечк', 'массажная', 'шиацу', 'шацу', 'массажжер'],
            'обогреватель': ['обогреват', 'быстро тепл', 'быстрое тепл' 'ру робот', 'теплорова', 'робус тепл', 'рубус тепл'],
            'термобельё': ['термобел', 'бельё', 'твоё тепло', 'твое тепло'],
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
            'Простатрикум': ['простатрикум']
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
        