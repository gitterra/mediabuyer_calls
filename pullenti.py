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
  for key in ['край','область','город','поселок','район','улица','дом','квартира']:
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
        if slot.key == 'CURRENCY':
          currency = slot.value
        if slot.key == 'VALUE':
          num = slot.value
      money_set.add(num + ' ' + currency.replace('RUB','руб')) ## Пока заменяю только рубли, вряд ли другая валюта найдется (но если найдется будет в оригинале)
  return money_set

# Функция визуализации цены
def show_me_your_money(money_set, color = 'Goldenrod'):
  for price in money_set:
    display(Markdown('<b><font color='+ color + '>' + price + '</font></b>'))