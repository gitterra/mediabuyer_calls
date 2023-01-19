# Загрузка и сохренение аудио файлов
import soundfile as sf
# Печать текста с подсветкой
from IPython.display import display, Markdown 
# Модуль для проигрывания аудио в colab
import IPython.display as ipd 

# Словарь сущностей с соответствующими триггерами
ENTITIES_DICT = {'ФИО': [' имя ', ' фамилия ', ' отчество ', ' зовут ', ' обращаться '],
                 'ТЕЛЕФОН': [' телефон', ' мобильн', ' номер '],
                 'АДРЕС': [' обращаетесь ', ' обращайтесь ', ' индекс ', ' адрес ', ' край ', ' район ', ' област', ' город', ' село ', ' деревня ', ' посёлок ', ' улица ', ' переулок ', ' проспект ', 'аллея ', ' проезд ', ' дом ', ' строение ', ' квартира '],
                 'ТОВАР': [' товар ', ' продукт ', ' заказ ', ' заказать ', ' купить ', ' отправ', ' вышлите ' , ' присла'],
                 'ЦЕНА': [' цена ', ' стоимость ', ' стоит ', ' скидка ', ' руб ', ' руб. ', ' рублей ', ' рубля ']
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

# Вспомогательная функция ограничения ширины печатаемого текста (возможно есть штатные средства?)
def print_lim(text, n_sym = 100):
  for i in range(0, len(text), n_sym):
    print(text[i:i+n_sym])

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
    print('\n'+'-.'*50+'-\n')
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

