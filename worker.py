# Печать текста с подсветкой
from IPython.display import display, Markdown 
# Загрузка и сохренение аудио файлов
import soundfile as sf
# Модуль для проигрывания аудио в colab
import IPython.display as ipd 

# Фильтрация предупреждений, чтобы не librosa.load не ругалась на mp3
import warnings
warnings.filterwarnings('ignore')

# Пакет для анализа аудио
import librosa
import cleaner
import cutter 
import recognizer
import ner
import phone
import pullenti
import classificator
import trigger
import dicter

# Вспомогательная функция ограничения ширины печатаемого текста (возможно есть штатные средства?)
def print_lim(text, n_sym = 100):
  for i in range(0, len(text), n_sym):
    print(text[i:i+n_sym])

def Total_Visualization(audio_path):
  # Вывод на прослушивание оригинального файла
  display(Markdown('<br><b>Прослушать оригинальный файл:</b>'))
  ipd.display(ipd.Audio(audio_path))

  # Удаление музыкальных фрагментов и автоответчика
  Signal, sr = librosa.load(audio_path)
  Cleared_Signal = cleaner.cut_all_music(Signal)

  # Проверка, что осталось что-либо для обработки
  if len(Cleared_Signal) == 0:
    print('Файл не содержит звуковых данных для анализа. Наиболее вероятный класс:', CLASS_LIST[5])
  else:
    # Создание wav файла из обработанного сигнала (возможно это костыль, но не wav вроде SpechRecognition не отдать)
    path_wav = '/content/temporary.wav'
    sf.write('temporary.wav', Cleared_Signal, sr)

    display(Markdown('<br><b>Прослушать файл после удаления музыки:</b>'))
    ipd.display(ipd.Audio(path_wav))

    # Определение интервалов для фраз и передача фраз на распознавание 
    List_of_Timing = cutter.phrase_by_phrase(Cleared_Signal)
    textes, starts, durations = recognizer.separec_Google_timed(path_wav, List_of_Timing)

    # Проверка на наличие распознанных текстов
    if len(textes) == 0:
      print('В файле не обнаружены речевые данные для анализа. Наиболее вероятный класс:', CLASS_LIST[4])
    else:
      # Склейка текста для классификации
      text = recognizer.glue_uncrossed_text(textes)
      print('\n'+'-.'*50+'-\n')
      display(Markdown('<b>Передача распознаного текста без предварительной обработки в NER для поиска ФИО дает следующий результат:</b>'))
      # Передача полного текста для распознавания имени в NER 
      result_names = ner.get_person_from_ner(textes, text)
      # Визуализация результата - печать текста с выделенными именами оператора и фио
      ner.printmd(text,result_names)
      print('\n'+'-.'*50+'-\n')
      display(Markdown('<b>Поиск цифрового блока без предварительной обработки дает следующий результат:</b>'))
      result_nums = phone.find_num(phone.num_space_remover(phone.nums_replacer(text)), phone.MIN_PHONE_DIGITS) # Перед поиском блоков цифр, заменяю все виды текстовых цифр и склеиваю в блоки
      if len(result_nums) == 0:
        print('Последовательность цифр заданной длины не обнаружена.')
      else:
        phone.printnums(phone.nums_replacer(text), result_nums) # Чтобы верно подсветить нужно заменить все виды цифр, а вот склеивать для наглядности уже не надо

      # Запуск PullEnty для обнаружение прочих сущностей
      print('\n'+'-.'*50+'-\n')
      display(Markdown('<b>PullEnty может найти следующие сущности.</b>'))  
      result_OTH = pullenti.processor_other(text)
      display(result_OTH.graph)

      # Запуск предсказания сети
      cls, ver = classificator.give_predict(text)

      # Отображение результата предсказания НС
      print('-.'*50+'-')
      message = '<b><br>Сеть предсказала класс "'+ cls + '" с вероятностью '+ str(ver) + ' %</b>'
      display(Markdown(message))

      # Этот цикл позволит запускать поиск сущностей только в случае подходящих нам классов, но пока мы не уверены в точности классификатора, он только для информации. 
      if cls in classificator.CLASS_LIST[:3]:
        print('В данном классе можно искать сущности.')
      else:
        print('Если файл классифицирован верно, то обнаружение сущностей является маловероятным.')

      # Поиск триггеров и соответствующих им наименований сущностей
      scanned_triggers, scanned_entities, tr_starts, tr_durations = trigger.entrigger_scanner(textes, starts, durations)

      # Выделение отдельных сущностей и соответствующих им временных интервалов
      if len(tr_starts) == 0:
        print('\nТриггеры сущностей не обнаружены!')
      else:
        # Сортировка фрагментов по каждой сущности
        sortr_starts, sortr_durations = trigger.entities_locator(scanned_entities, tr_starts, tr_durations)
        # Вывод фрагментов аудиофайла с найденными сущностями
        trigger.visualize_fragments(path_wav,  sortr_starts, sortr_durations, textes, starts)

        # Выделение интервалов фрагментов сущностей на основании заданных границ
        en_starts, en_ends = ner.entities_fragments(sortr_starts, sortr_durations)
        
        # Список текстов для каждой сущности
        NER_textes = ner.textes_for_NER(path_wav, en_starts, en_ends)

        # Отображение всех текстов для каждой сущности
        print('-'+'.-'*50)
        for entity in trigger.ENTITIES_UNIC:
          message = '<b><br>Тексты для поиска сущности '+ entity + ':</b>'
          index = trigger.ENTITIES_UNIC.index(entity)
          display(Markdown(message))
          if len(NER_textes[index]) != 0:
            for n in range(len(NER_textes[index])):
              print('Текст №',n,sep='')
              print_lim(NER_textes[index][n])
          else: print('Отсутствуют.')

        # Обработка текстов найденных для ФИО
        index = trigger.ENTITIES_UNIC.index('ФИО')
        if len(NER_textes[index]) != 0:
          print('\n' + '-.'*50+'-')
          display(Markdown('<b><br>Обработка текстов приготовленных для поиска сущности ФИО:</b>'))
          for n in range(len(NER_textes[index])):
            print('Текст №',n,sep='')
            ner.printmd(NER_textes[index][n], ner.get_person_from_ner([], NER_textes[index][n]))
        
        # Обработка текстов найденных для ТЕЛЕФОН
        index = trigger.ENTITIES_UNIC.index('ТЕЛЕФОН')
        if len(NER_textes[index]) != 0:
          print('-.'*50+'-')
          display(Markdown('<b><br>Обработка текстов приготовленных для поиска сущности ТЕЛЕФОН:</b>'))
          for n in range(len(NER_textes[index])):
            print('Текст №',n,sep='')
            result_nums = phone.find_num(phone.num_space_remover(phone.nums_replacer(NER_textes[index][n])), phone.MIN_PHONE_DIGITS) # В тексте от Google нужно по идее только удалить пробелы, но на всякий случай обработаю также текстовые цифры
            if len(result_nums) == 0:
              print('Последовательность цифр заданной длины не обнаружена.')
            else:
              phone.printnums(phone.nums_replacer(NER_textes[index][n]), result_nums, color = 'Maroon')

        # Обработка текстов найденных для АДРЕС
        index = trigger.ENTITIES_UNIC.index('АДРЕС')
        if len(NER_textes[index]) != 0:
          print('-.'*50+'-')
          display(Markdown('<b><br>Обработка текстов приготовленных для поиска сущности АДРЕС:</b>'))
          for n in range(len(NER_textes[index])):
            print('Текст №',n,sep='')
            result_ADR = pullenti.processor_address(NER_textes[index][n])
            adr_dict = pullenti.extract_address(result_ADR, {})
            pullenti.print_address(adr_dict, 'Indigo')

        # Обработка текстов найденных для ТОВАР
        index = trigger.ENTITIES_UNIC.index('ТОВАР')
        if len(NER_textes[index]) != 0:
          print('\n' + '-.'*50+'-')
          display(Markdown('<b><br>Обработка текстов приготовленных для поиска сущности ТОВАР:</b>'))
          for n in range(len(NER_textes[index])):
            print('Текст №',n,sep='')
            finded_prods_dict = dicter.main_product_search(NER_textes[index][n], dicter.PRODUCTS_DICT)
            if len(finded_prods_dict) != 0:
              prods_list = dicter.extract_word_for_marking(finded_prods_dict)
              dicter.printmarkedwords(NER_textes[index][n], prods_list, 'Olive')
            else: print('Товары в этом фрагменте не обнаружены.')

        # Обработка текстов найденных для ЦЕНА
        index = trigger.ENTITIES_UNIC.index('ЦЕНА')
        if len(NER_textes[index]) != 0:
          print('-.'*50+'-')
          display(Markdown('<b><br>Обработка текстов приготовленных для поиска сущности ЦЕНА:</b>'))
          for n in range(len(NER_textes[index])):
            print('Текст №',n,sep='')
            result_PE = pullenti.processor_price(NER_textes[index][n])
            pullenti.show_me_your_money(pullenti.extract_money(result_PE, set()), 'Goldenrod')