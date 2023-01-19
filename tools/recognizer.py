import speech_recognition as sR

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
          text_part = ''  # Если ошибка распознавания то обнуляю текс
        
        textes.append(text_part)
        starts.append(start)
        durations.append(duration)

  return textes, starts, durations

# Функция для склейки списка непересекающихся строк (для подачи на классификацию)
def glue_uncrossed_text(textes):
  result = textes[0]
  for i in range(1,len(textes)):
    result += ' ' + textes[i]
    result.replace('  ',' ')
  return result