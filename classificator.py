import gdown 
# Загрузка моделей
from keras.models import load_model

import pickle
import numpy as np

# Загрузка модели Tinkoff c Google диска
fmodel = gdown.download('https://storage.yandexcloud.net/terratraineeship/22_MediaBuyer_2/models/model.h5', None, quiet=True)
fpickle = gdown.download('https://storage.yandexcloud.net/terratraineeship/22_MediaBuyer_2/models/tokenizer.pickle', None, quiet=True)

# Распаковка модели
model = load_model(fmodel)

# Распаковка токенайзера
with open(fpickle, 'rb') as f:
    tokenizer = pickle.load(f)

# Список классов моделей (порядок который у меня был при обучении)
CLASS_LIST = ['1_Order','2_Recall','3_Phone','4_Break','5_Prank','6_Auto']

# Подготовка и запуск предсказания объединенные в функцию
def predict(text):
  # Преобразую строку в последовательность индексов согласно частотному словарю с помощью токенайзера, формирую в виде разреженной матрицы (bag of words) и отправляю в НС
  y_pred = model.predict(np.array(tokenizer.sequences_to_matrix(tokenizer.texts_to_sequences([text]))), verbose='0')  
  res = {CLASS_LIST[i]: round(y_pred[i],4) for i in range(len(CLASS_LIST))}

  return res
