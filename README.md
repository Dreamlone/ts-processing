# ts-processing

Python code, visualizations, and other materials related to time series processing

## rus

Данный репозиторий это сборник материалов по теме "Автоматическое машинное обучение для обработки временных рядов"

**Основная задумка:** про AutoML трудно рассказывать в отрыве от атомарных компонентов (предобработок и моделей), 
на которых строятся методы и (open-source) фреймворки. Поэтому в данном репозитории нарратив строится вокруг последовательного 
усложнения "взгляда" на концепцию временных рядов. Затем рассматриваются предобработки безотносительно решаемой задачи. 
И уже после этого с постепенным усложнением обсуждаются модели для задачи (например, прогнозирования). 

Ядро для повествования представляет собой набор jupyter notebooks: 

- `/notebooks/1_time_series_exploration.ipynb` - обсуждение трех взглядов на временные ряды. 
  Рассматриваются свойства временных рядов, стационарность / нестационарность, методы визуализации 
- `/notebooks/2_ts_preprocessing.ipynb` - методы предобработки
- `/notebooks/3_ts_forecasting.ipynb` - прогнозирование: AR, MA, ARMA, ARIMA, SARIMA, pmdarima, AutoTS, FEDOT

**Имейте в виду, что интерактивные ipywidgets элементы в тетрадках при просмотре через браузер не рендерятся.**
**Чтобы их увидеть, запустите локально следуя инструкциям ниже**

Чтобы запустить тетрадки: 

1. Склонируйте репозиторий
2. Убедитесь что у вас установлен Python 3.10 и poetry 
3. Команда в терминале из корня репозитрия `poetry install` 
4. Команда в терминале из корня репозитрия `jupyter lab` 

### Список литературы

1. Бурнашев Р. А., Арабов М. К., Миссаров М. Д. — Анализ и прогнозирование временных рядов: путь от классики к современным решениям
2. Эйлин Нильсен — Практический анализ временных рядов. Прогнозирование со статистикой и машинным обучением
3. medium: [How to Find Seasonality Patterns in Time Series](https://medium.com/data-science/how-to-find-seasonality-patterns-in-time-series-c3b9f11e89c6) (member-only story)
4. medium: [How can we quantify similarity between time series?](https://medium.com/gorillatech/how-can-we-quantify-similarity-between-time-series-ed1d0b633ca0)
5. medium: [From Default Python Line Chart to Journal-Quality Infographics](https://medium.com/data-science/from-default-python-line-chart-to-journal-quality-infographics-80e3949eacc3)
6. medium: [Real-Time Time Series Anomaly Detection](https://medium.com/data-science/real-time-time-series-anomaly-detection-981cf1e1ca13)

## eng

TODO: add after finalizing materials 

## Literature

TODO: add after finalizing materials 