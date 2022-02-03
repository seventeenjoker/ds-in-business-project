# python-flask-docker
Итоговый проект курса "Машинное обучение в бизнесе"

Стек:

ML: sklearn, pandas, numpy
API: flask
Данные: с kaggle - https://www.kaggle.com/uciml/red-wine-quality-cortez-et-al-2009/version/2?select=winequality-red.csv

Задача: определить по входным характеристикам вина хорошего оно качаства или нет. Бинарная классификация

Используемые признаки:

1 - фиксированная кислотность, fixed acidity 

2 - летучая кислотность, volatile acidity

3 - лимонная кислота, citric acid

4 - остаточный сахар, residual sugar

5 - хлориды, chlorides

6 - свободный диоксид серы, free sulfur dioxide

7 - общий диоксид серы, total sulfur dioxide

8 - плотность, density

9 - pH, pH

10 - сульфаты, sulphates

11 - алкоголь, alcohol

Все признаки численные

Модель: RandomForestClassifier

### Клонируем репозиторий и создаем образ
```
$ git clone https://github.com/seventeenjoker/ds-in-business-project.git
$ cd GB_docker_flask_example
$ docker build -t seventeenjoker/gb_docker_flask_example .
```

### Запускаем контейнер

Здесь Вам нужно создать каталог локально и сохранить туда предобученную модель (<your_local_path_to_pretrained_models> нужно заменить на полный путь к этому каталогу)
```
$ docker run -d -p 8180:8180 -v <your_local_path_to_pretrained_models>:/app/app/models seventeenjoker/gb_docker_flask_example
```

### Переходим на localhost:8180
