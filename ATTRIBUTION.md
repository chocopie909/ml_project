# Источники и авторство

Темы проекта вдохновлены материалами «Машинное обучение, ФКН ВШЭ», найденными в [perfectteatimer/practical-hw / ML-HSE](https://github.com/perfectteatimer/practical-hw/tree/b4ba2232c1b4b5258a10d3f50a0fba94451d0677/ML-HSE).

Первая сборка содержала копии пяти ноутбуков и неполные исходные реализации. Рабочая версия переработана: новые ноутбуки, минимальные реализации алгоритмов, протокол экспериментов, тесты и документация подготовлены с помощью Codex. Чужие тексты заданий, ответы, сохранённые результаты и исходные модули в рабочую поставку не включены. Это не выполнение всех первоначальных заданий один в один и не заявление о самостоятельном авторстве всех материалов владельцем GitHub.

Реализованные алгоритмы — стандартные GD, SGD, Momentum, Adam, бинарный CART и logistic gradient boosting. Библиотечные baseline, калибровка и слабые деревья для boosting предоставлены scikit-learn.

## Данные

- [Diabetes](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html): 442 объекта, 10 признаков. Загружается в исходном масштабе через `scaled=False`.
- [Wisconsin Diagnostic Breast Cancer](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html): 569 объектов, 30 признаков. В проекте метки перекодированы: malignant = 1, benign = 0.
- [Первичный источник WDBC, UCI](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic).

Наборы загружаются из установленного пакета scikit-learn. Их авторство и условия использования остаются у исходных поставщиков; проект не объявляет их собственными данными.
