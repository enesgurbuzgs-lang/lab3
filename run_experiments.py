"""Run from CMD: python run_experiments.py. Paths are relative to this file."""
import json
import platform
from pathlib import Path
from dataclasses import asdict
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from linear_models import (train_linear_regression, train_logistic_classifier,
                           compare_classical_models, classification_metrics)
from grader_contracts.linear_models import RegressionInput, ClassificationInput

ROOT = Path(__file__).resolve().parent


def prepare(frame, target, classification=False):
    x = frame.drop(columns=target)
    y = frame[target]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42,
        stratify=y if classification else None)
    scaler = StandardScaler()
    return (scaler.fit_transform(x_train), y_train.to_numpy(),
            scaler.transform(x_test), y_test.to_numpy())


def main():
    student = pd.read_csv(ROOT / 'Student_Performance.csv')
    student['Extracurricular Activities'] = student['Extracurricular Activities'].map({'Yes': 1, 'No': 0})
    credit = pd.read_csv(ROOT / 'german.csv', sep=';')
    regression_data = RegressionInput(*prepare(student, 'Performance Index'))
    classification_data = ClassificationInput(*prepare(credit, 'Creditability', True))
    regression = train_linear_regression(regression_data)
    own = train_logistic_classifier(classification_data)
    baselines = compare_classical_models(classification_data)
    results = {
        'linear_regression': {'test_mse': float(mean_squared_error(regression_data.test_target, regression.predictions))},
        'classification': {
            'own_logistic': asdict(classification_metrics(classification_data.test_target, own.probabilities, own.predictions)),
            **asdict(baselines),
        },
    }
    (ROOT / 'experiment_results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    experiments = []
    for rate, epochs in [(0.01, 200), (0.05, 800)]:
        model = train_linear_regression(RegressionInput(
            regression_data.train_features, regression_data.train_target,
            regression_data.test_features, regression_data.test_target, rate, epochs))
        experiments.append(f'| Linear | {rate} | {epochs} | train MSE = {model.loss_history[-1]:.6f} |')
    for rate, epochs in [(0.01, 200), (0.1, 1000)]:
        model = train_logistic_classifier(ClassificationInput(
            classification_data.train_features, classification_data.train_target,
            classification_data.test_features, classification_data.test_target, rate, epochs))
        experiments.append(f'| Logistic | {rate} | {epochs} | train BCE = {model.loss_history[-1]:.6f} |')
    table = '\n'.join('| ' + name + ' | ' + ' | '.join(f'{m[k]:.6f}' for k in ['roc_auc', 'accuracy', 'precision', 'recall', 'f1']) + ' |'
                      for name, m in results['classification'].items())
    comparisons = []
    for name, m in results['classification'].items():
        if name != 'own_logistic':
            delta = results['classification']['own_logistic']['roc_auc'] - m['roc_auc']
            comparisons.append(f'- ROC-AUC собственной модели относительно {name}: {delta:+.6f}.')
    report = f'''# Лабораторная №3: линейные модели

## 1. Данные и подготовка
Student_Performance.csv: {len(student)} строк, 5 признаков; цель Performance Index.
Extracurricular Activities: Yes=1, No=0. german.csv: {len(credit)} строк,
20 признаков; цель Creditability, положительный класс 1. Разделитель — точка с запятой.
Пропусков: student={int(student.isna().sum().sum())}, german={int(credit.isna().sum().sum())}.
Разбиение 80/20, random_state=42; для классификации stratify=y.
Размеры train/test: {len(regression_data.train_target)}/{len(regression_data.test_target)}
и {len(classification_data.train_target)}/{len(classification_data.test_target)}.
StandardScaler обучен только на train, затем применён к test. Цели не масштабируются.
Кодированные категории german используются как числовые признаки: это упрощение
навязывает порядок и расстояния категориям, особенно существенно для KNN.

## 2. Гипотезы и эксперименты
Гипотеза: масштабирование позволяет градиентному спуску устойчиво сходиться;
большее число эпох снижает обучающую ошибку. Ниже проверка параметров по train.
Тестовые метрики вычисляются для заранее выбранных параметров, подбор по test не проводится.

| Модель | learning_rate | epochs | Результат |
|---|---:|---:|---|
{chr(10).join(experiments)}

Линейная модель: y_hat=Xw+b; loss=mean((y_hat-y)^2),
grad_w=2 X.T@(y_hat-y)/n, grad_b=2 mean(y_hat-y).
Логистическая: p=sigmoid(Xw+b), loss=mean(logaddexp(0,z)-y*z),
grad_w=X.T@(p-y)/n, grad_b=mean(p-y). На каждой эпохе используется весь train.
Веса и смещение изначально нулевые. Истории содержат loss после каждого обновления.
Основной запуск: linear lr=0.05, epochs=800; logistic lr=0.1, epochs=1000,
без регуляризации; порог 0.5. Численно устойчивые sigmoid и BCE предотвращают переполнение.
Linear MSE: эпоха 1={regression.loss_history[0]:.6f}, последняя={regression.loss_history[-1]:.6f}.
Logistic BCE: эпоха 1={own.loss_history[0]:.6f}, последняя={own.loss_history[-1]:.6f}.

## 3. Метрики
Test MSE линейной регрессии: **{results['linear_regression']['test_mse']:.6f}**.
ROC-AUC вычисляется по вероятностям, остальные метрики по меткам, positive=1.

| Модель | ROC-AUC | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
{table}

Все классификаторы используют одно и то же разбиение и масштабирование.
Baselines: sklearn LogisticRegression(C=1, L2, max_iter=2000),
DecisionTreeClassifier(random_state=42, без ограничения глубины), KNN(k=5, евклидово расстояние).
Контракт BaselineComparisonResult содержит три готовые модели; собственная модель
записывается отдельно в JSON, чтобы выполнить и README, и стартовый контракт.

## 4. Выводы
{chr(10).join(comparisons)}
Собственная логистическая модель задаёт линейную границу. Дерево допускает нелинейные
правила, но без ограничения глубины может переобучаться; KNN чувствителен к метрике
расстояния и кодированию категорий. Это возможные объяснения различий, а не доказанные
причины: для проверки нужны кросс-валидация и дополнительные эксперименты.
Отличие от sklearn LogisticRegression ожидаемо из-за регуляризации и другого оптимизатора.
Единственное разбиение не позволяет утверждать универсальное превосходство модели.

## 5. Воспроизводимость
В CMD из папки lab3:

```cmd
python -m pip install -r requirements.txt
python run_experiments.py
```

Среда измерения: Python {platform.python_version()}, NumPy {np.__version__},
pandas {pd.__version__}, scikit-learn {sklearn.__version__}.
Точные версии зависимостей сохранены в requirements.txt.
Скрипт перезаписывает experiment_results.json и REPORT.md реальными результатами.
grader_contracts не изменяется. Истории ошибок сохранены в loss_history.json.
'''
    (ROOT / 'REPORT.md').write_text(report, encoding='utf-8')
    (ROOT / 'loss_history.json').write_text(json.dumps({'linear_mse': regression.loss_history, 'logistic_bce': own.loss_history}, indent=2), encoding='utf-8')
    (ROOT / 'requirements.txt').write_text(f'numpy=={np.__version__}\npandas=={pd.__version__}\nscikit-learn=={sklearn.__version__}\n', encoding='utf-8')
    assert np.isfinite(regression.predictions).all()
    assert np.isfinite(own.probabilities).all()
    assert all(np.diff(regression.loss_history) <= 1e-8)
    assert all(np.diff(own.loss_history) <= 1e-8)
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
