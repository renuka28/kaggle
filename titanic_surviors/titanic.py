import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import pandas as pd
import os


def read_df(filename):
    dir_name = os.path.dirname(os.path.abspath(__file__))
    full_file_name = os.path.join(dir_name, filename)
    df = pd.read_csv(full_file_name)
    print("# "*10, "file {} read. Shape after reading - {} ".format(full_file_name, df.shape), "#"*10)
    return df


def write_df(df, filename, write_index=False):
    dir_name = os.path.dirname(os.path.abspath(__file__))
    full_file_name = os.path.join(dir_name, filename)
    print("\nwriting to {}\n\n".format(full_file_name))
    df.to_csv(full_file_name, index=write_index)


def process_age(df, cut_points, label_names):
    df["Age"] = df["Age"].fillna(-0.5)
    df["Age_categories"] = pd.cut(df["Age"], cut_points, labels=label_names)
    return df


def update_age_column(train, test):
    cut_points = [-1, 0, 5, 12, 18, 35, 60, 100]
    label_names = ['Missing', 'Infant', 'Child',
                   'Teenager', 'Young Adult', 'Adult', 'Senior']

    train = process_age(train, cut_points, label_names)
    test = process_age(test, cut_points, label_names)

    return train, test


def create_dummies(df, column_name):
    dummies = pd.get_dummies(df[column_name], prefix=column_name)
    df = pd.concat([df, dummies], axis=1)
    return df


def create_dummy_columns(train, test):
    train = create_dummies(train, "Pclass")
    test = create_dummies(test, "Pclass")
    train = create_dummies(train, "Sex")
    test = create_dummies(test, "Sex")
    train = create_dummies(train, "Age_categories")
    test = create_dummies(test, "Age_categories")
    return train, test


def lr_model_and_validate(train, test):
    from sklearn.model_selection import cross_val_score
    import numpy as np
    features = ['Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male',
                'Age_categories_Missing', 'Age_categories_Infant',
                'Age_categories_Child', 'Age_categories_Teenager',
                'Age_categories_Young Adult', 'Age_categories_Adult',
                'Age_categories_Senior']

    target = "Survived"
    lr = LogisticRegression()
    lr.fit(train[features], train[target])
    predictions = lr.predict(test[features])
    return predictions


def create_submission_file(filename, predictions, test):
    test_ids = test["PassengerId"]
    submission = {"PassengerId": test_ids,
                  "Survived": predictions}
    submission_df = pd.DataFrame(submission)
    write_df(submission_df, filename)


if __name__ == '__main__':
    test = read_df("test.csv")
    train = read_df("train.csv")
    train, test = update_age_column(train, test)
    train, test = create_dummy_columns(train, test)
    predictions = lr_model_and_validate(train, test)
    create_submission_file("submission_lr.csv", predictions, test)
