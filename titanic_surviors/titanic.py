import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import pandas as pd
import os

current_version = 4


def read_df(filename):
    dir_name = os.path.dirname(os.path.abspath(__file__))
    full_file_name = os.path.join(dir_name, filename)
    df = pd.read_csv(full_file_name)
    print("# "*10, "file {} read. Shape after reading - {} ".format(full_file_name, df.shape), "#"*10)
    return df


def write_df(df, filename, write_index=False, append_date_time=True):
    import datetime
    dir_name = os.path.dirname(os.path.abspath(__file__))
    full_file_name = os.path.join(dir_name, filename)
    if append_date_time:
        f_name, extn = full_file_name.split(".")
        full_file_name = f_name + "_" + \
            datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S") + "." + extn

    print("\nwriting df to {}\n\n".format(full_file_name))
    df.to_csv(full_file_name, index=write_index)


def process_age(df, new_column_name):
    cut_points = [-1, 0, 5, 12, 18, 35, 60, 100]
    label_names = ['Missing', 'Infant', 'Child',
                   'Teenager', 'Young Adult', 'Adult', 'Senior']
    df["Age"] = df["Age"].fillna(-0.5)
    df[new_column_name] = pd.cut(df["Age"], cut_points, labels=label_names)

    return df


def process_fare(df, new_column_name):
    cut_points = [0, 12, 50, 100, 1000]
    label_names = ['0-12', '12-50', '50-100', '100+']
    df["Fare"] = df["Fare"].fillna(0)
    df[new_column_name] = pd.cut(df["Fare"], cut_points, labels=label_names)
    return df


def process_name(df, new_column_name):
    titles = {
        "Mr":         "Mr",
        "Mme":         "Mrs",
        "Ms":          "Mrs",
        "Mrs":        "Mrs",
        "Master":     "Master",
        "Mlle":        "Miss",
        "Miss":       "Miss",
        "Capt":        "Officer",
        "Col":         "Officer",
        "Major":       "Officer",
        "Dr":          "Officer",
        "Rev":         "Officer",
        "Jonkheer":    "Royalty",
        "Don":         "Royalty",
        "Sir":        "Royalty",
        "Countess":    "Royalty",
        "Dona":        "Royalty",
        "Lady":       "Royalty"
    }
    extracted_titles = df["Name"].str.extract(r' ([A-Za-z]+)\.', expand=False)
    df[new_column_name] = extracted_titles.map(titles)
    return df


def process_cabin(df, new_column_name):
    df[new_column_name] = df["Cabin"].str[0]
    df[new_column_name] = df["Cabin_type"].fillna("Unknown")
    return df


def process_columns(train, test, version=current_version):
    from sklearn.preprocessing import minmax_scale
    # process age columns
    train = process_age(train, "Age_categories")
    test = process_age(test, "Age_categories")
    dummy_candidates = ["Age_categories", "Pclass", "Sex"]
    train, test = create_dummy_columns(train, test, dummy_candidates)

    # columns = ['SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked']
    # print(train[columns].describe(include='all', percentiles=[]))

    test["Fare"] = test["Fare"].fillna(train["Fare"].mean())
    train["Fare"] = train["Fare"].fillna(train["Fare"].mean())

    test["Embarked"] = test["Embarked"].fillna("S")
    train["Embarked"] = train["Embarked"].fillna("S")

    train = create_dummies(train, "Embarked")
    test = create_dummies(test, "Embarked")

    rescale_cols = ["SibSp", "Parch", "Fare"]
    for column in rescale_cols:
        new_col_name = column+"_scaled"
        train[new_col_name] = minmax_scale(train[column])
        test[new_col_name] = minmax_scale(test[column])

    # V2 - in version two we also process fare column
    if version > 1:
        train = process_fare(train, "Fare_categories")
        test = process_fare(test, "Fare_categories")

        train = process_name(train, "Title")
        test = process_name(test, "Title")

        train = process_cabin(train, "Cabin_type")
        test = process_cabin(test, "Cabin_type")

        dummy_candidates = ["Fare_categories", "Title", "Cabin_type"]
        train, test = create_dummy_columns(train, test, dummy_candidates)

    # print("Final columns ... \n{}".format(train.columns))
    return train, test


def create_dummies(df, column_name):
    dummies = pd.get_dummies(df[column_name], prefix=column_name)
    df = pd.concat([df, dummies], axis=1)
    return df


def create_dummy_columns(train, test, dummy_candidates):
    for column in dummy_candidates:
        train = create_dummies(train, column)
        test = create_dummies(test, column)
    return train, test


def get_features_v1():
    features = ['Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male',
                'Age_categories_Missing', 'Age_categories_Infant',
                'Age_categories_Child', 'Age_categories_Teenager',
                'Age_categories_Young Adult', 'Age_categories_Adult',
                'Age_categories_Senior']
    return features


def get_features_v2(columns):
    no_impact_columns = ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
                         'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked', 'Age_categories']
    features = remove_columns(columns, no_impact_columns)
    return features


def remove_columns(columns, columns_to_remove):
    new_columns = [
        col for col in columns if col not in columns_to_remove]
    return new_columns


def get_features_v3(train, target, optimized = False, model = LogisticRegression()):
    # V2 - after feature engineering
    features = get_features_v2(train.columns)
    no_impact_columns = ["Fare_categories", "Title", "Cabin_type"]
    features = remove_columns(features, no_impact_columns)
    if optimized:
        if model == None:
            model = LogisticRegression()
        model.fit(train[features], train[target])
        coefficients = model.coef_
        feature_coeff = pd.Series(coefficients[0], index=features)
        # print(feature_coeff, "\n")
        ordered_feature_coeff = feature_coeff.abs().sort_values(ascending=False)
        # print(ordered_feature_coeff, "\n")
        top_features = ordered_feature_coeff[ordered_feature_coeff > 0.4]
        features = top_features.index
    return features


def get_features_v4(train, target, optimized = False, model = LogisticRegression()):
    from sklearn.feature_selection import RFECV
    features = get_features_v2(train.columns)
    no_impact_columns = ["Fare_categories", "Title", "Cabin_type", 'Fare_scaled', "Sex_male", "Sex_female",
                         "Pclass_2", "Age_categories_Teenager", "Fare_categories_12-50", "Title_Master", "Cabin_type_A"]
    features = remove_columns(features, no_impact_columns)
    # print("version 4 - Columns ...\n", features)
    if optimized:
        if model == None:
            model = LogisticRegression()
        selector = RFECV(model, cv=10)
        selector.fit(train[features], train[target])
        features = train[features].columns[selector.support_]
    print("version 4 - optimized features ...\n", features)
    return features


def get_features_and_target(train, version=current_version, optimized = True, model = LogisticRegression()):
    """
    provide features based on version.

    args
    model (sklearn) - sklearn model to fit
    train (dataframe) - dataframe containing training data
    version -
        version = "1" = provides columns with selection
        version = "2" = provides columns after manually rmeoving columns which has no impact
        version = "3" = upgrade to v2 where we check the coeff and provide only top 8 columns
        default value = None - Provides the most recent vesion of features

    return - features and target
    """
    target = "Survived"
    features = None
    if version == 1:
        features = get_features_v1()
    elif version == 2:
        features = get_features_v2(train.column)
    elif version == 3:
        features = get_features_v3(train, target, model = model, optimized = optimized)
    elif version == current_version:
        features = get_features_v4(train, target, model = model, optimized = optimized)

    return features, target


def lr_model_and_validate(train, test, features, target):
    from sklearn.model_selection import cross_val_score
    import numpy as np

    print("LogisticRegression - cross validating with training data\n")
    lr = LogisticRegression()

    print("LogisticRegression - features ... \n", features, "\n")
    print("LogisticRegression - target ... \n", target, "\n")
    scores = cross_val_score(lr, train[features], train[target], cv=10)
    accuracy = np.mean(scores)
    print("LogisticRegression - accuracy scores - ", scores, "\n")
    print("LogisticRegression - mean accuracy ", accuracy, "\n")

    print("LogisticRegression - predicting with test data\n")
    lr = LogisticRegression(max_iter=500)
    lr.fit(train[features], train[target])
    predictions = lr.predict(test[features])
    return predictions, scores, accuracy


def create_submission_file(filename, predictions, test):
    test_ids = test["PassengerId"]
    submission = {"PassengerId": test_ids,
                  "Survived": predictions}
    submission_df = pd.DataFrame(submission)
    write_df(submission_df, filename)


def get_features_and_target_v2(train, test):
    not_needed_training_columns = ['Fare_categories', 'Cabin_type', 'SibSp', 'Name', 'Pclass', 'Ticket', 'Embarked', 'Parch', 'Age_categories', 'Fare', 'Title', 'Sex', 'Age', 'Cabin']
    not_needed_test_columns = ['Survived', 'Fare_categories', 'Cabin_type', 'SibSp', 'Name', 'Pclass', 'Ticket', 'Embarked', 'Parch', 'Age_categories', 'Fare', 'Title', 'Sex', 'Age', 'Cabin']
    training_columns = remove_columns(train.columns, not_needed_training_columns)
    test_columns = remove_columns(test.columns, not_needed_test_columns)
    target = "Survived"
    return training_columns, test_columns, target


def model_lr(train = None, test = None, features = None, target = None, restart=True):
    if restart:
        test = read_df("test.csv")
        train = read_df("train.csv")
        train, test = process_columns(train, test)
    
    if(features == None):
        features, target = get_features_and_target(train, optimized = True, model = LogisticRegression(), version=current_version )

    # model LogisticRegression
    predictions, scores, accuracy = lr_model_and_validate(
        train, test, features, target)
    create_submission_file("submission_lr.csv", predictions, test)
    return predictions, scores, accuracy

def get_fresh_data():
    test_original = read_df("test.csv")
    train_original = read_df("train.csv")
    train_original, test_original = process_columns(train_original, test_original)
    training_features, test_features, target = get_features_and_target_v2(train_original, test_original)
    train = train_original[training_features]
    holdout = test_original[test_features]
    all_X = train.drop(['Survived','PassengerId', 'Cabin_type_T'],axis=1)
    all_y = train['Survived']
    return all_X, all_y, holdout


def model_knn(all_X, all_y, holdout):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import GridSearchCV

    #first lets get a baseline using LogisticRegression
    lr = LogisticRegression()
    scores = cross_val_score(lr, all_X, all_y, cv=10)
    accuracy_lr = scores.mean()
    print("LogisticRegression - - Base accuracy = {}".format(accuracy_lr))

    #knn with k score 19 (found with a loop)
    knn = KNeighborsClassifier(n_neighbors=19)
    scores = cross_val_score(knn, all_X, all_y, cv=10)
    accuracy_knn = scores.mean()
    print("KNeighborsClassifier - -  accuracy = {}".format(accuracy_knn))

    #lets use gridsearch 
    hyperparameters = {
    "n_neighbors": range(1,20,2),
    "weights": ["distance", "uniform"],
    "algorithm": ['brute'],
    "p": [1,2]
    }

    knn = KNeighborsClassifier()
    grid = GridSearchCV(knn,param_grid=hyperparameters,cv=10)
    grid.fit(all_X, all_y)
    best_knn = grid.best_estimator_
    print("KNeighborsClassifier with GridSearchCV - \nbest params = {}\nbest score {}".format(grid.best_params_, grid.best_score_))
   
    #remove id from test
    holdout_no_id = holdout.drop(['PassengerId'],axis=1)
    best_knn = grid.best_estimator_
    #predict on our best knn model
    holdout_predictions = best_knn.predict(holdout_no_id)
    #prepare file for kaggle subission 
    create_submission_file("submission_knn.csv", holdout_predictions, holdout)
    return grid.best_score_

   
def model_rf(ll_X, all_y, holdout):   
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV

    hyperparameters = {"criterion": ["entropy", "gini"],
                   "max_depth": [5, 10],
                   "max_features": ["log2", "sqrt"],
                   "min_samples_leaf": [1, 5],
                   "min_samples_split": [3, 5],
                   "n_estimators": [6, 9]
    }

    clf = RandomForestClassifier(random_state=1)
    grid = GridSearchCV(clf,param_grid=hyperparameters,cv=10)

    grid.fit(all_X, all_y)
    print("RandomForestClassifier with GridSearchCV - \nbest params = {}\nbest score {}".format(grid.best_params_, grid.best_score_))
    best_rf = grid.best_estimator_
    holdout_no_id = holdout.drop(['PassengerId'],axis=1)
    #predict on our best knn model
    holdout_predictions = best_rf.predict(holdout_no_id)
    #prepare file for kaggle subission 
    create_submission_file("submission_rf.csv", holdout_predictions, holdout)
    return grid.best_score_


if __name__ == '__main__':
    
    # predictions_lr, scores_lr, accuracy_lr =  model_lr() 
    # print("LogisticRegression - accuracy = {}".format(accuracy_lr))
    all_X, all_y, holdout = get_fresh_data()
    accuracy_kn = model_knn(all_X, all_y, holdout)
    print("KNeighborsClassifier using GridSearchCV - accuracy = {}".format(accuracy_kn))
    accuracy_rf = model_rf(all_X, all_y, holdout)
    print("RandomForestClassifier  with GridSearchCV- accuracy = {}".format(accuracy_rf))
    
