import warnings
import pandas as pd
import graphviz
from sklearn.metrics import classification_report, balanced_accuracy_score, accuracy_score
from sklearn.svm import SVC
from preprocess.preprocess import load_train_valid_matrix
from run_fc.eval_model import metrics_success
from run_fc.metadata import Metadata
import matplotlib.pyplot as plt
import numpy as np
from sklearn import tree
import graphviz

warnings.filterwarnings('ignore')


def train_svm(x_train, y_train, x_valid, y_valid, metadata):
    clf = SVC(gamma='auto', kernel='linear')
    clf.fit(x_train, y_train)
    eval_read_maid_models(clf, x_valid, y_valid, 'svm', x_train, y_train, metadata)


def train_tree(x_train, y_train, x_valid, y_valid, metadata):
    clf = tree.DecisionTreeClassifier(criterion='entropy')
    clf = clf.fit(x_train, y_train)

    dot_data = tree.export_graphviz(clf, out_file=None,
                                    # feature_names=['tp1', '2', '3', '4'],
                                    class_names=['common', 'rare_p300'],
                                    filled=True, rounded=True,
                                    special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.render(None, view=True)
    eval_read_maid_models(clf, x_valid, y_valid, 'tree', x_train, y_train, metadata)
    return clf.tree_.feature, clf.tree_.threshold, clf.tree_.value, clf


def train_all_models(data_dit, mean_substract, metadata):
    x_train, y_train, x_valid, y_valid = load_train_valid_matrix(data_dit, None, mean_substract)
    x_train_before, x_valid_before = x_train.reshape(x_train.shape[0], -1), x_valid.reshape(x_valid.shape[0], -1)
    x_train_one_col = x_train[:, 5, :]
    x_valid_one_col = x_valid[:, 5, :]
    # print("compare before after:",x_train_before[0][0][0]==x_train[0][0])
    # print("compare before after:",x_train_before[0][0][1]==x_train[0][1])
    # print("compare before after:",x_train_before[0][1][0]==x_train[0][217])

    print("result on one columns")
    train_tree(x_train_one_col, y_train, x_valid_one_col, y_valid, metadata)
    # print("result on all columns")
    # tree_feature, tree_threasholds, tree_values, clf = train_tree(x_train_before, y_train, x_valid_before, y_valid,
    #                                                               metadata)
    # thresh = tree_threasholds[0]
    # index = tree_feature[0]
    # eval_tree_condition(index, thresh, x_valid_before, y_train, y_valid)
    # eval_tree_condition(index, thresh, x_valid_one_col, y_train, y_valid)


def eval_tree_condition(index, thresh, x_valid_before, y_train, y_valid):
    df = pd.concat([pd.DataFrame(x_valid_before), pd.DataFrame(y_train, columns=['tag'])], axis=1, join='inner')
    df_2 = df[[index, 'tag']]
    shiran_pred = (df_2[index] > thresh).astype(int)
    shiran_correct = np.sum(shiran_pred == y_valid)
    total = len(df_2)
    print('correct tagging: {} / {} {}'.format(shiran_correct, total, shiran_correct * 1.0 / total))


def eval_read_maid_models(clf, x_valid, y_valid, model_name, x_train, y_train, metadata):
    eval_read_inner(clf, x_train, y_train, model_name + '_train', metadata)
    eval_read_inner(clf, x_valid, y_valid, model_name + '_test', metadata)


def eval_read_inner(clf, x, y, model_name, metadata):
    # import inspect
    # source = inspect.getsource(clf.tree_.predict)
    pred = clf.predict(x)
    metrics_success(model_name, y, pred, metadata)


# data_dir = '/Users/shiran.s/dev/p300_net/output/processed_data/small'
data_dir = '/Users/shiran.s/dev/p300_net/output/processed_data/processed_latest/'
train_all_models(data_dir, False, Metadata(None))


def run_iris_example():
    from sklearn.tree import export_graphviz
    import pandas as pd
    from sklearn.datasets import load_iris
    from sklearn.tree import DecisionTreeClassifier

    # Load data and store it into pandas DataFrame objects
    iris = load_iris()
    X = pd.DataFrame(iris.data[:, :], columns=iris.feature_names[:])
    y = pd.DataFrame(iris.target, columns=["Species"])

    # Defining and fitting a DecisionTreeClassifier instance
    tree = DecisionTreeClassifier()
    tree.fit(X, y)
    dot_data = export_graphviz(
        tree,
        out_file=None,
        feature_names=list(X.columns),
        class_names=iris.target_names,
        filled=True,
        rounded=True)
    graph = graphviz.Source(dot_data)
    graph.render("file_1", view=True)
