import numpy as np
import pandas as pd
from math import e, log
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

from typing import Tuple, Dict, Union


class Attribute:
    def __init__(self, name, values):
        self.name = name
        self.possible_values = set(values)

    def add_possible_value(self, value):
        self.possible_values.add(value)

    def __repr__(self):
        return self.name

class Node:
    def __init__(self, attribute, index):
        self.child_nodes = {}
        self.attribute = attribute
        self.index = index
        self.node_name = "node"


class Tree:
    def __init__(self):
        self.model = None

    @staticmethod
    def calculate_entropy(classes: np.ndarray, base=e) -> float:
        number_of_examples = classes.size
        frequency_vector = Tree.calculate_class_frequency(classes)
        prob_vector = frequency_vector / number_of_examples
        entropy = 0
        for prob in prob_vector:
            entropy -= prob * log(prob, base)
        return entropy

    # TODO how to return examples for best attributes etc.
    @staticmethod
    def calculation_gain(
        entropy_for_whole_set: float,
        attribute: Attribute,
        attr_idx: int,
        examples: np.ndarray,
        classes: np.ndarray,
    ) -> float:
        """
        1. calculate entropy for all possible value for concrete attribute
        2. calculate conditional entropy
        3. calculate gain for concrete attribute
        :return:
        """
        occurrency_with_entropy = []
        for value in attribute.possible_values:
            value_occurred_idxes = Tree.getting_indexes_of_examples_with_specific_value(
                examples[:, attr_idx], value
            )
            # print(value_occurred_idxes.size)
            value_classes = np.take(classes, value_occurred_idxes)
            occurrency_with_entropy.append(
                [
                    value,
                    value_occurred_idxes.size,
                    Tree.calculate_entropy(value_classes),
                ]
            )
        conditional_entropy = 0
        for ent in occurrency_with_entropy:
            conditional_entropy += (ent[1] / np.size(examples, axis=0)) * ent[2]
        return entropy_for_whole_set - conditional_entropy

    @staticmethod
    def deciding_to_division_attribute(
        attributes: np.ndarray, examples: np.ndarray, classes: np.ndarray
    ) -> int:
        """
        1. calculate entropy for whole input set
        2. calculate gain for all possible attributes

        :param attributes:
        :param examples:
        :param classes:

        :return: index of chosen attribute
        """
        whole_input_entropy = Tree.calculate_entropy(classes)
        attributes_gains = []
        for i, attribute in enumerate(attributes):
            attributes_gains.append(
                Tree.calculation_gain(
                    whole_input_entropy, attribute, i, examples, classes
                )
            )
        return np.argmax(attributes_gains)

    @staticmethod
    def getting_indexes_of_examples_with_specific_value(
        examples, searched_value
    ) -> np.ndarray:
        return np.where(examples == searched_value)[0]

    @staticmethod
    def calculate_most_common_label(classes: np.ndarray) -> int:
        labels, counts = np.unique(classes, return_counts=True)
        idx = np.argmax(counts)
        return labels[idx]

    @staticmethod
    def calculate_class_frequency(classes: np.ndarray) -> np.ndarray:
        return np.unique(classes, return_counts=True)[1]

    @staticmethod
    def create_examples_as_string(examples: np.ndarray, divide_param=3):
        string_examples = np.ndarray(shape=examples.shape, dtype=np.dtype("U100"))
        for i in range(np.size(examples, axis=1)):
            example = examples[:, i]
            if examples.dtype in ["float64", "int64"]:
                min_val = round(np.min(example), 2)
                max_val = round(np.max(example), 2)
                val_range = max_val-min_val
                if min_val == 0 and max_val == 1:
                    string_examples[:, i] = example.astype("uint8").astype("str")
                else:
                    val_range = round(val_range/divide_param, 2)
                    for j, val in enumerate(example):
                        if val < min_val+val_range:
                            string_examples[j, i] = f"<{min_val+val_range}"
                        elif val > max_val-val_range:
                            string_examples[j, i] = f">{max_val - val_range}"
                        else:
                            string_examples[j, i] = f"{min_val+val_range}-{max_val - val_range}"
            else:
                string_examples[:, i] = example.astype("str")
        return string_examples

    @staticmethod
    def preprocess_input_dataset(
        input_dataset: pd.DataFrame, class_name=None, divide_param=3
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not class_name:
            class_name = input_dataset.columns[0]
            print(
                f"class name was not passed, taking last column from given DataFrame: {class_name}"
            )
        #id_name = input_dataset.columns[0]
        classes = input_dataset[class_name].to_numpy()
        #classes.replace(('yes', 'no'), (1, 0), inplace=True)
        tmp_attributes = np.delete(dataset.columns.to_numpy(), [0])
        attributes = np.ndarray((tmp_attributes.size,), dtype=Attribute)
        examples = input_dataset.drop([class_name], axis=1).to_numpy()
        string_examples = Tree.create_examples_as_string(examples, divide_param)
        for idx, name in enumerate(tmp_attributes):
            attributes[idx] = Attribute(name, np.unique(string_examples[:, idx]))
        return attributes, string_examples, classes

    def train(
        self, dataset: pd.DataFrame, class_name=None
    ):
        attributes, all_examples, all_classes = self.preprocess_input_dataset(
            dataset, class_name
        )
        examples, t_examples = train_test_split(all_examples, test_size=0.4, random_state=0, shuffle=True)
        classes, t_classes = train_test_split(all_classes, test_size=0.4, random_state=0, shuffle=True)

        self.model = Tree._train(attributes, examples, classes, 1)

        self.evaluate(t_examples, t_classes)

    @staticmethod
    def _train(
        attributes: np.ndarray,
        examples: np.ndarray,
        classes: np.ndarray,
        most_common_label=None,
    ):
        # TODO protect from empty input data
        if examples.size == 0:
            return Leaf(most_common_label)
        min_label = np.min(classes)
        max_label = np.max(classes)
        if min_label == max_label:
            return Leaf(min_label)
        elif attributes.size == 0:
            return Leaf(Tree.calculate_most_common_label(classes))
        else:
            most_common_label = Tree.calculate_most_common_label(classes)
            attr_idx = Tree.deciding_to_division_attribute(
                attributes, examples, classes
            )
            best_attribute = attributes[attr_idx]
            new_node = Node(best_attribute,index=attr_idx)
            new_attributes = np.delete(attributes, attr_idx)
            new_examples = np.delete(examples, attr_idx, axis=1)
            for value in new_node.attribute.possible_values:
                value_occurred_idxes = (
                    Tree.getting_indexes_of_examples_with_specific_value(
                        examples[:, attr_idx], value
                    )
                )
                truncated_classes = np.take(classes, value_occurred_idxes)
                truncated_examples = np.take(new_examples, value_occurred_idxes, axis=0)
                new_node.child_nodes[value] = Tree._train(
                    new_attributes,
                    truncated_examples,
                    truncated_classes,
                    most_common_label,
                )
        return new_node

    def predict(self, example: np.ndarray) -> int:
        model = self.model
        if len(example.shape) > 1 and example.shape[1] == 1:
            raise Exception(f"Wrong input size, to predict must be single example")
        else:
            while model.node_name != "leaf":
                anazyzed_value = str(example[model.index])
                example = np.delete(example, model.index)
                model = model.child_nodes[anazyzed_value]
            return model.value

    def penalty_function(self):
        pass

    def evaluate(self, examples: np.ndarray, classes: np.ndarray):
        predicted_classes = np.zeros(shape=classes.shape, dtype=np.dtype("object"))
        for i, example in enumerate(examples):
            label = self.predict(example)
            predicted_classes[i] = label
        conf_matrix = confusion_matrix(classes, predicted_classes)
        accuracy = accuracy_score(classes, predicted_classes)
        print("confusion matrix:")
        print(conf_matrix)
        print("accuracy:")
        print(accuracy)

    # TODO
    def metrics(self):
        pass


class Leaf:
    def __init__(self, value):
        self.value = value
        self.node_name = "leaf"

    def __repr__(self):
        return self.value


dataset_path = "/home/stepi2299/studia/test.csv"
dataset_path1 = "/home/stepi2299/studia/WSI/breast-cancer.csv"
dataset_path2 = "/home/stepi2299/studia/WSI/agaricus-lepiota.csv"
dataset = pd.read_csv(dataset_path2)
print(dataset.to_numpy().shape)
tree = Tree()
tree.train(dataset)

