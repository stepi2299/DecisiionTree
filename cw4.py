import numpy as np
import pandas as pd
from math import e, log
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from typing import Tuple, Dict, Union

THRESHOLD = 0.6


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
        self.depth = 0
        self.nodes_count = 0
        self.depth_max = None
        self.nodes_max = 0

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

        :return: index of chosen attribute and value of max information gain
        """
        whole_input_entropy = Tree.calculate_entropy(classes)
        attributes_gains = []
        for i, attribute in enumerate(attributes):
            attributes_gains.append(
                Tree.calculation_gain(
                    whole_input_entropy, attribute, i, examples, classes
                )
            )
        return np.argmax(attributes_gains), np.max(attributes_gains)

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

    def preprocess_input_dataset(
            self,
        input_dataset: pd.DataFrame, class_idx=-1, id_idx=None, bins_nr=5
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        columns_to_drop = [input_dataset.columns[class_idx]]
        idx_to_drop = [class_idx]
        if id_idx is not None:
            columns_to_drop.append(input_dataset.columns[id_idx])
            idx_to_drop.append(id_idx)
        classes = input_dataset[input_dataset.columns[class_idx]].to_numpy()
        tmp_attributes = np.delete(input_dataset.columns.to_numpy(), idx_to_drop)
        attributes = np.ndarray((tmp_attributes.size,), dtype=Attribute)
        examples = input_dataset.drop(columns_to_drop, axis=1)
        for inx in range(tmp_attributes.size):
            processed_data = examples[examples.columns[inx]]
            pos_value = len(processed_data.unique())
            if pos_value > 20:
                self.nodes_max += bins_nr
                examples[examples.columns[inx]] = pd.cut(processed_data, bins=bins_nr, labels=np.arange(bins_nr))
            else:
                self.nodes_max += pos_value
        self.nodes_max *= tmp_attributes.size-1
        examples = examples.to_numpy(dtype=np.dtype("object"))
        for idx, name in enumerate(tmp_attributes):
            attributes[idx] = Attribute(name, np.unique(examples[:, idx]))
        return attributes, examples, classes

    def train(
        self, dataset: pd.DataFrame, class_idx=-1, id_idx=None, bins_nr=5
    ):
        attributes, all_examples, all_classes = self.preprocess_input_dataset(
            dataset, class_idx, id_idx, bins_nr
        )
        self.depth_max = len(attributes)
        self.leafs = 0
        examples, t_examples = train_test_split(all_examples, test_size=0.4, random_state=2, shuffle=True)
        classes, t_classes = train_test_split(all_classes, test_size=0.4, random_state=2, shuffle=True)

        self.model = self._train(attributes, examples, classes, 1)
        print("nodes", self.nodes_count)
        print("depth", self.depth)
        print("leafs", self.leafs)
        print("depth_max", self.depth_max)
        print("nodes_max", self.nodes_max)

        self.evaluate(t_examples, t_classes)

    def _train(
        self,
        attributes: np.ndarray,
        examples: np.ndarray,
        classes: np.ndarray,
        most_common_label=None,
    ):
        # TODO protect from empty input data
        if examples.size == 0:
            self.leafs += 1
            return Leaf(most_common_label)
        min_label = np.min(classes)
        max_label = np.max(classes)
        if min_label == max_label:
            self.leafs += 1
            return Leaf(min_label)
        elif attributes.size == 0:
            self.leafs += 1
            return Leaf(Tree.calculate_most_common_label(classes))
        else:
            most_common_label = Tree.calculate_most_common_label(classes)
            attr_idx, attribute_gain = Tree.deciding_to_division_attribute(
                attributes, examples, classes
            )
            best_attribute = attributes[attr_idx]
            current_depth = self.depth_max-len(attributes) + 1
            self.depth = max(self.depth, current_depth)
            if attribute_gain >= self.penalty_function(current_depth):
                new_node = Node(best_attribute, index=attr_idx)
            else:
                self.leafs += 1
                return Leaf(most_common_label)
            self.nodes_count += 1
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
                new_node.child_nodes[value] = self._train(
                    new_attributes,
                    truncated_examples,
                    truncated_classes,
                    most_common_label,
                )

        return new_node

    def predict(self, example: np.ndarray) -> int:
        tmp = 0
        model = self.model
        if len(example.shape) > 1 and example.shape[1] == 1:
            raise Exception(f"Wrong input size, to predict must be single example")
        else:
            while model.node_name != "leaf":
                tmp += 1
                anazyzed_value = example[model.index]
                example = np.delete(example, model.index)
                model = model.child_nodes[anazyzed_value]
            return model.value

    def penalty_function(self, current_depth: int):
        """
        Function which calculate penalty value, it is threshold which decide if information gain is big enough to
                create new Node, if not Leaf is created
        :param current_depth: depth of concrete node
        :return: threshold value
        """
        return THRESHOLD*(self.nodes_count/self.nodes_max + current_depth/self.depth_max)/2

    def evaluate(self, examples: np.ndarray, classes: np.ndarray):
        # TODO find solution with datatype of classes, best is int64 but sometimes it can failed if classes are not numbers
        wanted_type = classes.dtype
        # classes = classes.astype(np.dtype("object"))
        predicted_classes = np.zeros(shape=classes.shape, dtype=np.dtype(wanted_type))
        for i, example in enumerate(examples):
            label = self.predict(example)
            predicted_classes[i] = label
        conf_matrix = confusion_matrix(classes, predicted_classes)
        accuracy = accuracy_score(classes, predicted_classes)
        report = classification_report(classes, predicted_classes)
        print("confusion matrix:")
        print(conf_matrix)
        print("accuracy:")
        print(accuracy)
        print("classification report:")
        print(report)

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
dataset = pd.read_csv(dataset_path1)
tree = Tree()
tree.train(dataset, class_idx=-1, id_idx=None)

