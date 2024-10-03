# third party
# The code below is a modified version of https://github.com/seedatnabeel/Data-IQ
from autograd_lib import autograd_lib
import numpy as np
import torch


class DataIQTorch:
    def __init__(self, X, y, sparse_labels: bool = False):
        """
        The function takes in the training data and the labels, and stores them in the class variables X
        and y. It also stores the boolean value of sparse_labels in the class variable _sparse_labels

        Args:
          X: the input data
          y: the true labels
          sparse_labels (bool): boolean to identify if labels are one-hot encoded or not. If not=True.
        Defaults to False
        """
        self.X = X
        self.y = y
        self._sparse_labels = sparse_labels

        # placeholder
        self._gold_labels_probabilities = None
        self._true_probabilities = None
        self._grads = None

    def gradient(self, net, device):
        """
        Used to compute the norm of the gradient through training

        Args:
          net: pytorch neural network
          device: device to run the computation on
        """

        # setup
        data = torch.tensor(self.X, device=device)
        targets = torch.tensor(self.y, device=device).long()
        loss_fn = torch.nn.NLLLoss()

        model = net

        # register the model for autograd
        autograd_lib.register(model)

        activations = {}

        def save_activations(layer, A, _):
            activations[layer] = A

        with autograd_lib.module_hook(save_activations):
            output = model(data)
            loss = loss_fn(output, targets)

        norms = [torch.zeros(data.shape[0], device=device)]

        def per_example_norms(layer, _, B):
            A = activations[layer]
            norms[0] += (A * A).sum(dim=1) * (B * B).sum(dim=1)

        with autograd_lib.module_hook(per_example_norms):
            loss.backward()

        grads_train = norms[0].cpu().numpy()

        if self._grads is None:  # Happens only on first iteration
            self._grads = np.expand_dims(grads_train, axis=-1)
        else:
            stack = [self._grads, np.expand_dims(grads_train, axis=-1)]
            self._grads = np.hstack(stack)

    def on_epoch_end(self, net, device="cpu", **kwargs):
        """
        The function computes the gold label and true label probabilities over all samples in the
        dataset

        We iterate through the dataset, and for each sample, we compute the gold label probability (i.e.
        the actual ground truth label) and the true label probability (i.e. the predicted label).

        We then append these probabilities to the `_gold_labels_probabilities` and `_true_probabilities`
        lists.

        We do this for every sample in the dataset, and for every epoch.

        Args:
          net: the neural network
          device: the device to use for the computation. Defaults to cpu
        """

        # compute the gradient norm
        self.gradient(net, device)

        # Compute both the gold label and true label probabilities over all samples in the dataset
        gold_label_probabilities = (
            list()
        )  # gold label probabilities, i.e. actual ground truth label
        true_probabilities = list()  # true label probabilities, i.e. predicted label

        net.eval()
        with torch.no_grad():
            # iterate through the dataset
            for i in range(len(self.X)):

                # set as torch tensors
                x = torch.tensor(self.X[i, :], device=device)
                y = torch.tensor(self.y[i], device=device)

                # forward pass
                probabilities = net(x)

                # one hot encode the labels
                y = torch.nn.functional.one_hot(
                    y.to(torch.int64),
                    num_classes=probabilities.shape[-1],
                )

                # Now we extract the gold label and predicted true label probas

                # If the labels are binary [0,1]
                if len(torch.squeeze(y)) == 1:
                    # get true labels
                    true_probabilities = torch.tensor(probabilities)

                    # get gold labels
                    probabilities, y = torch.squeeze(
                        torch.tensor(probabilities),
                    ), torch.squeeze(y)

                    batch_gold_label_probabilities = torch.where(
                        y == 0,
                        1 - probabilities,
                        probabilities,
                    )

                # if labels are one hot encoded, e.g. [[1,0,0], [0,1,0]]
                elif len(torch.squeeze(y)) == 2:
                    # get true labels
                    batch_true_probabilities = torch.max(probabilities)

                    # get gold labels
                    batch_gold_label_probabilities = torch.masked_select(
                        probabilities,
                        y.bool(),
                    )
                else:

                    # get true labels
                    batch_true_probabilities = torch.max(probabilities)

                    # get gold labels
                    batch_gold_label_probabilities = torch.masked_select(
                        probabilities,
                        y.bool(),
                    )

                # move torch tensors to cpu as np.arrays()
                batch_gold_label_probabilities = (
                    batch_gold_label_probabilities.cpu().numpy()
                )
                batch_true_probabilities = batch_true_probabilities.cpu().numpy()

                # Append the new probabilities for the new batch
                gold_label_probabilities = np.append(
                    gold_label_probabilities,
                    [batch_gold_label_probabilities],
                )
                true_probabilities = np.append(
                    true_probabilities,
                    [batch_true_probabilities],
                )

        # Append the new gold label probabilities
        if self._gold_labels_probabilities is None:  # On first epoch of training
            self._gold_labels_probabilities = np.expand_dims(
                gold_label_probabilities,
                axis=-1,
            )
        else:
            stack = [
                self._gold_labels_probabilities,
                np.expand_dims(gold_label_probabilities, axis=-1),
            ]
            self._gold_labels_probabilities = np.hstack(stack)

        # Append the new true label probabilities
        if self._true_probabilities is None:  # On first epoch of training
            self._true_probabilities = np.expand_dims(true_probabilities, axis=-1)
        else:
            stack = [
                self._true_probabilities,
                np.expand_dims(true_probabilities, axis=-1),
            ]
            self._true_probabilities = np.hstack(stack)

    @property
    def get_grads(self):
        """
        Returns:
            Grad norm through training: np.array(n_samples, n_epochs)
        """
        return self._grads

    @property
    def gold_labels_probabilities(self) -> np.ndarray:
        """
        Returns:
            Gold label predicted probabilities of the "correct" label: np.array(n_samples, n_epochs)
        """
        return self._gold_labels_probabilities

    @property
    def true_probabilities(self) -> np.ndarray:
        """
        Returns:
            Actual predicted probabilities of the predicted label: np.array(n_samples, n_epochs)
        """
        return self._true_probabilities

    @property
    def confidence(self) -> np.ndarray:
        """
        Returns:
            Average predictive confidence across epochs: np.array(n_samples)
        """
        return np.mean(self._gold_labels_probabilities, axis=-1)

    @property
    def aleatoric(self):
        """
        Returns:
            Aleatric uncertainty of true label probability across epochs: np.array(n_samples): np.array(n_samples)
        """
        preds = self._gold_labels_probabilities
        return np.mean(preds * (1 - preds), axis=-1)

    @property
    def variability(self) -> np.ndarray:
        """
        Returns:
            Epistemic variability of true label probability across epochs: np.array(n_samples)
        """
        return np.std(self._gold_labels_probabilities, axis=-1)

    @property
    def correctness(self) -> np.ndarray:
        """
        Returns:
            Proportion of times a sample is predicted correctly across epochs: np.array(n_samples)
        """
        return np.mean(self._gold_labels_probabilities > 0.5, axis=-1)

    @property
    def entropy(self):
        """
        Returns:
            Predictive entropy of true label probability across epochs: np.array(n_samples)
        """
        X = self._gold_labels_probabilities
        return -1 * np.sum(X * np.log(X + 1e-12), axis=-1)

    @property
    def mi(self):
        """
        Returns:
            Mutual information of true label probability across epochs: np.array(n_samples)
        """
        X = self._gold_labels_probabilities
        entropy = -1 * np.sum(X * np.log(X + 1e-12), axis=-1)

        X = np.mean(self._gold_labels_probabilities, axis=1)
        entropy_exp = -1 * np.sum(X * np.log(X + 1e-12), axis=-1)
        return entropy - entropy_exp


class DataIQSKLearn:
    def __init__(self, X, y, sparse_labels: bool = False, model_type: str = "sklearn"):
        """
        The function takes in the training data and the labels, and stores them in the class variables X
        and y. It also stores the boolean value of sparse_labels in the class variable _sparse_labels

        Args:
          X: the input data
          y: the true labels
          sparse_labels (bool): boolean to identify if labels are one-hot encoded or not. If not=True.
        Defaults to False
        """
        self.X = X
        self.y = np.asarray(y).tolist()
        self._sparse_labels = sparse_labels

        # placeholder
        self._gold_labels_probabilities = None
        self._true_probabilities = None
        self.model_type = model_type

    def on_epoch_end(self, clf, device="cpu", iteration=1, **kwargs):
        """
        The function computes the gold label and true label probabilities over all samples in the
        dataset

        We iterate through the dataset, and for each sample, we compute the gold label probability (i.e.
        the actual ground truth label) and the true label probability (i.e. the predicted label).

        We then append these probabilities to the `_gold_labels_probabilities` and `_true_probabilities`
        lists.

        We do this for every sample in the dataset

        Args:
          clf: the classifier object
          device: The device to use for the computation. Defaults to cpu
          iteration: The current iteration of the training loop. Defaults to 1
        """
        # Compute both the gold label and true label probabilities over all samples in the dataset
        gold_label_probabilities = (
            list()
        )  # gold label probabilities, i.e. actual ground truth label
        true_probabilities = list()  # true label probabilities, i.e. predicted label

        x = self.X
        y = torch.tensor(self.y, device=device)

        match self.model_type:

            case "xgboost":
                probabilities = torch.tensor(
                    clf.predict_proba(x, iteration_range=(0, iteration)),
                    device=device,
                )
            case "catboost":
                probabilities = torch.tensor(
                    clf.predict_proba(x, ntree_start=0, ntree_end=iteration),
                    device=device,
                )
            case _:
                probabilities = torch.tensor(
                    clf.predict_proba(x),
                    device=device,
                )

        # one hot encode the labels
        y = torch.nn.functional.one_hot(
            y.to(torch.int64),
            num_classes=probabilities.shape[-1],
        )

        # Now we extract the gold label and predicted true label probas

        # If the labels are binary [0,1]
        if len(torch.squeeze(y)) == 1:
            # get true labels
            true_probabilities = torch.tensor(probabilities)

            # get gold labels
            probabilities, y = torch.squeeze(
                torch.tensor(probabilities),
            ), torch.squeeze(y)

            batch_gold_label_probabilities = torch.where(
                y == 0,
                1 - probabilities,
                probabilities,
            )

        # if labels are one hot encoded, e.g. [[1,0,0], [0,1,0]]
        elif len(torch.squeeze(y)) == 2:
            # get true labels
            batch_true_probabilities = torch.max(probabilities)

            # get gold labels
            batch_gold_label_probabilities = torch.masked_select(
                probabilities,
                y.bool(),
            )
        else:

            # get true labels
            batch_true_probabilities = torch.max(probabilities)

            # get gold labels
            batch_gold_label_probabilities = torch.masked_select(
                probabilities,
                y.bool(),
            )

        # move torch tensors to cpu as np.arrays()
        batch_gold_label_probabilities = batch_gold_label_probabilities.cpu().numpy()
        batch_true_probabilities = batch_true_probabilities.cpu().numpy()

        # Append the new probabilities for the new batch
        gold_label_probabilities = np.append(
            gold_label_probabilities,
            [batch_gold_label_probabilities],
        )
        true_probabilities = np.append(true_probabilities, [batch_true_probabilities])

        # Append the new gold label probabilities
        if self._gold_labels_probabilities is None:  # On first epoch of training
            self._gold_labels_probabilities = np.expand_dims(
                gold_label_probabilities,
                axis=-1,
            )
        else:
            stack = [
                self._gold_labels_probabilities,
                np.expand_dims(gold_label_probabilities, axis=-1),
            ]
            self._gold_labels_probabilities = np.hstack(stack)

        # Append the new true label probabilities
        if self._true_probabilities is None:  # On first epoch of training
            self._true_probabilities = np.expand_dims(true_probabilities, axis=-1)
        else:
            stack = [
                self._true_probabilities,
                np.expand_dims(true_probabilities, axis=-1),
            ]
            self._true_probabilities = np.hstack(stack)

    @property
    def gold_labels_probabilities(self) -> np.ndarray:
        """
        Returns:
            Gold label predicted probabilities of the "gold" label: np.array(n_samples, n_epochs)
        """
        return self._gold_labels_probabilities

    @property
    def true_probabilities(self) -> np.ndarray:
        """
        Returns:
            Actual predicted probabilities of the predicted label: np.array(n_samples, n_epochs)
        """
        return self._true_probabilities

    @property
    def confidence(self) -> np.ndarray:
        """
        Returns:
            Average predictive confidence across epochs: np.array(n_samples)
        """
        return np.mean(self._gold_labels_probabilities, axis=-1)

    @property
    def aleatoric(self):
        """
        Returns:
            Aleatric uncertainty of true label probability across epochs: np.array(n_samples): np.array(n_samples)
        """
        preds = self._gold_labels_probabilities
        return np.mean(preds * (1 - preds), axis=-1)

    @property
    def variability(self) -> np.ndarray:
        """
        Returns:
            Epistemic variability of true label probability across epochs: np.array(n_samples)
        """
        return np.std(self._gold_labels_probabilities, axis=-1)

    @property
    def correctness(self) -> np.ndarray:
        """
        Returns:
            Proportion of times a sample is predicted correctly across epochs: np.array(n_samples)
        """
        return np.mean(self._gold_labels_probabilities > 0.5, axis=-1)

    @property
    def entropy(self):
        """
        Returns:
            Predictive entropy of true label probability across epochs: np.array(n_samples)
        """
        X = self._gold_labels_probabilities
        return -1 * np.sum(X * np.log(X + 1e-12), axis=-1)

    @property
    def mi(self):
        """
        Returns:
            Mutual information of true label probability across epochs: np.array(n_samples)
        """
        X = self._gold_labels_probabilities
        entropy = -1 * np.sum(X * np.log(X + 1e-12), axis=-1)

        X = np.mean(self._gold_labels_probabilities, axis=1)
        entropy_exp = -1 * np.sum(X * np.log(X + 1e-12), axis=-1)
        return entropy - entropy_exp
