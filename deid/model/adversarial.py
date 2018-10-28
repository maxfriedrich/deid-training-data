from types import MappingProxyType

from keras import backend as K
from keras.layers import Input, Lambda, concatenate
from keras.losses import binary_crossentropy
from keras.models import Model

from . import get as get_deidentifier
from .adversary import get as get_adversary
from .optimizer import get as get_optimizer
from .representer import get as get_representer


class AdversarialModel:
    def __init__(self,
                 *_,  # don't allow any positional arguments
                 embedding_size,
                 output_size,
                 representation_size=None,
                 representation_type='lstm',
                 representation_args=MappingProxyType({}),
                 deidentifier_type='lstm',
                 deidentifier_args=MappingProxyType({}),
                 extra_input_size=0,
                 adversaries=('discriminate-representations', 'discriminate-representation-embedding-pair'),
                 adversary_args=MappingProxyType({}),
                 optimizer='adam',
                 optimizer_args=MappingProxyType({})):
        """ Initialize the adversarial model. It's components are
        - a representation model that transforms embeddings into a (noisy) representation
        - a deidentifier model that performs the deidentification task from the representation
        - an adversary model that tries to reconstruct information from the representation

        :param embedding_size: the representation input size
        :param output_size: the deidentifier output size
        :param representation_size: the representation size (or None to use the embedding size)
        :param representation_type: the type of representation model to use (see representer.py)
        :param representation_args: the kwargs for the representation model
        :param deidentifier_type: the type of deidentifier model to use (see deidentifier.py)
        :param deidentifier_args: the kwargs for the deidentifier model
        :param adversaries: a sequence of adversary type strings (see adversary.py)
        :param adversary_args: a dictionary of adversary args or a list of dictionaries (if every adversary should get
            its own args)
        :param optimizer: the type of optimizer to use (see optimizer.py)
        :param optimizer_args: the args passed to the optimizer
        """

        if representation_size is None:
            representation_size = embedding_size

        original_embeddings = Input(shape=(None, embedding_size))

        build_representer = get_representer(representation_type)
        self.train_representer = build_representer(embedding_size=embedding_size,
                                                   representation_size=representation_size,
                                                   apply_noise=True,
                                                   **representation_args)

        train_representation = self.train_representer(original_embeddings)

        deidentifier, deidentifier_loss = get_deidentifier(deidentifier_type)(
            name='deidentifier',
            input_size=representation_size,
            output_size=output_size,
            extra_input_size=extra_input_size,
            **deidentifier_args)

        extra_input = Input(shape=(None, extra_input_size))
        if extra_input_size > 0:
            train_deidentifier_input = [train_representation, extra_input]
        else:
            train_deidentifier_input = train_representation

        train_deidentifier_output = deidentifier(train_deidentifier_input)
        self.pretrain_deidentifier = Model([original_embeddings, extra_input], train_deidentifier_output)
        self.pretrain_deidentifier.compile(optimizer=get_optimizer(optimizer)(**optimizer_args), loss=deidentifier_loss,
                                           metrics=['accuracy'])

        self.train_representer.trainable = False

        adv_embeddings = Input(shape=(None, embedding_size))
        adv_representation = self.train_representer(adv_embeddings)

        adv_fake_embeddings = Input(shape=(None, embedding_size))
        adv_fake_representation = self.train_representer(adv_fake_embeddings)

        adversary_models = []
        adversary_outputs = []
        if isinstance(adversary_args, dict):
            adversary_args = [adversary_args for _ in adversaries]

        for adversary_type, args in zip(adversaries, adversary_args):
            adversary = get_adversary(adversary_type)(inputs={'train_representation': adv_representation,
                                                              'original_embeddings': adv_embeddings,
                                                              'fake_representation': adv_fake_representation},
                                                      representation_size=representation_size,
                                                      embedding_size=embedding_size,
                                                      **args)
            adversary_models.append(adversary.model)
            adversary_outputs.append(adversary.model(adversary.inputs))
            adversary.model.summary()
        adversary_output = concatenate(adversary_outputs, axis=-1)
        adversary_output = Lambda(lambda x: K.mean(x, axis=-1, keepdims=True), name='adversary')(adversary_output)

        self.pretrain_adversary = Model([adv_embeddings, adv_fake_embeddings], adversary_output)
        self.pretrain_adversary.summary()
        self.pretrain_adversary.compile(optimizer=get_optimizer(optimizer)(**optimizer_args),
                                        loss='binary_crossentropy',
                                        metrics=['accuracy'])

        self.fine_tune_branches = Model([original_embeddings, extra_input, adv_embeddings, adv_fake_embeddings],
                                        [train_deidentifier_output, adversary_output])
        self.fine_tune_branches.compile(optimizer=get_optimizer(optimizer)(**optimizer_args),
                                        loss=[deidentifier_loss, 'binary_crossentropy'],
                                        metrics=['accuracy'])

        self.train_representer.trainable = True
        deidentifier.trainable = False
        for adversary in adversary_models:
            adversary.trainable = False
        self.fine_tune_representer = Model([original_embeddings, extra_input, adv_embeddings, adv_fake_embeddings],
                                           [train_deidentifier_output, adversary_output])
        self.fine_tune_representer.compile(optimizer=get_optimizer(optimizer)(**optimizer_args),
                                           loss=[deidentifier_loss, adversarial_objective],
                                           loss_weights=[1, 1], metrics=['accuracy'])

    @property
    def complete_model(self):
        return self.fine_tune_branches


def adversarial_objective(y_true, y_pred):
    loss = binary_crossentropy(y_true, y_pred)
    random_guessing = -K.log(0.5)
    return K.abs(loss - random_guessing)
