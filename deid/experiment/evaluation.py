import contextlib
import os
from collections import OrderedDict

import numpy as np
from keras.callbacks import Callback
from keras.utils.generic_utils import Progbar
from terminaltables import SingleTable as TerminalTable

from ..data import TestSet, prediction_to_xml
from ..env import env
from ..tools.i2b2.classes import PHITrackEvaluation, Evaluate
from ..tools.i2b2.evaluate import evaluate as i2b2_evaluate


def _save_predictions_to_xmls(model, batch_size, embeddings, label2ind, ind2label, test_set, predictions_dir,
                              binary_classification, hipaa_only, extra_features, require_argmax):
    if not os.path.isdir(predictions_dir):
        os.mkdir(predictions_dir)

    print('Saving test XMLs to', predictions_dir)
    progress_bar = Progbar(target=TestSet.number_of_test_sets(test_set), verbose=env.keras_verbose)

    for i, te in enumerate(TestSet.test_sets(embeddings,
                                             test_set=test_set,
                                             label2ind=label2ind,
                                             binary_classification=binary_classification,
                                             hipaa_only=hipaa_only,
                                             extra_features=extra_features), start=1):
        preds = model.predict([te.X, te.X_extra], batch_size=batch_size)
        if require_argmax:
            preds = np.argmax(preds, axis=-1)
        xml = prediction_to_xml(te.X, preds, te.text, te.sents, ind2label)
        filename = os.path.basename(te.filename)[:-4] + '.xml'
        with open(os.path.join(predictions_dir, filename), 'w') as f:
            f.write(xml)

        progress_bar.update(i)


def _run_official_evaluation(predictions_dir, test_set, output_file, binary_classification=False, hipaa_only=False,
                             print_summary=True):
    xml_test_dir = os.path.join(env.data_dir, test_set + '_xml')

    def call_i2b2_evaluate():
        return i2b2_evaluate([predictions_dir], xml_test_dir, PHITrackEvaluation, verbose=False)

    if output_file is not None:
        with open(output_file, 'w') as f:
            with contextlib.redirect_stdout(f):
                evaluations = call_i2b2_evaluate()
    else:
        evaluations = call_i2b2_evaluate()

    result = OrderedDict()
    for evaluation in evaluations.evaluations:
        mp = evaluation.micro_precision()
        mr = evaluation.micro_recall()
        f1 = Evaluate.F_beta(mr, mp)
        result[evaluation.sys_id] = {'precision': mp, 'recall': mr, 'f1': f1}

    if print_summary:
        print('Evaluating', predictions_dir, xml_test_dir)
        print('Evaluation summary:')
        table_data = [['Evaluation', 'Precision', 'Recall', 'F1 (micro)']]
        for name, values in result.items():
            if binary_classification and 'Binary' not in name:
                continue
            if hipaa_only and 'HIPAA' not in name:
                continue
            if binary_classification and not hipaa_only and 'HIPAA' in name:
                continue  # evaluation is wrong for these because all tags get mapped to a HIPAA (name) tag
            table_data.append([name] + [round(values[key], 5) for key in ['precision', 'recall', 'f1']])

        table = TerminalTable(table_data)
        print(table.table)
        print(f'(see complete evaluation at {output_file})')

    return result


def evaluate_deid_performance(model, batch_size, embeddings, label2ind, ind2label, experiment_dir, epoch=1,
                              test_set='validation', binary_classification=False,
                              hipaa_only=False, extra_features=(), require_argmax=True):
    predictions_dir = os.path.join(experiment_dir, f'predictions_epoch_{epoch:02d}')
    _save_predictions_to_xmls(model=model, batch_size=batch_size, embeddings=embeddings, label2ind=label2ind,
                              ind2label=ind2label, test_set=test_set, predictions_dir=predictions_dir,
                              binary_classification=binary_classification, hipaa_only=hipaa_only,
                              extra_features=extra_features, require_argmax=require_argmax)

    output_file = predictions_dir + '.txt'
    return _run_official_evaluation(predictions_dir=predictions_dir, test_set=test_set, output_file=output_file,
                                    print_summary=True, binary_classification=binary_classification,
                                    hipaa_only=hipaa_only)


class DeidentificationEvaluationCallback(Callback):
    def __init__(self, deid_model, batch_size, embeddings, label2ind, ind2label, test_set, experiment_dir,
                 evaluate_every, binary_classification, hipaa_only, extra_features, call_model=False):
        super().__init__()
        self.deid_model = deid_model
        self.batch_size = batch_size
        self.embeddings = embeddings
        self.label2ind = label2ind
        self.ind2label = ind2label
        self.test_set = test_set
        self.experiment_dir = experiment_dir
        self.evaluate_every = evaluate_every
        self.binary_classification = binary_classification
        self.hipaa_only = hipaa_only
        self.extra_features = extra_features
        self.call_model = call_model

    def on_epoch_end(self, epoch, logs=None):
        deid_model = self.deid_model() if self.call_model else self.deid_model
        epoch = epoch + 1  # keras uses 0-indexed epochs
        if epoch % self.evaluate_every == 0:
            evaluate_deid_performance(model=deid_model, batch_size=self.batch_size, embeddings=self.embeddings,
                                      label2ind=self.label2ind, ind2label=self.ind2label, epoch=epoch,
                                      test_set=self.test_set, experiment_dir=self.experiment_dir,
                                      binary_classification=self.binary_classification,
                                      hipaa_only=self.hipaa_only,
                                      extra_features=self.extra_features)
