import os
import sys
from typing import Optional

deid_dir = os.path.dirname(__file__)


# Defining the attributes as static variables isn't super smart as they are all created even if the config is not used.
# This means os.environ['SOME_SPECIFIC_VAR'] will crash in other environments, so we have to use os.environ.get().
class Environment:
    name: str
    deid_dir: str = deid_dir
    data_dir: str
    work_dir: str
    resources_dir: str
    results_dir: str
    limit_training_documents: Optional[int]
    limit_validation_documents: Optional[int]
    use_short_sentences: bool
    keras_verbose: int
    save_model: int
    embeddings_cache: bool
    experiment_dir_postfix: Optional[str] = None

    unk_token: str = '<unk>'
    sent_start = '<s>'
    sent_end = '</s>'


class Development(Environment):
    name = 'development'
    work_dir = os.path.join(os.environ['HOME'], 'deid_work')
    resources_dir = os.path.join(os.environ['HOME'], 'deid_resources')
    data_dir = os.path.join(resources_dir, 'i2b2_data')
    limit_training_documents = None  # set this to e.g. 10 for faster experimentation
    limit_validation_documents = None
    use_short_sentences = False
    keras_verbose = 1
    save_model = True
    embeddings_cache = True


class Test(Environment):
    name = 'unit test'
    work_dir = os.path.join(deid_dir, 'fixtures', 'deid_work')
    resources_dir = os.path.join(deid_dir, 'fixtures', 'deid_resources')
    data_dir = os.path.join(resources_dir, 'i2b2_data')
    limit_training_documents = 4
    limit_validation_documents = 2
    use_short_sentences = True
    keras_verbose = 1
    save_model = False
    embeddings_cache = True


env: Environment
if 'DEID_TEST_CONFIG' in os.environ.keys() and os.environ['DEID_TEST_CONFIG']:
    env = Test()
else:
    env = Development()
sys.stderr.write(f'Using {env.name} environment.\n')
