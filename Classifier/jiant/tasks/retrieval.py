import os
from typing import Optional

from Classifier.jiant.tasks.lib.abductive_nli import AbductiveNliTask
from Classifier.jiant.tasks.lib.acceptability_judgement.definiteness import AcceptabilityDefinitenessTask
from Classifier.jiant.tasks.lib.adversarial_nli import AdversarialNliTask
from Classifier.jiant.tasks.lib.arc_easy import ArcEasyTask
from Classifier.jiant.tasks.lib.arc_challenge import ArcChallengeTask
from Classifier.jiant.tasks.lib.boolq import BoolQTask
from Classifier.jiant.tasks.lib.bucc2018 import Bucc2018Task
from Classifier.jiant.tasks.lib.ccg import CCGTask
from Classifier.jiant.tasks.lib.cola import ColaTask
from Classifier.jiant.tasks.lib.commitmentbank import CommitmentBankTask
from Classifier.jiant.tasks.lib.commonsenseqa import CommonsenseQATask
from Classifier.jiant.tasks.lib.edge_probing.nonterminal import NonterminalTask
from Classifier.jiant.tasks.lib.copa import CopaTask
from Classifier.jiant.tasks.lib.edge_probing.coref import CorefTask
from Classifier.jiant.tasks.lib.cosmosqa import CosmosQATask
from Classifier.jiant.tasks.lib.edge_probing.dep import DepTask
from Classifier.jiant.tasks.lib.edge_probing.dpr import DprTask
from Classifier.jiant.tasks.lib.fever_nli import FeverNliTask
from Classifier.jiant.tasks.lib.glue_diagnostics import GlueDiagnosticsTask
from Classifier.jiant.tasks.lib.hellaswag import HellaSwagTask
from Classifier.jiant.tasks.lib.mctaco import MCTACOTask
from Classifier.jiant.tasks.lib.mctest import MCTestTask
from Classifier.jiant.tasks.lib.mlm_simple import MLMSimpleTask
from Classifier.jiant.tasks.lib.mlm_premasked import MLMPremaskedTask
from Classifier.jiant.tasks.lib.mlm_pretokenized import MLMPretokenizedTask
from Classifier.jiant.tasks.lib.mlqa import MlqaTask
from Classifier.jiant.tasks.lib.mnli import MnliTask
from Classifier.jiant.tasks.lib.mnli_mismatched import MnliMismatchedTask
from Classifier.jiant.tasks.lib.mrpc import MrpcTask
from Classifier.jiant.tasks.lib.mrqa_natural_questions import MrqaNaturalQuestionsTask
from Classifier.jiant.tasks.lib.multirc import MultiRCTask
from Classifier.jiant.tasks.lib.mutual import MutualTask
from Classifier.jiant.tasks.lib.mutual_plus import MutualPlusTask
from Classifier.jiant.tasks.lib.edge_probing.ner import NerTask
from Classifier.jiant.tasks.lib.newsqa import NewsQATask
from Classifier.jiant.tasks.lib.panx import PanxTask
from Classifier.jiant.tasks.lib.pawsx import PawsXTask
from Classifier.jiant.tasks.lib.edge_probing.pos import PosTask
from Classifier.jiant.tasks.lib.qamr import QAMRTask
from Classifier.jiant.tasks.lib.qasrl import QASRLTask
from Classifier.jiant.tasks.lib.qqp import QqpTask
from Classifier.jiant.tasks.lib.qnli import QnliTask
from Classifier.jiant.tasks.lib.quail import QuailTask
from Classifier.jiant.tasks.lib.quoref import QuorefTask
from Classifier.jiant.tasks.lib.record import ReCoRDTask
from Classifier.jiant.tasks.lib.rte import RteTask
from Classifier.jiant.tasks.lib.scitail import SciTailTask
from Classifier.jiant.tasks.lib.senteval.tense import SentevalTenseTask
from Classifier.jiant.tasks.lib.edge_probing.semeval import SemevalTask
from Classifier.jiant.tasks.lib.snli import SnliTask
from Classifier.jiant.tasks.lib.socialiqa import SocialIQATask
from Classifier.jiant.tasks.lib.edge_probing.spr1 import Spr1Task
from Classifier.jiant.tasks.lib.edge_probing.spr2 import Spr2Task
from Classifier.jiant.tasks.lib.squad import SquadTask
from Classifier.jiant.tasks.lib.edge_probing.srl import SrlTask
from Classifier.jiant.tasks.lib.sst import SstTask
from Classifier.jiant.tasks.lib.stsb import StsbTask
from Classifier.jiant.tasks.lib.superglue_axg import SuperglueWinogenderDiagnosticsTask
from Classifier.jiant.tasks.lib.superglue_axb import SuperglueBroadcoverageDiagnosticsTask
from Classifier.jiant.tasks.lib.swag import SWAGTask
from Classifier.jiant.tasks.lib.tatoeba import TatoebaTask
from Classifier.jiant.tasks.lib.tydiqa import TyDiQATask
from Classifier.jiant.tasks.lib.udpos import UdposTask
from Classifier.jiant.tasks.lib.wic import WiCTask
from Classifier.jiant.tasks.lib.wnli import WnliTask
from Classifier.jiant.tasks.lib.wsc import WSCTask
from Classifier.jiant.tasks.lib.xnli import XnliTask
from Classifier.jiant.tasks.lib.xquad import XquadTask
from Classifier.jiant.tasks.lib.mcscript import MCScriptTask
from Classifier.jiant.tasks.lib.arct import ArctTask
from Classifier.jiant.tasks.lib.winogrande import WinograndeTask
from Classifier.jiant.tasks.lib.piqa import PiqaTask
from Classifier.jiant.tasks.lib.trec import trectask
from Classifier.jiant.tasks.lib.FakeNews import FakeNewstask
from Classifier.jiant.tasks.lib.Irony import Ironytask
from Classifier.jiant.tasks.lib.SST2 import SST2task
from Classifier.jiant.tasks.lib.ironyb import ironybtask

from Classifier.jiant.tasks.core import Task
from Classifier.jiant.utils.python.io import read_json


TASK_DICT = {
    "abductive_nli": AbductiveNliTask,
    "arc_easy": ArcEasyTask,
    "arc_challenge": ArcChallengeTask,
    "superglue_axg": SuperglueWinogenderDiagnosticsTask,
    "acceptability_definiteness": AcceptabilityDefinitenessTask,
    "adversarial_nli": AdversarialNliTask,
    "boolq": BoolQTask,
    "bucc2018": Bucc2018Task,
    "cb": CommitmentBankTask,
    "ccg": CCGTask,
    "cola": ColaTask,
    "commonsenseqa": CommonsenseQATask,
    "nonterminal": NonterminalTask,
    "copa": CopaTask,
    "coref": CorefTask,
    "cosmosqa": CosmosQATask,
    "dep": DepTask,
    "dpr": DprTask,
    "fever_nli": FeverNliTask,
    "glue_diagnostics": GlueDiagnosticsTask,
    "hellaswag": HellaSwagTask,
    "mctaco": MCTACOTask,
    "mctest": MCTestTask,
    "mlm_simple": MLMSimpleTask,
    "mlm_premasked": MLMPremaskedTask,
    "mlm_pretokenized": MLMPretokenizedTask,
    "mlqa": MlqaTask,
    "mnli": MnliTask,
    "mnli_mismatched": MnliMismatchedTask,
    "multirc": MultiRCTask,
    "mutual": MutualTask,
    "mutual_plus": MutualPlusTask,
    "mrpc": MrpcTask,
    "mrqa_natural_questions": MrqaNaturalQuestionsTask,
    "ner": NerTask,
    "newsqa": NewsQATask,
    "pawsx": PawsXTask,
    "panx": PanxTask,
    "pos": PosTask,
    "qamr": QAMRTask,
    "qasrl": QASRLTask,
    "qnli": QnliTask,
    "qqp": QqpTask,
    "quail": QuailTask,
    "quoref": QuorefTask,
    "record": ReCoRDTask,
    "rte": RteTask,
    "scitail": SciTailTask,
    "senteval_tense": SentevalTenseTask,
    "semeval": SemevalTask,
    "snli": SnliTask,
    "socialiqa": SocialIQATask,
    "spr1": Spr1Task,
    "spr2": Spr2Task,
    "squad": SquadTask,
    "srl": SrlTask,
    "sst": SstTask,
    "stsb": StsbTask,
    "superglue_axb": SuperglueBroadcoverageDiagnosticsTask,
    "swag": SWAGTask,
    "tatoeba": TatoebaTask,
    "tydiqa": TyDiQATask,
    "udpos": UdposTask,
    "wic": WiCTask,
    "wnli": WnliTask,
    "wsc": WSCTask,
    "xnli": XnliTask,
    "xquad": XquadTask,
    "mcscript": MCScriptTask,
    "arct": ArctTask,
    "winogrande": WinograndeTask,
    "piqa": PiqaTask,
    "trec":trectask,
    "TREC6":trectask,
    "FakeNews":FakeNewstask,
    "Irony":Ironytask,
    "IronyB":ironybtask,
    "SST2":SST2task,
}


def get_task_class(task_name: str):
    task_class = TASK_DICT[task_name]
    assert issubclass(task_class, Task)
    return task_class


def create_task_from_config(config: dict, base_path: Optional[str] = None, verbose: bool = False):
    """Create task instance from task config.

    Args:
        config (Dict): task config map.
        base_path (str): if the path is not absolute, path is assumed to be relative to base_path.
        verbose (bool): True if task config should be printed during task creation.

    Returns:
        Task instance.

    """
    task_class = get_task_class(config["task"])
    for k in config["paths"].keys():
        path = config["paths"][k]
        # TODO: Refactor paths  (issue #1180)
        if isinstance(path, str) and not os.path.isabs(path):
            assert base_path
            config["paths"][k] = os.path.join(base_path, path)
    task_kwargs = config.get("kwargs", {})
    if verbose:
        print(task_class.__name__)
        for k, v in config["paths"].items():
            print(f"  [{k}]: {v}")
    # noinspection PyArgumentList
    return task_class(name=config["name"], path_dict=config["paths"], **task_kwargs)


def create_task_from_config_path(config_path: str, verbose: bool = False):
    """Creates task instance from task config filepath.

    Args:
        config_path (str): config filepath.
        verbose (bool): True if task config should be printed during task creation.

    Returns:
        Task instance.

    """
    return create_task_from_config(
        read_json(config_path), base_path=os.path.split(config_path)[0], verbose=verbose,
    )
