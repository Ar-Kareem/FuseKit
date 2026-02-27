from enum import Enum
import random

import torch
from transformers import PreTrainedTokenizer
from datasets import load_dataset

from Datasets.core import IterableDataset
from Datasets.core import GenerationSample
from fusekit.Common.EvalType import EvalType as EvalType
from Common.utils import split_alnum_and_lower

class MMLUSubject(Enum):
    abstract_algebra                    = ('cais/mmlu', 'abstract_algebra')
    anatomy                             = ('cais/mmlu', 'anatomy')
    astronomy                           = ('cais/mmlu', 'astronomy')
    business_ethics                     = ('cais/mmlu', 'business_ethics')
    clinical_knowledge                  = ('cais/mmlu', 'clinical_knowledge')
    college_biology                     = ('cais/mmlu', 'college_biology')
    college_chemistry                   = ('cais/mmlu', 'college_chemistry')
    college_computer_science            = ('cais/mmlu', 'college_computer_science')
    college_mathematics                 = ('cais/mmlu', 'college_mathematics')
    college_medicine                    = ('cais/mmlu', 'college_medicine')
    college_physics                     = ('cais/mmlu', 'college_physics')
    computer_security                   = ('cais/mmlu', 'computer_security')
    conceptual_physics                  = ('cais/mmlu', 'conceptual_physics')
    econometrics                        = ('cais/mmlu', 'econometrics')
    electrical_engineering              = ('cais/mmlu', 'electrical_engineering')
    elementary_mathematics              = ('cais/mmlu', 'elementary_mathematics')
    formal_logic                        = ('cais/mmlu', 'formal_logic')
    global_facts                        = ('cais/mmlu', 'global_facts')
    high_school_biology                 = ('cais/mmlu', 'high_school_biology')
    high_school_chemistry               = ('cais/mmlu', 'high_school_chemistry')
    high_school_computer_science        = ('cais/mmlu', 'high_school_computer_science')
    high_school_european_history        = ('cais/mmlu', 'high_school_european_history')
    high_school_geography               = ('cais/mmlu', 'high_school_geography')
    high_school_government_and_politics = ('cais/mmlu', 'high_school_government_and_politics')
    high_school_macroeconomics          = ('cais/mmlu', 'high_school_macroeconomics')
    high_school_mathematics             = ('cais/mmlu', 'high_school_mathematics')
    high_school_microeconomics          = ('cais/mmlu', 'high_school_microeconomics')
    high_school_physics                 = ('cais/mmlu', 'high_school_physics')
    high_school_psychology              = ('cais/mmlu', 'high_school_psychology')
    high_school_statistics              = ('cais/mmlu', 'high_school_statistics')
    high_school_us_history              = ('cais/mmlu', 'high_school_us_history')
    high_school_world_history           = ('cais/mmlu', 'high_school_world_history')
    human_aging                         = ('cais/mmlu', 'human_aging')
    human_sexuality                     = ('cais/mmlu', 'human_sexuality')
    international_law                   = ('cais/mmlu', 'international_law')
    jurisprudence                       = ('cais/mmlu', 'jurisprudence')
    logical_fallacies                   = ('cais/mmlu', 'logical_fallacies')
    machine_learning                    = ('cais/mmlu', 'machine_learning')
    management                          = ('cais/mmlu', 'management')
    marketing                           = ('cais/mmlu', 'marketing')
    medical_genetics                    = ('cais/mmlu', 'medical_genetics')
    miscellaneous                       = ('cais/mmlu', 'miscellaneous')
    moral_disputes                      = ('cais/mmlu', 'moral_disputes')
    moral_scenarios                     = ('cais/mmlu', 'moral_scenarios')
    nutrition                           = ('cais/mmlu', 'nutrition')
    philosophy                          = ('cais/mmlu', 'philosophy')
    prehistory                          = ('cais/mmlu', 'prehistory')
    professional_accounting             = ('cais/mmlu', 'professional_accounting')
    professional_law                    = ('cais/mmlu', 'professional_law')
    professional_medicine               = ('cais/mmlu', 'professional_medicine')
    professional_psychology             = ('cais/mmlu', 'professional_psychology')
    public_relations                    = ('cais/mmlu', 'public_relations')
    security_studies                    = ('cais/mmlu', 'security_studies')
    sociology                           = ('cais/mmlu', 'sociology')
    us_foreign_policy                   = ('cais/mmlu', 'us_foreign_policy')
    virology                            = ('cais/mmlu', 'virology')
    world_religions                     = ('cais/mmlu', 'world_religions')

all_subjects = list(MMLUSubject.__members__.items())

class MMLUDataset(IterableDataset):
    """
    Loads one MMLU subject split and converts rows into GenerationSample instances.
    Text format:
      <question>

      Options:
      A. <choice 0>
      B. <choice 1>
      C. <choice 2>
      D. <choice 3>

      Answer:
    Label is the correct letter (A/B/C/D).
    """
    def __init__(self, tokenizer: PreTrainedTokenizer, subject: MMLUSubject, split: str = 'test', data_limit: int = -1, max_new_tokens: int = 100, num_shots: int = 0, seed: int = 42):
        super().__init__()
        assert split != 'dev' or num_shots == 0, "MMLU dev split cannot have shots"
        assert num_shots <= 5, "MMLU cannot have more than 5 shots"
        self.name = 'MMLU_' + subject.name
        self.type = EvalType.SIMILARITY
        self.subject = subject
        path, name = subject.value
        assert split in ['test', 'validation'], "MMLU split must be 'test' or 'validation'"
        self.hf_dataset = (path, name, split)
        self.dataset = [self.parse_sample(sample) for sample in load_dataset(path=path, name=name, split=split)]
        self.all_targets = set(label_letter for _, label_letter in self.dataset)
        if num_shots > 0:
            self.shots_dataset = [self.parse_sample(sample) for sample in load_dataset(path=path, name=name, split='dev')]
            self.shots_dataset = [
                f'Prompt: {text}\nAnswer: {label_letter}'
                for text, label_letter in self.shots_dataset
            ]
        self.samples: list[MMLUSample] = []
        seeder = random.Random(seed)
        for i, (text, label_letter) in enumerate(self.dataset):
            if i >= data_limit and data_limit != -1:
                break
            shots = []
            if num_shots > 0:
                shots = seeder.sample(self.shots_dataset, num_shots)
            self.samples.append(MMLUSample(tokenizer, text, label_letter, name, max_new_tokens=max_new_tokens, shots=shots, all_targets=self.all_targets))

    def parse_sample(self, sample):
        letters = ['A', 'B', 'C', 'D']
        choices = sample['choices']
        assert len(choices) == 4, "MMLU choices should have length 4"
        options_str = "\n".join(f"{letters[j]}. {choices[j]}" for j in range(4))
        text = f"{sample['question']}\n\nOptions:\n{options_str}"
        label_letter = letters[int(sample['answer'])]
        return text, label_letter

class MMLUSample(GenerationSample):
    def __init__(self, tokenizer: PreTrainedTokenizer, text: str, label: str, task_name: str, max_new_tokens: int, shots: list[str] = [], all_targets: set = set()):
        super().__init__(tokenizer, text, label, max_new_tokens=max_new_tokens)
        self.task_name = task_name
        self.shots = shots
        self.all_targets = [split_alnum_and_lower(target) for target in all_targets]
        assert all(len(target) == 1 for target in self.all_targets), "All targets should have length 1"
        self.all_targets = set(target[0] for target in self.all_targets)

    def get_text(self) -> str:
        if len(self.shots) == 0:
            return self.prompt
        return '\n\n'.join(self.shots) + '\n\n' + self.prompt

    def get_labels(self) -> torch.Tensor:
        if self.labels is None:
            labels = self.tokenizer.encode(
                self.answer,
                add_special_tokens=False
            )
            eos_token_id = self.tokenizer.eos_token_id
            if eos_token_id is not None:
                labels.append(eos_token_id)
            self.labels = torch.tensor(labels, dtype=torch.long).unsqueeze(0)
        return self.labels
    
    def get_accuracy(self):
        gold = split_alnum_and_lower(self.answer)
        assert len(gold) == 1, "Gold label should have length 1"
        gold = gold[0]

        predicted = split_alnum_and_lower(self.pred_text)
        for i in range(len(predicted)-1, -1, -1):
            if predicted[i] in self.all_targets:
                return {"Accuracy": 1.0 if predicted[i] == gold else 0.0, "Valid": 1}
        return {"Accuracy": 0.0, "Valid": 0}


MMLUAbstractAlgebra                 = lambda tokenizer, split='test', data_limit=-1, max_new_tokens=100, num_shots=0: MMLUDataset(tokenizer, MMLUSubject.abstract_algebra, split=split, data_limit=data_limit, max_new_tokens=max_new_tokens, num_shots=num_shots)
MMLUAnatomy                         = lambda tokenizer, split='test', data_limit=-1, max_new_tokens=100, num_shots=0: MMLUDataset(tokenizer, MMLUSubject.anatomy, split=split, data_limit=data_limit, max_new_tokens=max_new_tokens, num_shots=num_shots)
MMLUAstronomy                       = lambda tokenizer, split='test', data_limit=-1, max_new_tokens=100, num_shots=0: MMLUDataset(tokenizer, MMLUSubject.astronomy, split=split, data_limit=data_limit, max_new_tokens=max_new_tokens, num_shots=num_shots)
MMLUBusinessEthics                  = lambda tokenizer, split='test', data_limit=-1, max_new_tokens=100, num_shots=0: MMLUDataset(tokenizer, MMLUSubject.business_ethics, split=split, data_limit=data_limit, max_new_tokens=max_new_tokens, num_shots=num_shots)
MMLUClinicalKnowledge               = lambda tokenizer, split='test', data_limit=-1, max_new_tokens=100, num_shots=0: MMLUDataset(tokenizer, MMLUSubject.clinical_knowledge, split=split, data_limit=data_limit, max_new_tokens=max_new_tokens, num_shots=num_shots)
MMLUCollegeBiology                  = lambda tokenizer, split='test', data_limit=-1, max_new_tokens=100, num_shots=0: MMLUDataset(tokenizer, MMLUSubject.college_biology, split=split, data_limit=data_limit, max_new_tokens=max_new_tokens, num_shots=num_shots)
MMLUCollegeChemistry                = lambda tokenizer, split='test', data_limit=-1, max_new_tokens=100, num_shots=0: MMLUDataset(tokenizer, MMLUSubject.college_chemistry, split=split, data_limit=data_limit, max_new_tokens=max_new_tokens, num_shots=num_shots)
MMLUCollegeCS                       = lambda tokenizer, split='test', data_limit=-1, max_new_tokens=100, num_shots=0: MMLUDataset(tokenizer, MMLUSubject.college_computer_science, split=split, data_limit=data_limit, max_new_tokens=max_new_tokens, num_shots=num_shots)
MMLUCollegeMathematics              = lambda tokenizer, split='test', data_limit=-1, max_new_tokens=100, num_shots=0: MMLUDataset(tokenizer, MMLUSubject.college_mathematics, split=split, data_limit=data_limit, max_new_tokens=max_new_tokens, num_shots=num_shots)
MMLUCollegeMedicine                 = lambda tokenizer, split='test', data_limit=-1, max_new_tokens=100, num_shots=0: MMLUDataset(tokenizer, MMLUSubject.college_medicine, split=split, data_limit=data_limit, max_new_tokens=max_new_tokens, num_shots=num_shots)
MMLUCollegePhysics                  = lambda tokenizer, split='test', data_limit=-1, max_new_tokens=100, num_shots=0: MMLUDataset(tokenizer, MMLUSubject.college_physics, split=split, data_limit=data_limit, max_new_tokens=max_new_tokens, num_shots=num_shots)
MMLUComputerSecurity                = lambda tokenizer, split='test', data_limit=-1, max_new_tokens=100, num_shots=0: MMLUDataset(tokenizer, MMLUSubject.computer_security, split=split, data_limit=data_limit, max_new_tokens=max_new_tokens, num_shots=num_shots)
MMLUConceptualPhysics               = lambda tokenizer, split='test', data_limit=-1, max_new_tokens=100, num_shots=0: MMLUDataset(tokenizer, MMLUSubject.conceptual_physics, split=split, data_limit=data_limit, max_new_tokens=max_new_tokens, num_shots=num_shots)
MMLUEconometrics                    = lambda tokenizer, split='test', data_limit=-1, max_new_tokens=100, num_shots=0: MMLUDataset(tokenizer, MMLUSubject.econometrics, split=split, data_limit=data_limit, max_new_tokens=max_new_tokens, num_shots=num_shots)
MMLUElectricalEngineering           = lambda tokenizer, split='test', data_limit=-1, max_new_tokens=100, num_shots=0: MMLUDataset(tokenizer, MMLUSubject.electrical_engineering, split=split, data_limit=data_limit, max_new_tokens=max_new_tokens, num_shots=num_shots)
MMLUElementaryMathematics           = lambda tokenizer, split='test', data_limit=-1, max_new_tokens=100, num_shots=0: MMLUDataset(tokenizer, MMLUSubject.elementary_mathematics, split=split, data_limit=data_limit, max_new_tokens=max_new_tokens, num_shots=num_shots)
MMLUFormalLogic                     = lambda tokenizer, split='test', data_limit=-1, max_new_tokens=100, num_shots=0: MMLUDataset(tokenizer, MMLUSubject.formal_logic, split=split, data_limit=data_limit, max_new_tokens=max_new_tokens, num_shots=num_shots)
MMLUGlobalFacts                     = lambda tokenizer, split='test', data_limit=-1, max_new_tokens=100, num_shots=0: MMLUDataset(tokenizer, MMLUSubject.global_facts, split=split, data_limit=data_limit, max_new_tokens=max_new_tokens, num_shots=num_shots)
MMLUHighSchoolBiology               = lambda tokenizer, split='test', data_limit=-1, max_new_tokens=100, num_shots=0: MMLUDataset(tokenizer, MMLUSubject.high_school_biology, split=split, data_limit=data_limit, max_new_tokens=max_new_tokens, num_shots=num_shots)
MMLUHighSchoolChemistry             = lambda tokenizer, split='test', data_limit=-1, max_new_tokens=100, num_shots=0: MMLUDataset(tokenizer, MMLUSubject.high_school_chemistry, split=split, data_limit=data_limit, max_new_tokens=max_new_tokens, num_shots=num_shots)
MMLUHighSchoolCS                    = lambda tokenizer, split='test', data_limit=-1, max_new_tokens=100, num_shots=0: MMLUDataset(tokenizer, MMLUSubject.high_school_computer_science, split=split, data_limit=data_limit, max_new_tokens=max_new_tokens, num_shots=num_shots)
MMLUHighSchoolEuropeanHistory       = lambda tokenizer, split='test', data_limit=-1, max_new_tokens=100, num_shots=0: MMLUDataset(tokenizer, MMLUSubject.high_school_european_history, split=split, data_limit=data_limit, max_new_tokens=max_new_tokens, num_shots=num_shots)
MMLUHighSchoolGeography             = lambda tokenizer, split='test', data_limit=-1, max_new_tokens=100, num_shots=0: MMLUDataset(tokenizer, MMLUSubject.high_school_geography, split=split, data_limit=data_limit, max_new_tokens=max_new_tokens, num_shots=num_shots)
MMLUHighSchoolGovtPolitics          = lambda tokenizer, split='test', data_limit=-1, max_new_tokens=100, num_shots=0: MMLUDataset(tokenizer, MMLUSubject.high_school_government_and_politics, split=split, data_limit=data_limit, max_new_tokens=max_new_tokens, num_shots=num_shots)
MMLUHighSchoolMacroeconomics        = lambda tokenizer, split='test', data_limit=-1, max_new_tokens=100, num_shots=0: MMLUDataset(tokenizer, MMLUSubject.high_school_macroeconomics, split=split, data_limit=data_limit, max_new_tokens=max_new_tokens, num_shots=num_shots)
MMLUHighSchoolMathematics           = lambda tokenizer, split='test', data_limit=-1, max_new_tokens=100, num_shots=0: MMLUDataset(tokenizer, MMLUSubject.high_school_mathematics, split=split, data_limit=data_limit, max_new_tokens=max_new_tokens, num_shots=num_shots)
MMLUHighSchoolMicroeconomics        = lambda tokenizer, split='test', data_limit=-1, max_new_tokens=100, num_shots=0: MMLUDataset(tokenizer, MMLUSubject.high_school_microeconomics, split=split, data_limit=data_limit, max_new_tokens=max_new_tokens, num_shots=num_shots)
MMLUHighSchoolPhysics               = lambda tokenizer, split='test', data_limit=-1, max_new_tokens=100, num_shots=0: MMLUDataset(tokenizer, MMLUSubject.high_school_physics, split=split, data_limit=data_limit, max_new_tokens=max_new_tokens, num_shots=num_shots)
MMLUHighSchoolPsychology            = lambda tokenizer, split='test', data_limit=-1, max_new_tokens=100, num_shots=0: MMLUDataset(tokenizer, MMLUSubject.high_school_psychology, split=split, data_limit=data_limit, max_new_tokens=max_new_tokens, num_shots=num_shots)
MMLUHighSchoolStatistics            = lambda tokenizer, split='test', data_limit=-1, max_new_tokens=100, num_shots=0: MMLUDataset(tokenizer, MMLUSubject.high_school_statistics, split=split, data_limit=data_limit, max_new_tokens=max_new_tokens, num_shots=num_shots)
MMLUHighSchoolUSHistory             = lambda tokenizer, split='test', data_limit=-1, max_new_tokens=100, num_shots=0: MMLUDataset(tokenizer, MMLUSubject.high_school_us_history, split=split, data_limit=data_limit, max_new_tokens=max_new_tokens, num_shots=num_shots)
MMLUHighSchoolWorldHistory          = lambda tokenizer, split='test', data_limit=-1, max_new_tokens=100, num_shots=0: MMLUDataset(tokenizer, MMLUSubject.high_school_world_history, split=split, data_limit=data_limit, max_new_tokens=max_new_tokens, num_shots=num_shots)
MMLUHumanAging                      = lambda tokenizer, split='test', data_limit=-1, max_new_tokens=100, num_shots=0: MMLUDataset(tokenizer, MMLUSubject.human_aging, split=split, data_limit=data_limit, max_new_tokens=max_new_tokens, num_shots=num_shots)
MMLUHumanSexuality                  = lambda tokenizer, split='test', data_limit=-1, max_new_tokens=100, num_shots=0: MMLUDataset(tokenizer, MMLUSubject.human_sexuality, split=split, data_limit=data_limit, max_new_tokens=max_new_tokens, num_shots=num_shots)
MMLUInternationalLaw                = lambda tokenizer, split='test', data_limit=-1, max_new_tokens=100, num_shots=0: MMLUDataset(tokenizer, MMLUSubject.international_law, split=split, data_limit=data_limit, max_new_tokens=max_new_tokens, num_shots=num_shots)
MMLUJurisprudence                   = lambda tokenizer, split='test', data_limit=-1, max_new_tokens=100, num_shots=0: MMLUDataset(tokenizer, MMLUSubject.jurisprudence, split=split, data_limit=data_limit, max_new_tokens=max_new_tokens, num_shots=num_shots)
MMLULogicalFallacies                = lambda tokenizer, split='test', data_limit=-1, max_new_tokens=100, num_shots=0: MMLUDataset(tokenizer, MMLUSubject.logical_fallacies, split=split, data_limit=data_limit, max_new_tokens=max_new_tokens, num_shots=num_shots)
MMLUMachineLearning                 = lambda tokenizer, split='test', data_limit=-1, max_new_tokens=100, num_shots=0: MMLUDataset(tokenizer, MMLUSubject.machine_learning, split=split, data_limit=data_limit, max_new_tokens=max_new_tokens, num_shots=num_shots)
MMLUManagement                      = lambda tokenizer, split='test', data_limit=-1, max_new_tokens=100, num_shots=0: MMLUDataset(tokenizer, MMLUSubject.management, split=split, data_limit=data_limit, max_new_tokens=max_new_tokens, num_shots=num_shots)
MMLUMarketing                       = lambda tokenizer, split='test', data_limit=-1, max_new_tokens=100, num_shots=0: MMLUDataset(tokenizer, MMLUSubject.marketing, split=split, data_limit=data_limit, max_new_tokens=max_new_tokens, num_shots=num_shots)
MMLUMedicalGenetics                 = lambda tokenizer, split='test', data_limit=-1, max_new_tokens=100, num_shots=0: MMLUDataset(tokenizer, MMLUSubject.medical_genetics, split=split, data_limit=data_limit, max_new_tokens=max_new_tokens, num_shots=num_shots)
MMLUMiscellaneous                   = lambda tokenizer, split='test', data_limit=-1, max_new_tokens=100, num_shots=0: MMLUDataset(tokenizer, MMLUSubject.miscellaneous, split=split, data_limit=data_limit, max_new_tokens=max_new_tokens, num_shots=num_shots)
MMLUMoralDisputes                   = lambda tokenizer, split='test', data_limit=-1, max_new_tokens=100, num_shots=0: MMLUDataset(tokenizer, MMLUSubject.moral_disputes, split=split, data_limit=data_limit, max_new_tokens=max_new_tokens, num_shots=num_shots)
MMLUMoralScenarios                  = lambda tokenizer, split='test', data_limit=-1, max_new_tokens=100, num_shots=0: MMLUDataset(tokenizer, MMLUSubject.moral_scenarios, split=split, data_limit=data_limit, max_new_tokens=max_new_tokens, num_shots=num_shots)
MMLUNutrition                       = lambda tokenizer, split='test', data_limit=-1, max_new_tokens=100, num_shots=0: MMLUDataset(tokenizer, MMLUSubject.nutrition, split=split, data_limit=data_limit, max_new_tokens=max_new_tokens, num_shots=num_shots)
MMLUPhilosophy                      = lambda tokenizer, split='test', data_limit=-1, max_new_tokens=100, num_shots=0: MMLUDataset(tokenizer, MMLUSubject.philosophy, split=split, data_limit=data_limit, max_new_tokens=max_new_tokens, num_shots=num_shots)
MMLUPrehistory                      = lambda tokenizer, split='test', data_limit=-1, max_new_tokens=100, num_shots=0: MMLUDataset(tokenizer, MMLUSubject.prehistory, split=split, data_limit=data_limit, max_new_tokens=max_new_tokens, num_shots=num_shots)
MMLUProfessionalAccounting          = lambda tokenizer, split='test', data_limit=-1, max_new_tokens=100, num_shots=0: MMLUDataset(tokenizer, MMLUSubject.professional_accounting, split=split, data_limit=data_limit, max_new_tokens=max_new_tokens, num_shots=num_shots)
MMLUProfessionalLaw                 = lambda tokenizer, split='test', data_limit=-1, max_new_tokens=100, num_shots=0: MMLUDataset(tokenizer, MMLUSubject.professional_law, split=split, data_limit=data_limit, max_new_tokens=max_new_tokens, num_shots=num_shots)
MMLUProfessionalMedicine            = lambda tokenizer, split='test', data_limit=-1, max_new_tokens=100, num_shots=0: MMLUDataset(tokenizer, MMLUSubject.professional_medicine, split=split, data_limit=data_limit, max_new_tokens=max_new_tokens, num_shots=num_shots)
MMLUProfessionalPsychology          = lambda tokenizer, split='test', data_limit=-1, max_new_tokens=100, num_shots=0: MMLUDataset(tokenizer, MMLUSubject.professional_psychology, split=split, data_limit=data_limit, max_new_tokens=max_new_tokens, num_shots=num_shots)
MMLUPublicRelations                 = lambda tokenizer, split='test', data_limit=-1, max_new_tokens=100, num_shots=0: MMLUDataset(tokenizer, MMLUSubject.public_relations, split=split, data_limit=data_limit, max_new_tokens=max_new_tokens, num_shots=num_shots)
MMLUSecurityStudies                 = lambda tokenizer, split='test', data_limit=-1, max_new_tokens=100, num_shots=0: MMLUDataset(tokenizer, MMLUSubject.security_studies, split=split, data_limit=data_limit, max_new_tokens=max_new_tokens, num_shots=num_shots)
MMLUSociology                       = lambda tokenizer, split='test', data_limit=-1, max_new_tokens=100, num_shots=0: MMLUDataset(tokenizer, MMLUSubject.sociology, split=split, data_limit=data_limit, max_new_tokens=max_new_tokens, num_shots=num_shots)
MMLUSForeignPolicy                  = lambda tokenizer, split='test', data_limit=-1, max_new_tokens=100, num_shots=0: MMLUDataset(tokenizer, MMLUSubject.us_foreign_policy, split=split, data_limit=data_limit, max_new_tokens=max_new_tokens, num_shots=num_shots)
MMLUVirology                        = lambda tokenizer, split='test', data_limit=-1, max_new_tokens=100, num_shots=0: MMLUDataset(tokenizer, MMLUSubject.virology, split=split, data_limit=data_limit, max_new_tokens=max_new_tokens, num_shots=num_shots)
MMLUWorldReligions                  = lambda tokenizer, split='test', data_limit=-1, max_new_tokens=100, num_shots=0: MMLUDataset(tokenizer, MMLUSubject.world_religions, split=split, data_limit=data_limit, max_new_tokens=max_new_tokens, num_shots=num_shots)

all_mmlu_datasets = [MMLUAbstractAlgebra, MMLUAnatomy, MMLUAstronomy, MMLUBusinessEthics, MMLUClinicalKnowledge, MMLUCollegeBiology, MMLUCollegeChemistry, MMLUCollegeCS, MMLUCollegeMathematics, MMLUCollegeMedicine, MMLUCollegePhysics, MMLUComputerSecurity, MMLUConceptualPhysics, MMLUEconometrics, MMLUElectricalEngineering, MMLUElementaryMathematics, MMLUFormalLogic, MMLUGlobalFacts, MMLUHighSchoolBiology, MMLUHighSchoolChemistry, MMLUHighSchoolCS, MMLUHighSchoolEuropeanHistory, MMLUHighSchoolGeography, MMLUHighSchoolGovtPolitics, MMLUHighSchoolMacroeconomics, MMLUHighSchoolMathematics, MMLUHighSchoolMicroeconomics, MMLUHighSchoolPhysics, MMLUHighSchoolPsychology, MMLUHighSchoolStatistics, MMLUHighSchoolUSHistory, MMLUHighSchoolWorldHistory, MMLUHumanAging, MMLUHumanSexuality, MMLUInternationalLaw, MMLUJurisprudence, MMLULogicalFallacies, MMLUMachineLearning, MMLUManagement, MMLUMarketing, MMLUMedicalGenetics, MMLUMiscellaneous, MMLUMoralDisputes, MMLUMoralScenarios, MMLUNutrition, MMLUPhilosophy, MMLUPrehistory, MMLUProfessionalAccounting, MMLUProfessionalLaw, MMLUProfessionalMedicine, MMLUProfessionalPsychology, MMLUPublicRelations, MMLUSecurityStudies, MMLUSociology, MMLUSForeignPolicy, MMLUVirology, MMLUWorldReligions]
