from typing import Callable, List

from src.datasets.crypto.bitcoin_block_propagation import \
    BitcoinBlockPropagationDataset
from src.datasets.cybersecurity.ddos import DDoSDataset
from src.datasets.dataset import Dataset
from src.datasets.epidemics.covid19 import Covid19Dataset
from src.datasets.epidemics.waves import WavesDataset
from src.datasets.random.synthetic import SyntheticDataset
from src.datasets.social_media.fake_news import FakeNewsDataset


class CurriculumLesson:
    def __init__(self, duration: int, edge_probability: float = None, num_graphs: int = None, num_nodes: int = None):
        self.num_graphs = num_graphs
        self.num_nodes = num_nodes
        self.edge_probability = edge_probability
        self.duration = duration

    def __str__(self) -> str:
        return f"Lesson: GRAPHS: {self.num_graphs}, NODES: {self.num_nodes}, EDGES: {self.edge_probability}, D: {self.duration}"


class Curriculum:
    def __init__(self, name: str, lessons: List[CurriculumLesson]):
        self.name = name
        self.lessons = lessons


def complexity(incremental=True):
    return [10, 20, 30, 40, 50] if incremental else [50, 40, 30, 20, 10]


def diversity(incremental=True):
    return [1, 10, 20, 50, 100] if incremental else [100, 50, 20, 10, 1]


def density(incremental=True):
    return [0.1, 0.3, 0.5, 0.7, 0.9] if incremental else [0.9, 0.7, 0.5, 0.3, 0.1]


def complexity_curriculum(lesson_duration: int = 1, num_epochs: int = 30, incremental=True) -> Curriculum:
    lessons = []
    num_nodes = complexity(incremental)
    scheduled_epochs = 0
    i = 0
    while scheduled_epochs < num_epochs:
        idx = i if i < len(num_nodes) else -1
        lesson = CurriculumLesson(
            duration=lesson_duration, num_nodes=num_nodes[idx])
        lessons.append(lesson)
        scheduled_epochs += lesson_duration
        i += 1
    return Curriculum(name="Complexity", lessons=lessons)


def diversity_curriculum(lesson_duration: int = 1, num_epochs: int = 30, incremental=True) -> Curriculum:
    lessons = []
    num_graphs = diversity(incremental)
    scheduled_epochs = 0
    i = 0
    while scheduled_epochs < num_epochs:
        idx = i if i < len(num_graphs) else -1
        lesson = CurriculumLesson(
            duration=lesson_duration, num_graphs=num_graphs[idx])
        lessons.append(lesson)
        scheduled_epochs += lesson_duration
        i += 1
    return Curriculum(name="Diversity", lessons=lessons)


def density_curriculum(lesson_duration: int = 1, num_epochs: int = 30, incremental=True) -> Curriculum:
    lessons = []
    edge_probabilities = density(incremental)
    scheduled_epochs = 0
    i = 0
    while scheduled_epochs < num_epochs:
        idx = i if i < len(edge_probabilities) else -1
        lesson = CurriculumLesson(
            duration=lesson_duration, edge_probability=edge_probabilities[idx])
        lessons.append(lesson)
        scheduled_epochs += lesson_duration
        i += 1
    return Curriculum(name="Density", lessons=lessons)


def complexity_diversity_curriculum(lesson_duration: int = 1, num_epochs: int = 30, incremental=True) -> Curriculum:
    lessons = []
    num_nodes = complexity(incremental)
    num_graphs = diversity(incremental)
    scheduled_epochs = 0
    i = 0
    while scheduled_epochs < num_epochs:
        node_i, graph_i = divmod(i, len(num_graphs))
        # Use the last element if the index is out of bounds
        node_i = min(node_i, len(num_nodes) - 1)
        lesson = CurriculumLesson(
            duration=lesson_duration,
            num_graphs=num_graphs[graph_i],
            num_nodes=num_nodes[node_i]
        )
        lessons.append(lesson)
        scheduled_epochs += lesson_duration
        if graph_i != len(num_graphs) - 1 or node_i != len(num_nodes) - 1:
            i += 1
    return Curriculum(name="Complexity+Diversity", lessons=lessons)


def complexity_density_curriculum(lesson_duration: int = 1, num_epochs: int = 30, incremental=True) -> Curriculum:
    lessons = []
    num_nodes = complexity(incremental)
    edge_probabilities = density(incremental)
    scheduled_epochs = 0
    i = 0
    while scheduled_epochs < num_epochs:
        node_i, edge_i = divmod(i, len(edge_probabilities))
        # Use the last element if the index is out of bounds
        node_i = min(node_i, len(num_nodes) - 1)
        lesson = CurriculumLesson(
            duration=lesson_duration,
            edge_probability=edge_probabilities[edge_i],
            num_nodes=num_nodes[node_i]
        )
        lessons.append(lesson)
        scheduled_epochs += lesson_duration
        if edge_i != len(edge_probabilities) - 1 or node_i != len(num_nodes) - 1:
            i += 1
    return Curriculum(name="Complexity+Density", lessons=lessons)


def diversity_density_curriculum(lesson_duration: int = 1, num_epochs: int = 30, incremental=True) -> Curriculum:
    lessons = []
    edge_probabilities = density(incremental)
    num_graphs = diversity(incremental)
    scheduled_epochs = 0
    i = 0
    while scheduled_epochs < num_epochs:
        edge_i, graph_i = divmod(i, len(num_graphs))
        # Use the last element if the index is out of bounds
        edge_i = min(edge_i, len(edge_probabilities) - 1)
        lesson = CurriculumLesson(
            duration=lesson_duration,
            edge_probability=edge_probabilities[edge_i],
            num_graphs=num_graphs[graph_i]
        )
        lessons.append(lesson)
        scheduled_epochs += lesson_duration
        if graph_i != len(num_graphs) - 1 or edge_i != len(edge_probabilities) - 1:
            i += 1
    return Curriculum(name="Diversity+Density", lessons=lessons)


def create_dataset_from_lesson(dataset_class: Dataset, lesson: CurriculumLesson, incremental=False, **dataset_kwargs):
    dataset = dataset_class(
        log_progress=True,
        **({'train_distinct_behavior_graphs': lesson.num_graphs} if lesson.num_graphs is not None else {}),
        **({'num_nodes': lesson.num_nodes} if lesson.num_nodes is not None else {}),
        **({'behavior_edge_probability': lesson.edge_probability} if lesson.edge_probability is not None else {}),
        name_suffix="inc" if incremental else "dec",
        **dataset_kwargs
    )
    return dataset


dataset_helper = {
    "Synthetic": SyntheticDataset,
    "COVID-19": Covid19Dataset,
    "FakeNews": FakeNewsDataset,
    "BTC-BlockPropagation": BitcoinBlockPropagationDataset,
    "DDoS": DDoSDataset,
    "Waves": WavesDataset,
}


def train_with_curriculum(train_fn: Callable, curriculum: Curriculum, base_dataset: Dataset, incremental=True, train_original_epochs=10, **dataset_kwargs):
    for lesson in curriculum.lessons:
        dataset_name = base_dataset.name.split("&")[0]
        dataset = create_dataset_from_lesson(
            dataset_class=dataset_helper[dataset_name] if dataset_name in dataset_helper else dataset_helper["Synthetic"], lesson=lesson, incremental=incremental, **dataset_kwargs
        )
        train_fn(dataset, lesson.duration)
    if train_original_epochs > 0:
        return train_fn(base_dataset, train_original_epochs)
