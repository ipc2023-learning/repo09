import argparse
import math
import os
import re
from pathlib import Path
from typing import List, Optional
from statistics import mean, median


class TrainingStep:
    def __init__(self, update_magnitude: float, delta_seconds: float, total_seconds: float):
        self.update_magnitude = update_magnitude
        self.delta_seconds = delta_seconds
        self.total_seconds = total_seconds


class TrainingTrajectoryStep:
    def __init__(self, trajectory: List[TrainingStep]):
        self.trajectory = trajectory


class ValidationStep:
    def __init__(self, average_policy_evaluation_value: float, delta_seconds: float, total_seconds: float):
        self.average_policy_evaluation_value = average_policy_evaluation_value
        self.delta_seconds = delta_seconds
        self.total_seconds = total_seconds


class ValidationSession:
    def __init__(self, name: str, validation_steps: List[ValidationStep], optimal_validation_loss: float):
        self.name = name
        self.validation_steps = validation_steps
        self.optimal_validation_loss = optimal_validation_loss


class Epoch:
    def __init__(self, epoch_id: int, training_trajectory_steps: List[TrainingTrajectoryStep], validation_step: ValidationStep):
        self.id = epoch_id
        self.training_trajectory_steps = training_trajectory_steps
        self.validation_step = validation_step


class TrainingSession:
    def __init__(self, name, optimal_validation_loss: float, epochs: List[Epoch]):
        self.name = name
        self.optimal_validation_loss = optimal_validation_loss
        self.epochs = epochs


def _parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, type=Path, help='Input directory with logs')
    parser.add_argument('--label', required=True, type=str, help='Label of figure')
    parser.add_argument('--max_time', required=False, type=int, default=None, help='Clip data past max_time')
    args = parser.parse_args()
    return args


def _parse_validation_session(file: Path) -> Optional[ValidationSession]:
    optimal_value_regex = re.compile(r'Optimal validation loss: (\d+\.\d+)')
    validation_regex = re.compile(r'\[(\d+)\] Validation: (\d+\.\d+), d = (\d+\.\d+) s, t = (\d+\.\d+) s')
    optimal_validation_loss = None
    validation_steps = []
    with file.open() as file_stream:
        for line in file_stream:
            match = optimal_value_regex.match(line)
            if match:
                optimal_validation_loss = float(match.group(1))
            match = validation_regex.match(line)
            if match:
                loss = float(match.group(2))
                delta_seconds = float(match.group(3))
                total_seconds = float(match.group(4))
                validation_steps.append(ValidationStep(loss, delta_seconds, total_seconds))
    if optimal_validation_loss is not None:
        name = os.path.splitext(file.name)[0]
        name = name.replace('_', '').replace('-', '').replace('.', '')
        return ValidationSession(name, validation_steps, optimal_validation_loss)
    else:
        return None


# def _parse_file(file: Path) -> Optional[Session]:
#     opt_val_regex = re.compile(r'Optimal validation loss: (\d+\.\d+)')
#     train_regex = re.compile(r'\[(\d+), (\d+)/(\d+), (\d+)/(\d+)\] Train: (-?\d+\.\d+), d = (\d+\.\d+) s, t = (\d+\.\d+)')
#     batch_regex = re.compile(r'\[(\d+), (\d+)/(\d+)\] Batch: d = (\d+\.\d+) s, t = (\d+\.\d+)')
#     validation_regex = re.compile(r'\[(\d+)\] Validation: (\d+\.\d+), d = (\d+\.\d+) s, t = (\d+\.\d+)')
#     epochs = []
#     trajectory = []
#     trajectories = []
#     optimal_validation_loss = None
#     last_train_loss_match = None
#     last_validation_step = None
#     with file.open() as file_stream:
#         for line in file_stream:
#             match = opt_val_regex.match(line)
#             if match:
#                 optimal_validation_loss = float(match.group(1))
#             match = train_regex.match(line)
#             if match:
#                 last_train_loss_match = match
#             match = batch_regex.match(line)
#             if match:
#                 step_index = int(last_train_loss_match.group(4))
#                 num_steps = int(last_train_loss_match.group(5))
#                 loss = float(last_train_loss_match.group(6))
#                 last_train_loss_match = None  # Ensure we never use a match twice
#                 epoch_index = int(match.group(1))
#                 delta_seconds = float(match.group(4))
#                 total_seconds = float(match.group(5))
#                 # Check if we've rolled over to a new epoch
#                 if len(epochs) < epoch_index:
#                     epochs.append(Epoch(len(epochs), trajectories, last_validation_step))
#                     trajectories = []
#                     last_validation_step = None
#                 # Add a step in the trajectory and check if it has terminated
#                 trajectory.append(TrainingStep(loss, delta_seconds, total_seconds))
#                 if step_index == num_steps:
#                     trajectories.append(TrainingTrajectoryStep(trajectory))
#                     trajectory = []
#             match = validation_regex.match(line)
#             if match:
#                 loss = float(match.group(2))
#                 delta_seconds = float(match.group(3))
#                 total_seconds = float(match.group(4))
#                 last_validation_step = ValidationStep(loss, delta_seconds, total_seconds)
#     if optimal_validation_loss is not None:
#         name = os.path.splitext(file.name)[0]
#         name = name.replace('_', '').replace('-', '')
#         return Session(name, optimal_validation_loss, epochs)
#     else:
#         return None


def _plot_sessions(sessions: List[ValidationSession], label: str, time_limit: Optional[int]):
    if time_limit is None:
        session_validation_steps = [session.validation_steps for session in sessions]
    else:
        session_validation_steps = [[step for step in session.validation_steps if step.total_seconds < time_limit] for session in sessions]
    print('\\begin{tikzpicture}')
    print('  \\begin{axis}[')
    print('    axis x line=bottom,')
    print('    axis y line=left,')
    print('    legend style={at={(1.0, 0.5)}, anchor=east},')
    print('    xlabel=Time (minutes),')
    print('    ylabel=Average Policy Evaluation Value')
    print('  ]')
    for (index, validation_steps) in enumerate(session_validation_steps):
        name = sessions[index].name
        print('    \\addlegendentry{{{}}}'.format(name))
        print('    \\addplot coordinates {')
        for validation_step in validation_steps:
            validation_loss = validation_step.average_policy_evaluation_value
            time_seconds = validation_step.total_seconds
            print('      ({:.5f}, {:.5f})'.format(time_seconds / 60, validation_loss))
        print('    };')
    print('  \\end{axis}')
    print('\\end{tikzpicture}')


def _main():
    args = _parse_arguments()
    input_directory: Path = args.input
    files = [file for file in input_directory.glob('**/*.txt')]
    files.sort()
    sessions: List[ValidationSession] = []
    for file in files:
        parsed_file = _parse_validation_session(file)
        if parsed_file is not None:
            sessions.append(parsed_file)
    _plot_sessions(sessions, args.label, args.max_time)


if __name__ == '__main__':
    _main()
