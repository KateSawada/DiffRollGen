from multiprocessing import Manager

import numpy as np
from torch.utils.data import Dataset
from torchaudio.functional import resample
import torchaudio
import pathlib
import torch
from hydra.utils import to_absolute_path


def read_txt(file_list):
    """Read .txt file list

    Arg:
        file_list (str): txt file filename

    Return:
        (list): list of read lines

    """
    with open(file_list, "r") as f:
        filenames = f.readlines()
    return [filename.replace("\n", "") for filename in filenames]


class MuseGANDataset(Dataset):
    """PyTorch compatible audio and acoustic feat. dataset."""

    def __init__(
        self,
        pianoroll_list,
        n_tracks=5,
        measure_resolution=48,
        n_pitches=84,
        n_measures=4,
        return_filename=False,
        allow_cache=False,
    ):
        """Initialize dataset.

        Args:
            pianoroll_list (str): Filename of the list of pianoroll npy files.
            n_tracks (int): Number of tracks.
            measure_resolution (int): timestep resolution per measure.
            n_pitches (int): Number of pitches.
            n_measures (int): Number of measures.
            return_filename (bool): Whether to return the filename with arrays.
            allow_cache (bool): Whether to allow cache of the loaded files.
        """
        # load pianoroll files & check filename
        pianoroll_files = read_txt(to_absolute_path(pianoroll_list))

        self.pianoroll_files = pianoroll_files
        self.n_tracks = n_tracks
        self.measure_resolution = measure_resolution
        self.n_pitches = n_pitches
        self.n_measures = n_measures
        self.allow_cache = allow_cache

        if allow_cache:
            # NOTE(kan-bayashi): Manager is need to share memory in dataloader
            # with num_workers > 0
            self.manager = Manager()
            self.caches = self.manager.list()
            self.caches += [() for _ in range(len(pianoroll_files))]

    def __getitem__(self, idx):
        """Get specified idx items.

        Args:
            idx (int): Index of the item.

        Returns:
            str: Utterance id (only in return_filename = True).
            ndarray: Pianoroll (measures, timestep, pitches, tracks).
        """
        if self.allow_cache and len(self.caches[idx]) != 0:
            return self.caches[idx]
        # load audio and features
        pianoroll = np.load(to_absolute_path(self.pianoroll_files[idx]))
        pianoroll = pianoroll.transpose(3, 0, 1, 2)
        pianoroll = pianoroll.astype(np.float32)

        # expand n_pitches to DiffRoll(88)
        pianoroll_musical_inst = np.pad(
            pianoroll[1:],
            ((0, 0), (0, 0), (0, 0), (4, 0)),
            mode="constant",
            constant_values=0)
        pianoroll_drum = np.pad(
            pianoroll[0:1],
            ((0, 0), (0, 0), (0, 0), (0, 4)),
            mode="constant",
            constant_values=0)
        pianoroll = np.concatenate((pianoroll_musical_inst, pianoroll_drum))

        items = {
            "frame": pianoroll,
            # "audio": None,  # None is not allowed in the default collate
            "audio": np.zeros(1),
        }

        if self.allow_cache:
            self.caches[idx] = items

        return items

    def __len__(self):
        """Return dataset length.

        Returns:
            int: The length of dataset.

        """
        return len(self.pianoroll_files)


class LPDTrackDataset(Dataset):
    """LPDTrackDataset Extracting single track """

    def __init__(
        self,
        pianoroll_list,
        n_tracks=5,
        used_tracks=(1),
        measure_resolution=48,
        n_pitches=84,
        n_measures=4,
        return_filename=False,
        allow_cache=False,
    ):
        """Initialize dataset.

        Args:
            pianoroll_list (str): Filename of the list of pianoroll npy files.
            n_tracks (int): Number of tracks.
            used_tracks (Tuple[int]): track indexes to be extracted
            measure_resolution (int): timestep resolution per measure.
            n_pitches (int): Number of pitches.
            n_measures (int): Number of measures.
            return_filename (bool): Whether to return the filename with arrays.
            allow_cache (bool): Whether to allow cache of the loaded files.
        """
        # load pianoroll files & check filename
        pianoroll_files = read_txt(to_absolute_path(pianoroll_list))

        self.pianoroll_files = pianoroll_files
        self.n_tracks = n_tracks
        self.used_tracks = used_tracks
        self.measure_resolution = measure_resolution
        self.n_pitches = n_pitches
        self.n_measures = n_measures
        self.allow_cache = allow_cache

        if allow_cache:
            # NOTE(kan-bayashi): Manager is need to share memory in dataloader
            # with num_workers > 0
            self.manager = Manager()
            self.caches = self.manager.list()
            self.caches += [() for _ in range(len(pianoroll_files))]

    def __getitem__(self, idx):
        """Get specified idx items.

        Args:
            idx (int): Index of the item.

        Returns:
            str: Utterance id (only in return_filename = True).
            ndarray: Pianoroll (measures, timestep, pitches, tracks).
        """
        if self.allow_cache and len(self.caches[idx]) != 0:
            return self.caches[idx]
        # load audio and features
        pianoroll = np.load(to_absolute_path(self.pianoroll_files[idx]))
        pianoroll = pianoroll.transpose(3, 0, 1, 2)
        pianoroll = pianoroll.astype(np.float32)

        # expand n_pitches to DiffRoll(88)
        pianoroll = np.pad(
            pianoroll,
            ((0, 0), (0, 0), (0, 0), (4, 0)),
            mode="constant",
            constant_values=0)
        pianoroll = pianoroll[self.used_tracks]

        items = {
            "frame": pianoroll,
            # "audio": None,  # None is not allowed in the default collate
            "audio": np.zeros(1),
        }

        if self.allow_cache:
            self.caches[idx] = items

        return items

    def __len__(self):
        """Return dataset length.

        Returns:
            int: The length of dataset.

        """
        return len(self.pianoroll_files)
