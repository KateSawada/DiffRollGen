import glob
import os

import hydra
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf

import SubjectiveMuseEvaluator as SME


@hydra.main(
    config_path="SubjectiveMuseEvaluator/config", config_name="evaluation")
def main(cfg):
    if (not os.path.exists(to_absolute_path(cfg.output_dir))):
        os.makedirs(to_absolute_path(cfg.output_dir))
    # save config
    OmegaConf.save(
        cfg, os.path.join(to_absolute_path(cfg.output_dir), "config.yaml"))

    cfg.samples_root_dir = to_absolute_path(cfg.samples_root_dir)
    loader = getattr(SME.loader, cfg.evaluation_loader.name)()
    reshaper = getattr(SME.reshaper, cfg.reshaper.name)(
        songs_per_sample=cfg.songs_per_sample,
        n_tracks=len(cfg.track_names),
        measures_per_song=cfg.measures_per_song,
        timesteps_per_measure=cfg.timesteps_per_measure,
        n_pitches=cfg.n_pitches,
        hard_threshold=cfg.hard_threshold,
        **cfg.reshaper.args,
    )
    postprocess = getattr(SME.stats, cfg.stats.name)()

    filenames = glob.glob(
        os.path.join(
            to_absolute_path(cfg.samples_root_dir),
            f"*.{cfg.samples_extension}"))

    methods = []
    metrics_keys = list(cfg.metrics.keys())
    for i in range(len(metrics_keys)):
        print(metrics_keys[i])
        methods += [
            getattr(SME.metrics, cfg.metrics[metrics_keys[i]].name)(
                reshaper=reshaper,
                postprocess=postprocess,
                n_samples=len(filenames),
                **cfg.metrics[metrics_keys[i]].args,
            )
        ]

    evaluator = SME.Evaluator(
        methods=methods,
        filenames=filenames,
        loader=loader,
    )
    evaluator.run(
        os.path.join(to_absolute_path(cfg.output_dir), "result.yaml"))


if __name__ == '__main__':
    main()
