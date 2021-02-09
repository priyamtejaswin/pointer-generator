#!/usr/bin/env python
"""
created at: Sun 07 Feb 2021 08:47:01 AM EST
created by: Priyam Tejaswin

Compute ROUGE scores for TARGET and HYPO.
"""


import os
import plac
import subprocess
import time


epoch = int(time.time())
@plac.pos('target_file', "File with target sentences.")
@plac.pos('pred_file', "File with predicted sentences.")
@plac.opt('scores_file', "Path to save report.")
def main(target_file, pred_file,
         scores_file=os.path.join('/tmp', 'priyam_rouges_%d.txt'%epoch)):
    """
    Cleans, truncates and saves the target, pred files to /tmp
    then run ROUGE.
    Output saved in specified location, or default `priyam_rouges_<EPOCH>.txt`.
    """

    assert os.path.isfile(target_file), target_file
    assert os.path.isfile(pred_file), pred_file
    assert not os.path.isfile(scores_file), "Scores file already exists!"

    print("Preparing files ...")

    # Load and clean targets.
    arr_targets = []
    with open(target_file) as fp:
        for line in fp.readlines():
            arr_targets.append(line.strip())

    # Load and clean preds.
    arr_preds = []
    with open(pred_file) as fp:
        for line in fp.readlines():
            clean = []
            for word in line.strip().split():
                if word == '<end>':
                    break
                else:
                    clean.append(word)

            arr_preds.append(' '.join(clean))

    print("Targets %d ,Preds %d" % (len(arr_targets), len(arr_preds)))
    assert len(arr_targets) >= len(arr_preds), "Targets are less than Preds!"

    path_targets = os.path.join('/tmp', 'priyam_targets_%d.txt'%epoch)
    with open(path_targets, 'w') as fp:
        fp.write('\n'.join(arr_targets[:len(arr_preds)]) + '\n')

    path_preds = os.path.join('/tmp', 'priyam_preds_%d.txt'%epoch)
    with open(path_preds, 'w') as fp:
        fp.write('\n'.join(arr_preds) + '\n')

    print("Target:", path_targets)
    print("Preds:", path_preds)
    print("Scores:", scores_file)
    print("Running ROUGE ...")
    command = ['python', '-m', 'rouge_score.rouge', '--target_filepattern=%s'\
        %path_targets, '--prediction_filepattern=%s'%path_preds,
        '--output_filename=%s'%scores_file, '--use_stemmer=true',
        '--rouge_types=rouge1,rouge2,rouge4,rougeL']
    print(command)
    subprocess.run(command)

    print("Done.")


if __name__ == '__main__':
    plac.call(main)

