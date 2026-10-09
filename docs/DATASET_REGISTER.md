# Dataset register

## Football Image Dataset

| Field | Recorded value |
|---|---|
| Source | `https://github.com/skaczylo/Football-Image-Dataset` |
| Local path | `data/external/football-image-dataset` |
| Retrieved commit | `264212872f1bfa6a9a3413f298793474da741aec` |
| Declared license | CC BY 4.0 |
| Images | 612 Full HD JPG frames |
| Provided split | 512 train / 100 validation |
| Classes | ball, player, referee |
| Boxes | 500 ball / 8,844 player / 792 referee |
| Local packed download | approximately 219 MB |
| Intended use | Development fine-tuning experiment |
| Portfolio redistribution | Do not redistribute until source-footage provenance is independently reviewed |

The repository declares CC BY 4.0 and includes a license file. The images appear to be extracted from several broadcast sources, online highlights and PES 21 gameplay. The repository does not provide original match identifiers or a source-rights chain for each image. Its license declaration is useful evidence, but it is not enough for us to represent that every underlying broadcast frame is cleared for unrestricted portfolio redistribution.

Automated audit results:

- Every image has a matching label file.
- Every label has a supported class and normalized coordinates within `[0, 1]`.
- No byte-identical image appears in both splits.
- A simple average-hash screen returned seven low-distance cross-split pairs; visual inspection showed different pitch scenes, so these were false positives caused by large green regions.
- Match-grouped isolation cannot be verified from the filenames.
- The validation set contains 43 PES 21 frames, 34 Netherlands versus France frames, 19 Newcastle versus Chelsea frames and four Serbia U21 versus England U21 frames.

Use this dataset to test the training pipeline and produce a preliminary checkpoint. Do not use its validation score as the final product metric. The later evaluation set must contain separately sourced, authorized real-match footage grouped by complete match.
