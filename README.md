# Robustness-of-Panoptic-Segmentation-for-Degraded-Automotive-Cameras-Data (T-ASE 2025, to be presented in ICRA2026!)

This is the **PyTorch re-implementation** of our T-ASE accepted paper: 
Robustness-of-Panoptic-Segmentation-for-Degraded-Automotive-Cameras-Data, [link](https://drive.google.com/file/d/1tu9D4qzwBZrjX63JRrqgI5hpND569SWd/view). 

While panoptic quality might be affected by automotive camera data quality, a comprehensive understanding and modelling of their relationship remains underexplored. Motivated by such a need, this work proposes a unifying pipeline to evaluate the robustness of panoptic segmentation models for automotive cameras, correlating it with 8 traditional image quality metrics (IQA).

<img src="docs/D-Cityscapes+.png" alt="Illustration of D-Cityscapes+" width="400"/>
- **New robustness dataset**: **Degraded-Cityscapes+ (D-Cityscapes+)** (Novel model for snow and unfavourable light)
<img src="docs/pipeline.png" alt="Illustrating of the unifying degradation data generation pipeline." width="700"/>
- **Unifying degradation pipeline**: unifying pipeline to assess the robustness of panoptic segmentation models for assisted and automated driving (AAD) systems, correlating it with image quality.
<img src="docs/PQ_FPS_results.png" alt="Benchmarking Results." width="400"/>
- **Benchmarking experiments**: 14 state-of-the-art CNN- and transformer-based panoptic segmentation networks are used to compare their robustness.

**Summary**: The presented insights provide practitioners with a valuable benchmark, predictive metrics, and clear guidelines for developing robust, real-time-capable segmentation models suitable for challenging automotive environments.

## Timeline 
- First submission (14-Jan-2025) 
- 🚢 Revise and Resubmit: 3 months (07-Apr-2025)
- Resub: 2 months revision time (06-Jun-2025)
- 🥹 Conditionally Accept: 1 month (24-Jul-2025)
- Resub: 1 month revision time (21-Aug-2025)
- 🎉 T-ASE Accepted: 1 week (25-Aug-2025)
- ⌛️ Total: ～8 months
- 🎹 Accepted to be presented in ICRA 2026, see you in 🇦🇹！(1st-Feb-2026)

## Requirements
Here, we show examples of using the EfficeintPS, DeepLab, and Oneformer for the Cityscape dataset. 
- Requirement for EfficientPS is from [here](https://github.com/DeepSceneSeg/EfficientPS#system-requirements).
- Requirement for DeepLab is from [here](https://github.com/bowenc0221/panoptic-deeplab/blob/master/tools_d2/README.md).
- Requirement for Oneformer is from [here](https://github.com/SHI-Labs/OneFormer).
- Download the Cityscape validation [here](https://mega.nz/folder/tS8QSaxL#5yhdfe9ogpKk18dRwX7WCw](https://www.cityscapes-dataset.com/downloads/)https://www.cityscapes-dataset.com/downloads/).

## Degrading Tools
#### Step 1- Noises Generation
run codes with the following cmd (noises aligned with .py names):

```
cd ./Dataset/Degrading
python Compression.py
python Motion_blur.py
python Raindrop/Raindrops.py
python ...
```

## More coming soon upon!

#### To-do List
- Share the D-Cityscapes+ dataset with 19 noise factors
- Share the noise generation table and tools
- Share the re-trained model on the new noisy data under adverse weather conditions

## Citation
If you find this code helpful in your research or wish to refer to the baseline results, please use the following BibTeX entry.

```BibTeX
@ARTICLE{11148081,
  author={Wang, Yiting and Zhao, Haonan and Gummadi, Daniel and Dianati, Mehrdad and Debattista, Kurt and Donzella, Valentina},
  journal={IEEE Transactions on Automation Science and Engineering}, 
  title={Robustness of Panoptic Segmentation for Degraded Automotive Cameras Data}, 
  year={2025},
  volume={22},
  number={},
  pages={21020-21032},
  doi={10.1109/TASE.2025.3605753}}

