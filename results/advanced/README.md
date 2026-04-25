# Advanced Post-hoc Analysis

This add-on analysis keeps the original training results unchanged and adds confidence + subgroup diagnostics.

## Calibration snapshot

- Model: **EffNet-B2 260px (accuracy-focused)**
- Test accuracy: **0.8614**
- Mean top-1 confidence: **0.8735**
- Expected calibration error (ECE): **0.0484**
- Multi-class Brier score: **0.2211**
- Negative log-likelihood: **0.5244**

## Melanoma recall trade-off

- EffNet-B2 260px (accuracy-focused): accuracy **0.8614**, melanoma recall **0.5403**
- EffNet-B2 260px (sensitivity-first): accuracy **0.5374**, melanoma recall **0.8548**

## Melanoma recall by sex

- `male` (n=91 melanoma cases): EffNet-B2 260px (accuracy-focused)=0.582, EffNet-B2 260px (sensitivity-first)=0.901
- `female` (n=33 melanoma cases): EffNet-B2 260px (accuracy-focused)=0.424, EffNet-B2 260px (sensitivity-first)=0.727
- `unknown` (n=0 melanoma cases): EffNet-B2 260px (accuracy-focused)=0.000, EffNet-B2 260px (sensitivity-first)=0.000

## Melanoma recall by age bucket

- `80+` (n=17 melanoma cases): EffNet-B2 260px (accuracy-focused)=0.412, EffNet-B2 260px (sensitivity-first)=0.882
- `40-59` (n=35 melanoma cases): EffNet-B2 260px (accuracy-focused)=0.514, EffNet-B2 260px (sensitivity-first)=0.971
- `60-79` (n=61 melanoma cases): EffNet-B2 260px (accuracy-focused)=0.607, EffNet-B2 260px (sensitivity-first)=0.803
- `<40` (n=11 melanoma cases): EffNet-B2 260px (accuracy-focused)=0.455, EffNet-B2 260px (sensitivity-first)=0.727
- `unknown` (n=0 melanoma cases): EffNet-B2 260px (accuracy-focused)=0.000, EffNet-B2 260px (sensitivity-first)=0.000

## Generated figures

- `advanced/reliability_effnetb2_260_acc.png`
- `advanced/confidence_hist_effnetb2_260_acc.png`
- `advanced/mel_recall_by_sex_compare.png`
- `advanced/mel_recall_by_age_compare.png`
