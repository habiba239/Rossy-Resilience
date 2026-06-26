# Data Setup — Task 1 (Mammogram Segmentation)

Datasets are **not** included in this repository. Download and place them as follows:

## CBIS-DDSM (Primary Training)
Download from [The Cancer Imaging Archive](https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM).
Expected structure:
```
data/
└── CBIS-DDSM/
    ├── jpeg/
    └── csv/
        ├── mass_case_description_train_set.csv
        ├── mass_case_description_test_set.csv
        └── dicom_info.csv
```

## INbreast (Fine-Tuning)
Download from [Kaggle](https://www.kaggle.com/datasets/martholi/inbreast).
Expected structure:
```
data/
└── INbreast/
    ├── AllDICOMs/
    └── AllXML/
```

## BCDR (External Test Only)
Download from [bcdr.eu](https://bcdr.eu/). Place in:
```
data/
└── BCDR/
```
